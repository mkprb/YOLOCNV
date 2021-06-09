import torch
import os
import pandas as pd
import numpy as np
import torch.nn as nn
#from tqdm import tqdm as tq
import tqdm.notebook as tq

def resize(data, new_size=150000):
	if data.shape[0] > new_size:
		data = np.delete(data,np.arange(0,data.shape[0],1/(1-new_size/data.shape[0])).astype(int),0)
	elif data.shape[0] < new_size:
		l = np.arange(0,data.shape[0],1/(new_size/data.shape[0]-1)).astype(int)
		data = np.insert(data, l, data[l])      
	return(data)


def iou(boxes_preds, boxes_labels):

	box1_x1 = boxes_preds[..., 0:1] - boxes_preds[..., 1:2] / 2
	box1_x2 = boxes_preds[..., 0:1] + boxes_preds[..., 1:2] / 2
	box2_x1 = boxes_labels[..., 0:1] - boxes_labels[..., 1:2] / 2
	box2_x2 = boxes_labels[..., 0:1] + boxes_labels[..., 1:2] / 2   
	x1 = torch.max(box1_x1, box2_x1)
	x2 = torch.min(box1_x2, box2_x2)

	intersection = (x2-x1).clamp(0)
	box1_a = abs(box1_x2 - box1_x1)
	box2_a = abs(box2_x2 - box2_x1)
	iou = intersection/(box1_a+box2_a-intersection+1e-6)

	return iou


class MyDataset(torch.utils.data.Dataset):
	def __init__(self, annotation_file, data_dir,label_dir, S = 50 , B = 1, C = 2, transform = None):
		self.annotations = pd.read_csv(annotation_file,header = None)
		self.data_dir = data_dir
		self.label_dir = label_dir
		self.S=S
		self.B=B
		self.C=C
		self.transform = transform

	def __len__(self):
		return len(self.annotations)

	def __getitem__(self,index):
		torch.set_printoptions(precision = 5, sci_mode = True)
		label_path = os.path.join(self.label_dir, self.annotations.iloc[index, 1])
		boxes = []
		with open(label_path) as f:
			for label in f.readlines():
				label = label.split()[1:-1]
				x1, x2, class_label = [float(x) for x in label]
				boxes.append([class_label,(x2+x1)/2,x2-x1]) # x from 0 to 1

		boxes = torch.tensor(boxes)
		data_path = os.path.join(self.data_dir, self.annotations.iloc[index, 0])
		data = pd.read_csv(data_path, sep = ' ',header = None).to_numpy()
		if self.transform:
			data = self.transform(data); 
		data = torch.from_numpy(np.transpose(data))

		label_matrix = torch.zeros((self.S, self.C + 3 * self.B))
		for box in boxes:
			class_label, x, length = box.tolist()
			i = int(self.S * x) #cell
			x_cell = self.S*x - i
			l_cell =self.S*length
			if i >=50:
					i = 49

			if label_matrix[i,2]==0:
				label_matrix[i,2]=1

			box_coord=torch.tensor([x_cell,l_cell])
			label_matrix[i, 3:5] = box_coord
			label_matrix[i, int(class_label)-1] = 1

		return data, label_matrix


class AdjustedYoloLoss(nn.Module):
	def __init__(self, l_noobj_under_tr = 0.5, l_noobj_over_tr = 0.5, l_coord = 5, threshold=0.3, S=50, B=1, C=2):
		super(AdjustedYoloLoss, self).__init__()
		self.mse = nn.MSELoss(reduction="mean")
		self.threshold = threshold
		self.lambda_noobj_under_tr = l_noobj_under_tr
		self.lambda_noobj_over_tr = l_noobj_over_tr
		self.lambda_coord = l_coord
		self.S = S
		self.B = B
		self.C = C


	def forward(self, prediction, target):
		#shape (N, S, C + 3*B)

		prediction = prediction.reshape(-1, self.S, self.C + self.B * 3)
		prediction_over_tr = prediction*((prediction[...,2:3] > self.threshold).float())
		prediction_under_tr = prediction*((prediction[...,2:3] <= self.threshold).float()) 
		idf = target[..., 2].unsqueeze(-1)

		box_prediction = idf * prediction[..., 3:5]
		box_target = idf * target[..., 3:5]

		box_prediction[..., 1] = torch.sign(box_prediction[..., 1]) * torch.sqrt(torch.abs(box_prediction[..., 1] + 1e-6))
		box_target[..., 1] = torch.sqrt(box_target[..., 1])

		box_loss = self.mse(torch.flatten(box_prediction, end_dim=-1), torch.flatten(box_target, end_dim=-1))
		object_loss = self.mse(torch.flatten(idf[..., 0] * prediction[..., 2]), torch.flatten(idf[..., 0] * target[..., 2]))
		no_object_loss_under_tr = self.mse(torch.flatten((1 - idf[..., 0]) * prediction_under_tr[..., 2]), torch.flatten((1 - idf[..., 0]) * target[..., 2]))
		no_object_loss_over_tr = self.mse(torch.flatten((1 - idf[..., 0]) * prediction_over_tr[..., 2]), torch.flatten((1 - idf[..., 0]) * target[..., 2]))
		class_loss = self.mse(torch.flatten(idf * prediction[..., :2], end_dim=-1), torch.flatten(idf * target[..., :2], end_dim=-1))

		loss = self.lambda_coord * box_loss + object_loss + self.lambda_noobj_under_tr * no_object_loss_under_tr + self.lambda_noobj_over_tr * no_object_loss_over_tr + class_loss  
		return loss


def readable_stats(p, t, threshold = 0.3, S=50, B=1, C=2):
	p = p.reshape(-1, S, C + B * 3)
	tp = float((t[...,0:2] * p[...,0:2]).sum())
	tn = float(((1-t[...,0:2]) * (1-p[...,0:2])).sum())
	fp = float(((1-t[...,0:2]) * p[...,0:2]).sum())
	fn = float((t[...,0:2] * (1-p[...,0:2])).sum())

	tpr = tp/(tp+fn+1e-6)
	fdr = fp/(fp+tp+1e-6)

	p = p*((p[...,2:3] > threshold).float())

	tp = float((t[...,0:2] * p[...,0:2]).sum())
	tn = float(((1-t[...,0:2]) * (1-p[...,0:2])).sum())
	fp = float(((1-t[...,0:2]) * p[...,0:2]).sum())
	fn = float((t[...,0:2] * (1-p[...,0:2])).sum())

	a_tpr = tp/(tp+fn+1e-6)
	a_fdr = fp/(fp+tp+1e-6)

	avg_iou = float(iou(p[...,3:5],t[...,3:5]).sum()/(tp+1e-6))

	return tpr, fdr, a_tpr, a_fdr, avg_iou


def cachemall(annotation_dir, data_dir, label_dir):
	annotation = pd.read_csv(annotation_dir,header = None)
	for index in tq.tqdm(range(len(annotation))):
		label_path= os.path.join(label_dir, annotation.iloc[index, 1])
		data_path = os.path.join(data_dir, annotation.iloc[index, 0])
   
		with open(label_path) as f:
			for label in f.readlines():
				label = label.split()[1:-1]
		data = pd.read_csv(data_path, sep = ' ',header = None).to_numpy()


def load_checkpoint(model,optimizer,PATH):
	checkpoint = torch.load(PATH)
	model.load_state_dict(checkpoint['model_state_dict'])
	if optimizer:
		optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
	model.train()


def save_checkpoint(model,optimizer,PATH):
	torch.save({'model_state_dict': model.state_dict(),
				'optimizer_state_dict': optimizer.state_dict()},
				PATH)