import torch
import os
import pandas as pd
import numpy as np
import torch.nn as nn
import torch.optim as optim
#from tqdm import tqdm as tq
import tqdm.notebook as tq
from torch.utils.data import DataLoader
from model import Yolocnv
from utility import AdjustedYoloLoss, MyDataset, resize, readable_stats, load_checkpoint, save_checkpoint


DEVICE = "cuda" if torch.cuda.is_available else "cpu"
BATCH_SIZE = 50
learning_rate = 0.0005 
weight_decay = 0.001

model = Yolocnv().to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

train_dataset = MyDataset('train/annotation.csv','train/Dataset','train/Labels', transform = resize)
test_dataset = MyDataset('test/annotation.csv','test/Dataset','test/Labels', transform = resize)

train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

stats_train = np.array([[0.,0.,0.,0.,0.,0.]])
stats_test = np.array([[0.,0.,0.,0.,0.,0.]]) 

tr, ts = training(1,8, l_under = 0.5, l_over = 0.5, l_coord = 2, trs = 0.3, run_train = True, run_tests = True)
stats_train = np.append(stats_train, tr, axis = 0)
stats_test = np.append(stats_test, ts, axis = 0)
tr, ts = training(9,15, l_under = 0.4, l_over = 0.6, l_coord = 2, trs = 0.3, run_train = True, run_tests = True)
stats_train = np.append(stats_train, tr, axis = 0)
stats_test = np.append(stats_test, ts, axis = 0)

torch.cuda.empty_cache()

save_checkpoint(model,optimizer,'model-name')



def training(epoch_start, epoch_end, l_under = 0.5, l_over = 0.5, l_coord = 5, trs = 0.3, run_train = True, run_tests = False, silent = False):
	loss_fn = AdjustedYoloLoss(l_under, l_over, l_coord, trs)

	stats_train = np.array([[0.,0.,0.,0.,0.,0.]])
	stats_test = np.array([[0.,0.,0.,0.,0.,0.]]) 

	for epoch in range(epoch_start,epoch_end+1):
		if not silent:
			print('\n\n','---- EPOCH ',epoch,'----')

		if run_train:
			if not silent:
				loop1 = tq.tqdm(train_loader, leave=True)
			else:
				loop1 = train_loader
			batch_stats1 = np.array([[0.,0.,0.,0.,0.,0.]])
			for x, y in loop1:
				x, y = x.to(DEVICE, dtype=torch.float), y.to(DEVICE, dtype=torch.float)
				out = model(x)
				loss = loss_fn(out, y)
				tpr, fdr, adj_tpr, adj_fdr, avg_iou = readable_stats(out,y,trs)

				batch_stats1 += np.array([[float(loss.item())*10,tpr,fdr,adj_tpr,adj_fdr,avg_iou]])

				optimizer.zero_grad()
				loss.backward()
				optimizer.step()

				loop1.set_postfix({'loss':float(loss.item()), 'TPR':adj_tpr, 'FDR':adj_fdr})

			mean_stats1 = batch_stats1/len(loop1)
			stats_train = np.append(stats_train, mean_stats1, axis = 0)
			if not silent:
				print(f'Average Loss: {mean_stats1[0][0]:.4f}, TPR: {mean_stats1[0][1]:.4f}, FDR: {mean_stats1[0][2]:.4f},')
				print(f'Adjusted TPR: {mean_stats1[0][3]:.4f}, Adjusted FDR: {mean_stats1[0][4]:.4f}, Average Adjusted IOU: {mean_stats1[0][5]:.4f}')


		if run_tests:
			model.eval()
			if not silent:
				loop2 = tq.tqdm(test_loader, leave=True)
			else:
				loop2 = test_loader
			batch_stats2 = np.array([[0.,0.,0.,0.,0.,0.]])

			with torch.no_grad():
				for x, y in loop2:
					x, y = x.to(DEVICE, dtype=torch.float), y.to(DEVICE, dtype=torch.float)
					out = model(x)
					loss = loss_fn(out, y)
					tpr, fdr, adj_tpr, adj_fdr, avg_iou = readable_stats(out,y,trs)

					batch_stats2 += np.array([[float(loss.item())*10,tpr,fdr,adj_tpr,adj_fdr,avg_iou]])

					loop2.set_postfix({'loss':float(loss.item()), 'TPR':adj_tpr, 'FDR':adj_fdr})

			mean_stats2 = batch_stats2/len(loop2)
			stats_test = np.append(stats_test, mean_stats2, axis = 0)
			if not silent:
				print(f'Average Loss: {mean_stats2[0][0]:.4f}, TPR: {mean_stats2[0][1]:.4f}, FDR: {mean_stats2[0][2]:.4f},')
				print(f'Adjusted TPR: {mean_stats2[0][3]:.4f}, Adjusted FDR: {mean_stats2[0][4]:.4f}, Average Adjusted IOU: {mean_stats2[0][5]:.4f}')
			model.train()

	return (stats_train, stats_test)