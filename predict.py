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
model = Yolocnv().to(DEVICE)

predict_dataset = PredictDataset() #ToDo
predict_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
load_checkpoint(model,None,'path')
model.eval()

result = pd.DataFrame()

if not silent:
	loop = tq.tqdm(prepict_loader, leave=True)
else:
	loop = predict_loader
with torch.no_grad():
	for x, _ in loop:
		x = x.to(DEVICE, dtype=torch.float)
		out = model(x)
		result = process(result,out)


print(result)