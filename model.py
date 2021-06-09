import torch
import torch.nn as nn

architecture = [
	('C',10,16,2,4),
	('M',2,2),
	('C',10,32,1,5),
	('M',3,3),
	('C',10,128,1,6),
	('M',5,5),
	('C',3,256,1,2),
	('M',10,10),
	('C',5,512,5,2),
	('C',1,128,1,0)
]



class Yolocnv(nn.Module):
	def __init__(self, in_channels=4, S = 50, B = 1, C = 2):
		super(Yolocnv , self).__init__()
		self.architecture = architecture
		self.in_channels = in_channels
		self.S = S
		self.B = B
		self.C = C
		self.convlayers = self._create_conv_layers(self.architecture)
		self.fcs = self._create_fcs()


	def forward(self, x):
		x = self.convlayers(x)

		return self.fcs(torch.flatten(x, start_dim=1))


	def _create_conv_layers(self, architecture):
		layers = []
		in_channels = self.in_channels

		for x in architecture:
			if x[0] == 'C':
				layers += [CNNBlock(in_channels, x[2], kernel_size=x[1], stride=x[3], padding=x[4])]
				in_channels = x[2]

			elif x[0] == 'M':
				layers += [nn.MaxPool1d(kernel_size=x[1], stride=x[2])]

		return nn.Sequential(*layers)


	def _create_fcs(self):
		return nn.Sequential(nn.Flatten(),
							 nn.Linear(self.S*128,4096),
							 nn.Dropout(0.2),
							 nn.LeakyReLU(0.1),
							 nn.Linear(4096,self.S*(self.C+self.B*3)))


class CNNBlock(nn.Module):
	def __init__(self, in_channels, out_channels, **kwargs):
		super(CNNBlock, self).__init__()
		self.conv = nn.Conv1d(in_channels, out_channels, bias=False, **kwargs)
		self.batchnorm = nn.BatchNorm1d(out_channels)
		self.leakyrelu = nn.LeakyReLU(0.1)

	def forward(self, x):
		return self.leakyrelu(self.batchnorm(self.conv(x)))