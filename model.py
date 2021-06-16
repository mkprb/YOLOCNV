import torch
import torch.nn as nn

architecture = [
    [('C',50,32,2,24),
    ('M',2,2),
    
    ('C',20,64,5,8),
    ('M',5,5),
    
    ('C',1,32,1,0),
    ('C',10,64,1,4),
    ('C',1,64,1,0),
    ('C',10,128,1,5),
    ('M',5,5)],
    
    [('C',1,64,1,0),
    ('C',3,128,1,1),
    ('C',1,128,1,0),
    ('C',3,256,1,1),
    ('C',1,128,1,0)],
    
    [('M',6,6)]
]



class Yolocnv(nn.Module):
    def __init__(self, S = 50, B = 1, C = 2):
        super(Yolocnv , self).__init__()
        self.architecture = architecture
        #self.in_channels = in_channels
        self.S = S
        self.B = B
        self.C = C
        self.convlayers0 = self._create_conv_layers(self.architecture[0],4)
        self.convlayers1 = self._create_conv_layers(self.architecture[1],128)
        self.convlayers2 = self._create_conv_layers(self.architecture[2],128)
        self.norm = nn.BatchNorm1d(128)
        self.fcs = self._create_fcs()


    def forward(self, x):
        x = self.convlayers0(x)
        x_fine = self.convlayers1(x)
        x = self.norm(x + x_fine)
        x = self.convlayers2(x)
        x = self.fcs(torch.flatten(x, start_dim=1))
        return x


    def _create_conv_layers(self, architecture, in_channels):
        layers = []

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
                             nn.Dropout(0.5),
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