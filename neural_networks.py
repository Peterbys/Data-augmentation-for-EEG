# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.utils.data import Dataset

class trainData(Dataset):
    
    def __init__(self, X_data, y_data):
            self.X_data = X_data
            self.y_data = y_data
            
    def __getitem__(self, index):
            return self.X_data[index], self.y_data[index]
            
    def __len__ (self):
            return len(self.X_data)



class testData(Dataset):
        
    def __init__(self, X_data,y_data):
            self.X_data = X_data
            self.y_data = y_data
            
    def __getitem__(self, index):
            return self.X_data[index],self.y_data[index]
            
    def __len__ (self):
            return len(self.X_data)
        
class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class binaryClassification(nn.Module):
    def __init__(self,in_chan1,out_chan1,kernel_size1,l1,l2,l3):
        super(binaryClassification, self).__init__()        # Number of input features is 12.
        self.cnn_1 = nn.Conv1d(in_chan1, out_chan1, kernel_size1, stride=1, dilation=1, bias=True, padding_mode='zeros')
        self.layer_2 = nn.Linear(l1, l2)
        self.layer_3 = nn.Linear(l2,l3)
        self.flatten = nn.Flatten()
        self.layer_out = nn.Linear(l3, 1) 
        self.relu = nn.ReLU()
        self.dropout2 = nn.Dropout(p=0.25)
        self.batchnorm2 = nn.BatchNorm1d(l2)
        self.batchnorm3 = nn.BatchNorm1d(l3)
        
    def forward(self, inputs):
        x = self.cnn_1(inputs)
        x = self.flatten(x)
        x = self.relu(x)
        #print(x)
        x = self.layer_2(x)
        x = self.batchnorm2(x)
        x = self.relu(x)
        x = self.dropout2(x)
        x = self.layer_3(x)
        x = self.batchnorm3(x)
        x = self.relu(x)
        x = self.layer_out(x)

        return x


        
class EEGNet(nn.Module):
    def __init__(self,f1,f2,k1,d,ch1):
        super(EEGNet, self).__init__()
        self.T = 134
        
        # Layer 1
        self.conv1 = nn.Conv2d(1, f1, (1, k1), padding = (0,int((k1-1)/2)),bias=False)
        self.batchnorm1 = nn.BatchNorm2d(f1, False)
        
        # Layer 2
        self.conv2 = nn.Conv2d(f1,f1*d,(ch1,1),groups = f1,bias=False)
        self.batchnorm2 = nn.BatchNorm2d(d*f1, False)
        self.elu2 = nn.ELU()
        self.pooling2 = nn.AvgPool2d(1, 2)
        self.dropout2 = nn.Dropout(p=0.5)
        # Layer 3

        self.conv3_depth = nn.Conv2d(d*f1,d*f1,(1,16),groups=d*f1,padding = (0,int((16-1)/2)),bias=False)
        self.pointwise = nn.Conv2d(d*f1,f2,1,1,0,1,1,bias=False) 
        self.batchnorm3 = nn.BatchNorm2d(f2, False)
        self.elu3 = nn.ELU()
        self.pooling3 = nn.AvgPool2d((1,2))                
        self.dropout3 = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear((self.T-2)//(4)*f2, 1)
        

    def forward(self, x):
        # Layer 1
        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = self.conv2(x)
        x = self.batchnorm2(x)
        x = self.elu2(x)

        x = self.pooling2(x)
        
        x = self.dropout2(x)
        # Layer 2
        x = self.conv3_depth(x)
        x = self.pointwise(x)
        x = self.batchnorm3(x)
        x = self.elu3(x)
        x = self.pooling3(x)
        x = self.dropout3(x)

        # FC Layer
        x = torch.flatten(x,start_dim=1,end_dim=3)
        x = self.fc1(x)
        return x
    