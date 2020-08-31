#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 20 19:53:46 2020

@author: peter
"""


"""
Created on Wed Jun 17 15:18:19 2020

@author: peter
"""


# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import StandardScaler    
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import mne 
import numpy as np
import get_data
import os
import re
import pickle 
import random
from sklearn.linear_model import LogisticRegression
import sklearn.preprocessing as preprocessing
subject_dir = '/home/peter/my_subjects'
import scipy.signal as sig
a = range(74)    
b = [x for i,x in enumerate(a) if i!=60 and i != 61 and i != 62]
#subjects = ['04','06']
#subjects = ['01','04']
save = 0
fs = 1100/10# -*- coding: utf-8 -*-
from scipy.fft import fft, ifft
#%%

all_subjects = ['01','04','05','06','07','08','09']
k = 1
error_out = np.zeros(k)
error_in = np.zeros(k)
error_out_artificial = np.zeros(k)
error_in_artificial = np.zeros(k)
confusion_in = []
confusion_out = []
confusion_in_art = []
confusion_out_art = []
p = 0.2
seed_trainsplit = 0
seed_subjectsplit = 0
data_artificial_normal = pickle.load( open("data-sub04050607-dnn/sub04to09-normal.pkl","rb"))
data_artificial_scrambled = pickle.load( open("data-sub04050607-dnn/sub04to09-scrambled.pkl","rb" ))

random.seed(seed_subjectsplit)
subjects = random.sample(all_subjects,5)
subjects_test = []
for subject in all_subjects:
    if subject not in subjects:
        subjects_test.append(subject)
subjects_normal_dict = {}
subjects_scrambled_dict = {}
subject_dir = '/home/peter/my_subjects'
for subject in subjects:
    subjects_scrambled_dict[subject] =  np.load(subject_dir + '/data/EEG/sub-' + subject + '/scrambled.npy')[:,b,:]
    #subjects_normal_dict[subject] = np.load(subject_dir + '/data/EEG/sub-' + subject + '/normal.npy')[0:subjects_scrambled_dict[subject].shape[0],b,:]
    subjects_normal_dict[subject] = np.load(subject_dir + '/data/EEG/sub-' + subject + '/normal.npy')[:,b,:]

normal_ind_dict = get_data.get_training_ind(subjects_normal_dict,p,seed_trainsplit)
scrambled_ind_dict = get_data.get_training_ind(subjects_scrambled_dict,p,seed_trainsplit)
subject_scrambled_dict_train = {}
subject_normal_dict_train = {}
subject_scrambled_dict_test= {}
subject_normal_dict_test = {}
for subject in subjects:
    subject_scrambled_dict_train[subject] = subjects_scrambled_dict[subject][scrambled_ind_dict[subject]['train'],:,:] 
    subject_normal_dict_train[subject] = subjects_normal_dict[subject][normal_ind_dict[subject]['train'],:,:]
    subject_scrambled_dict_test[subject] = subjects_scrambled_dict[subject][scrambled_ind_dict[subject]['test'],:,:] 
    subject_normal_dict_test[subject] = subjects_normal_dict[subject] =subjects_normal_dict[subject][normal_ind_dict[subject]['test'],:,:]
    subject_normal_dict_test[subject] = subject_normal_dict_test[subject][0:subject_scrambled_dict_test[subject].shape[0],:,:]    


subject_normal_dict_test_other = {}
subject_scrambled_dict_test_other = {}
for subject in subjects_test:
    subject_scrambled_dict_test_other[subject] =  np.load(subject_dir + '/data/EEG/sub-' + subject + '/scrambled.npy')[:,b,:]
    subject_normal_dict_test_other[subject] = np.load(subject_dir + '/data/EEG/sub-' + subject + '/normal.npy')[0:subject_scrambled_dict_test_other[subject].shape[0],b,:]
    #subjects_normal_dict_test[subject] = np.load(subject_dir + '/data/EEG/sub-' + subject + '/normal.npy')[0:,b,:]
data_artificial_normal_red = {}
data_artificial_scrambled_red = {}
for subject in subjects:
    for subject2 in subjects_test:
        #subject2 = random.sample(subjects_test,1)[0]
        data_artificial_normal_red[subject + '-to-' + subject2] = data_artificial_normal[subject + '-to-' + subject2][normal_ind_dict[subject]['train'],:,:]
        data_artificial_scrambled_red[subject + '-to-' + subject2] = data_artificial_scrambled[subject + '-to-' + subject2][scrambled_ind_dict[subject]['train'],:,:] 
        
    
#%% 
'''
ch = 50
#plt.plot(np.mean(morph_ds[:,ch,:],axis=0))
#plt.plot(sig.decimate(np.mean(morph_nods[:,ch,:],axis=0),q=downsample))
plt.plot(np.log(np.abs(fft(np.mean(data_artificial_normal['01-to-04'][:200,ch,:],axis=0)))))
plt.plot(np.log(np.abs(fft(np.mean(subjects_normal_dict['04'][:200,ch,:],axis=0)))))
#%%
plt.phase_spectrum(np.mean(data_artificial_normal['09-to-04'][:400,ch,:],axis=0))
plt.phase_spectrum(np.mean(subjects_normal_dict['41'][:400,ch,:],axis=0))
'''
#%%
deduct_mean = 1 
X_train_normal = get_data.gen_vectorized_data(subject_normal_dict_train,deduct_mean)
X_train_scrambled = get_data.gen_vectorized_data(subject_scrambled_dict_train,deduct_mean)
X_test_normal = get_data.gen_vectorized_data(subject_normal_dict_test,deduct_mean)
X_test_scrambled = get_data.gen_vectorized_data(subject_scrambled_dict_test,deduct_mean)
X_test_other_normal = get_data.gen_vectorized_data(subject_normal_dict_test_other,deduct_mean)
X_test_other_scrambled = get_data.gen_vectorized_data(subject_scrambled_dict_test_other,deduct_mean)
X_artificial_normal = get_data.gen_vectorized_data(data_artificial_normal_red,deduct_mean)
X_artificial_scrambled = get_data.gen_vectorized_data(data_artificial_scrambled_red,deduct_mean)


X = np.concatenate((X_train_normal,X_train_scrambled))
y = np.concatenate((np.ones(X_train_normal.shape[0]),np.zeros(X_train_scrambled.shape[0])))
X_test = np.concatenate((X_test_normal,X_test_scrambled))
y_test = np.concatenate((np.ones(X_test_normal.shape[0]),np.zeros(X_test_scrambled.shape[0])))
X_test_other = np.concatenate((X_test_other_normal,X_test_other_scrambled))
y_test_other = np.concatenate((np.ones(X_test_other_normal.shape[0]),np.zeros(X_test_other_scrambled.shape[0])))
X_art = np.concatenate((X_artificial_normal,X_artificial_scrambled))
X_art = np.concatenate((X,X_art))
y_art = np.concatenate((np.ones(X_artificial_normal.shape[0]),np.zeros(X_artificial_scrambled.shape[0])))
y_art = np.concatenate((y,y_art))
del X_train_normal, X_train_scrambled, X_test_normal, X_test_scrambled
del X_test_other_normal, X_test_other_scrambled, X_artificial_normal, X_artificial_scrambled 



#%%
aug = 1
if aug:  
    X_train = X_art
    y_train = y_art
    train_label = "Train-aug"
else:
    train_label = "Train-noaug"
    X_train = X
    y_train = y
del X,y,X_art,y_art
X_train =  np.reshape(X_train,[X_train.shape[0],71,int(X_train.shape[1]/71)])
X_test_other = np.reshape(X_test_other,[X_test_other.shape[0],71,int(X_test_other.shape[1]/71)])
X_test = np.reshape(X_test,[X_test.shape[0],71,int(X_test.shape[1]/71)])
#%%
EPOCHS =  2000
#BATCH_SIZE = 256   
BATCH_SIZE = 512
#LEARNING_RATE = 0.000005
LEARNING_RATE = 0.000003
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
        self.cnn_1 = nn.Conv1d(in_chan1, out_chan1, kernel_size1, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros')
        self.layer_2 = nn.Linear(l1, l2)
        self.layer_3 = nn.Linear(l2,l3)
        self.flatten = nn.Flatten()
        self.layer_out = nn.Linear(l3, 1) 
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.25)
        self.batchnorm1 = nn.BatchNorm1d(l2)
        self.batchnorm2 = nn.BatchNorm1d(l3)
        
    def forward(self, inputs):
        x = self.cnn_1(inputs)
        x = self.relu(x)
        x = self.flatten(x)
        #print(x)

        x = self.layer_2(x)
        x = self.batchnorm1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.layer_3(x)
        x = self.batchnorm2(x)
        x = self.relu(x)
        x = self.layer_out(x)

        return x
        

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#%%
train_data = trainData(torch.FloatTensor(X_train), 
                       torch.FloatTensor(y_train))        

test_data = testData(torch.FloatTensor(X_test),torch.FloatTensor(y_test))
test_data_other = testData(torch.FloatTensor(X_test_other),torch.FloatTensor(y_test_other))
train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(dataset=test_data, batch_size=1)   
test_loader_other = DataLoader(dataset=test_data_other,batch_size=1)
seed = 2
torch.manual_seed(seed)
chans_in1 = 71
#chans_out1 = 10
chans_out1 = 20
kernel_size1 = 40
l1 = (105-kernel_size1+1)*chans_out1
l2 = 100
l3 = 10
model = binaryClassification(chans_in1,chans_out1,kernel_size1,l1,l2,l3)
#model(torch.FloatTensor(subject_normal_dict_train['04'][:15,:,:])).shape
#%%
model.to(device)
print(model)
criterion = nn.BCEWithLogitsLoss(pos_weight=torch.FloatTensor([len(y_train)/sum(y_train)-1]).to(device))
#criterion = nn.BCEWithLogitsLoss()
#criterion = nn.BCEWithLogitsLoss(pos_weight=torch.FloatTensor([sum(y_train)/len(y_train)]).to(device))
#criterion = nn.BCEWithLogitsLoss(pos_weight=torch.FloatTensor([(len(y_train)-sum(y_train))/sum(y_train)]).to(device))

#optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE,weight_decay = 0.08)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE,weight_decay = 0.05)
def binary_acc(y_pred, y_test):
    y_pred_tag = torch.round(torch.sigmoid(y_pred))
    
    correct_results_sum = (y_pred_tag == y_test).sum().float()
    acc = correct_results_sum/y_test.shape[0]
    acc = torch.round(acc * 100)
    
    return acc

#%%
model.train()
epoch_loss = np.zeros(EPOCHS)
epoch_acc = np.zeros(EPOCHS)
error_out = np.zeros(EPOCHS)
acc_test_in = np.zeros(EPOCHS)
acc_test_out = np.zeros(EPOCHS)
e_end = 1
#%%
for e in range(e_end, EPOCHS):
    for X_batch, y_batch in train_loader:
        model.train()
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        y_pred = model(X_batch)
        loss = criterion(y_pred, y_batch.unsqueeze(1))
        acc = binary_acc(y_pred, y_batch.unsqueeze(1))
        loss.backward()
        optimizer.step()
        model.eval()
        epoch_loss[e-1] += loss.item()/len(train_loader)
        epoch_acc[e-1] += acc.item()/ len(train_loader)
        
    model.eval()
    with torch.no_grad():
        for X_batch_test,y_batch_test in test_loader_other:
            X_batch_test = X_batch_test.to(device)
            y_batch_test = y_batch_test.to(device)
            y_test_pred = model(X_batch_test)
           # y_test_pred = torch.sigmoid(y_test_pred)
           # y_pred_tag = torch.round(y_test_pred)
            acc_test_out[e-1] += binary_acc(y_test_pred, y_batch_test.unsqueeze(1)).item()/ len(y_test_other)
    model.eval()
    with torch.no_grad():
        for X_batch_test,y_batch_test in test_loader:
            X_batch_test = X_batch_test.to(device)
            y_batch_test = y_batch_test.to(device)
            y_test_pred = model(X_batch_test)
            #y_test_pred = torch.sigmoid(y_test_pred)
            #y_pred_tag = torch.round(y_test_pred)
            acc_test_in[e-1] += binary_acc(y_test_pred, y_batch_test.unsqueeze(1)).item()/ len(y_test)
    
    e_end = e
    print(f'Epoch {e+0:03}: | Loss: {epoch_loss[e-1]:.5f} | Acc: {epoch_acc[e-1]:.3f}') 
    '''
    if e % 10 == 0: 
        plt.plot(epoch_acc[:e_end], label = "Train")
        plt.plot(acc_test[:e_end],label = "Other-subj")
        plt.xlabel("Trial[n]")
        plt.ylabel("Acc[%]")
        plt.legend()
    '''   
#%%

plt.plot(epoch_acc[:e_end], label = train_label)
plt.plot(acc_test_out[:e_end],label = "Other-subj")
plt.plot(acc_test_in[:e_end],label = "In-subj")
plt.xlabel("Trial[n]")
plt.ylabel("Acc[%]")
plt.legend()
plt.ylim([50, 100])
plt.show()
plt.plot(acc_test_out[:e_end-1]-acc_test_in[:e_end-1])

#%%
plt.plot(np.log(np.abs(fft(model.cnn_1.weight.cpu()[0,0,:].detach().numpy()))))