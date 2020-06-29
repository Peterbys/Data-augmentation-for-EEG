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
subject_dir = '/home/peter/my_subjects'
a = range(74)    
b = [x for i,x in enumerate(a) if i!=60 and i != 61 and i != 62]
subjects = ['04','06']
#subjects = ['01','04']
save = 0
fs = 1100/10
import mfocuss
#%%
def create_morph(data,subject_dir,fs,subject,subject2,fwd,fwd2,gain,gain2):
    datanew = np.zeros(data.shape)

    src_est = np.linalg.pinv(gain) @ (data[:,:,:])

    for i in range(data.shape[0]):
        print(i)
       # src_est = np.linalg.pinv(gain) @ (data[i,:,:]-np.mean(data[i,:,:],axis=1)[:,None])
        X,A,ind,q = mfocuss.Mfocuss(gain,data[i,:,:],0.8,70,0,72)
        Across = np.zeros(gain.shape)
        Across[:,ind] = gain[:,ind]
        Across = np.linalg.pinv(Across)
        src_est[i,:,:] = Across @  data[i,:,:]
        stc = mne.SourceEstimate(src_est[i,:,:],[fwd['src'][0]['vertno'],fwd['src'][1]['vertno']],tmin=0,tstep=1/fs)
        #stc = mne.SourceEstimate(src_est[i,:,:],[fwd['src'][0]['vertno'],fwd['src'][1]['vertno']],tmin=0,tstep=1/fs)
        morph = mne.compute_source_morph(stc, subject_from= 'sub-' + subject + '_free',
                                     subject_to = 'sub-' + subject2 + '_free',
                                     subjects_dir=subject_dir, spacing = [fwd2['src'][0]['vertno'],fwd2['src'][1]['vertno']])
        stc_new = morph.apply(stc)
        datanew[i,:,:] = gain2[:,:] @ stc_new.data
    return datanew

def gen_fwd_list(subject_dir,subjects,conductivity):
    fwd_list = {}
    for subject in subjects:
        src = mne.read_source_spaces(subject_dir + '/sub-' + subject + '_free' + '/sub-' + subject + '_free-src.fif')
        trans = subject_dir + '/sub-' + subject + '_free' + '/sub-' + subject + '_free-trans.fif'
        raw_fname = subject_dir + '/sub-' + subject +  '_free/meg/sub-' + subject + '_ses-meg_task-facerecognition_run-01_proc-sss_meg.fif'
        model = mne.make_bem_model(subject = 'sub-' + subject + '_free', ico=4,
                               conductivity=conductivity,
                               subjects_dir=subject_dir)
        bem = mne.make_bem_solution(model)
        fwd_list[subject]= mne.convert_forward_solution(mne.make_forward_solution(raw_fname, trans=trans, src=src, bem=bem,
                                    eeg=True,meg=False, mindist=5.0, n_jobs=2),force_fixed=True)
    return fwd_list
#%%
fwd_list = gen_fwd_list(subject_dir,subjects,[0.3,0.01,0.3])
#%%
fs = 1100/10
data_artificial_normal = {}
data_artificial_scrambled = {}
for subject in subjects:
    data_normal = np.load(subject_dir + '/data/EEG/sub-' + subject + '/normal.npy')[:,b,:]
    data_scrambled = np.load(subject_dir + '/data/EEG/sub-' + subject + '/scrambled.npy')[:,b,:]
    for subject2 in subjects:
        if subject != subject2 and subject != '06':
            print("Current: " + subject + " to " + subject2)
            data_artificial_normal[subject + '-to-' + subject2] = create_morph(data_normal, subject_dir, fs, subject, subject2, fwd_list[subject], fwd_list[subject2], fwd_list[subject]['sol']['data'][b,:], fwd_list[subject2]['sol']['data'][b,:])
            data_artificial_scrambled[subject + '-to-' + subject2] = create_morph(data_scrambled, subject_dir, fs, subject, subject2, fwd_list[subject], fwd_list[subject2], fwd_list[subject]['sol']['data'][b,:], fwd_list[subject2]['sol']['data'][b,:])
#%%
if save == 1 :
    pickle.dump( data_artificial_normal, open( "data-sub04050607-dnn/sub04to09-normal.pkl", "wb" ) )
    pickle.dump( data_artificial_scrambled, open( "data-sub04050607-dnn/sub04to09-scrambled.pkl", "wb" ) )
else:
    data_artificial_normal = pickle.load( open("data-sub04050607-dnn/sub04to09-normal.pkl","rb"))
    data_artificial_scrambled = pickle.load( open("data-sub04050607-dnn/sub04to09-scrambled.pkl","rb" ))
#%%
data_artificial_normal_red = {}
data_artificial_scrambled_red = {}
#for keys in list(data_artificial_normal.keys())[0:-12]:
    #data_artificial_normal_red[keys] = data_artificial_normal[keys]
#    data_artificial_scrambled_red[keys] = data_artificial_scrambled[keys]
subjects = ['04','05','06','07']
subjects_test = ['08','09']

for subject in subjects:
    for subject2 in subjects_test:
        if subject != subject2:
            data_artificial_normal_red[subject + '-to-' + subject2] = data_artificial_normal[subject + '-to-' + subject2]
            data_artificial_scrambled_red[subject + '-to-' + subject2] = data_artificial_scrambled[subject + '-to-' + subject2]
#%%
data_artificial_normal_red = {}
data_artificial_scrambled_red = {}
#for keys in list(data_artificial_normal.keys())[0:-12]:
    #data_artificial_normal_red[keys] = data_artificial_normal[keys]
#    data_artificial_scrambled_red[keys] = data_artificial_scrambled[keys]
subjects = ['01','04','05','06','07','08','09']

for subject in subjects:
    subject_rand = random.sample(subjects,4)
    for subject2 in subject_rand:
        if subject != subject2:
            data_artificial_normal_red[subject + '-to-' + subject2] = data_artificial_normal[subject + '-to-' + subject2]
            data_artificial_scrambled_red[subject + '-to-' + subject2] = data_artificial_scrambled[subject + '-to-' + subject2]
      
#%%
all_subjects = ['01','04','05','06','07','08','09']
k = 10
error_out = np.zeros(k)
error_in = np.zeros(k)
error_out_artificial = np.zeros(k)
error_in_artificial = np.zeros(k)
confusion_in = []
confusion_out = []
confusion_in_art = []
confusion_out_art = []
p = 0.2
data_artificial_normal = pickle.load( open("data-sub04050607-dnn/sub04to09-normal.pkl","rb"))
data_artificial_scrambled = pickle.load( open("data-sub04050607-dnn/sub04to09-scrambled.pkl","rb" ))
for trial in range(k):
    print("---------------------------------------------------------------")
    print("Current iter: " + str(trial))
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
    
    normal_ind_dict = get_data.get_training_ind(subjects_normal_dict,p)
    scrambled_ind_dict = get_data.get_training_ind(subjects_scrambled_dict,p)
    subject_scrambled_dict_train = {}
    subject_normal_dict_train = {}
    subject_scrambled_dict_test= {}
    subject_normal_dict_test = {}
    for subject in subjects:
        subject_scrambled_dict_train[subject] = subjects_scrambled_dict[subject][scrambled_ind_dict[subject]['train'],:,:] 
        subject_normal_dict_train[subject] = subjects_normal_dict[subject][normal_ind_dict[subject]['train'],:,:]
        subject_scrambled_dict_test[subject] = subjects_scrambled_dict[subject][scrambled_ind_dict[subject]['test'],:,:] 
        subject_normal_dict_test[subject] = subjects_normal_dict[subject] =subjects_normal_dict[subject][normal_ind_dict[subject]['test'],:,:]
        
    
    
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
            data_artificial_normal_red[subject + '-to-' + subject2] = data_artificial_normal[subject + '-to-' + subject2][normal_ind_dict[subject]['train'],:,:]
            data_artificial_scrambled_red[subject + '-to-' + subject2] = data_artificial_scrambled[subject + '-to-' + subject2][scrambled_ind_dict[subject]['train'],:,:] 
   
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
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=69)

X_train = X_art
y_train = y_art
#scaler = StandardScaler()
#X_train = scaler.fit_transform(X_train)
#X_test = scaler.transform(X_test)
#X_test_other_norm = scaler.transform(X_test_other)
#%%
EPOCHS =  200
BATCH_SIZE = 64
LEARNING_RATE = 0.001
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
            return self.X_data[index]
            
    def __len__ (self):
            return len(self.X_data)


class binaryClassification(nn.Module):
    def __init__(self,l1,l2,l3):
        super(binaryClassification, self).__init__()        # Number of input features is 12.
        self.layer_1 = nn.Linear(l1, l2) 
        self.layer_2 = nn.Linear(l2, l3)
        self.layer_out = nn.Linear(l3, 1) 
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.1)
        self.batchnorm1 = nn.BatchNorm1d(l2)
        self.batchnorm2 = nn.BatchNorm1d(l3)
        
    def forward(self, inputs):
        x = self.relu(self.layer_1(inputs))
        x = self.batchnorm1(x)
        x = self.relu(self.layer_2(x))
        x = self.batchnorm2(x)
        x = self.dropout(x)
        x = self.layer_out(x)
        
        return x

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#%%
def gen_weights(y_batch):
    weight_vector = Knp.zeros(len(y_batch))
    ones = np.where(y_batch == 1)
    zeros = np.where(y_batch == 0)
    weight_vector[ones] = 1-sum(y_batch)/len(y_batch)
    weight_vector[zeros] = sum(y_batch)/len(y_batch)
    return weight_vector
#%%
train_data = trainData(torch.FloatTensor(X_train), 
                       torch.FloatTensor(y_train))        

test_data = testData(torch.FloatTensor(X_test),torch.FloatTensor(y_test))
test_data_other = testData(torch.FloatTensor(X_test_other),torch.FloatTensor(y_test_other))
train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(dataset=test_data, batch_size=1)   
test_loader_other = DataLoader(dataset=test_data_other,batch_size=1)
model = binaryClassification(X_train.shape[1],int(X_train.shape[1]*2),int(X_train.shape[1]/4))
model.to(device)
print(model)
criterion = nn.BCEWithLogitsLoss(pos_weight=torch.FloatTensor([len(y_train)/sum(y_train)-1]).to(device))
#criterion = nn.BCEWithLogitsLoss()
#criterion = nn.BCEWithLogitsLoss(pos_weight=torch.FloatTensor([sum(y_train)/len(y_train)]).to(device))
#criterion = nn.BCEWithLogitsLoss(pos_weight=torch.FloatTensor([(len(y_train)-sum(y_train))/sum(y_train)]).to(device))

optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
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
for e in range(1, 50):   
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
        epoch_loss[e] += loss.item()
        epoch_acc[e] += acc.item()
    print(f'Epoch {e+0:03}: | Loss: {epoch_loss[e]/len(train_loader):.5f} | Acc: {epoch_acc[e]/len(train_loader):.3f}')
#%%

model.train()
epoch_loss = np.zeros(EPOCHS)
epoch_acc = np.zeros(EPOCHS)
error_out = np.zeros(EPOCHS)
acc_test = np.zeros(EPOCHS)

e_end = 1
#%%
for e in range(e_end, 100):
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
            y_test_pred = torch.sigmoid(y_test_pred)
            y_pred_tag = torch.round(y_test_pred)
            acc_test[e-1] += binary_acc(y_pred_tag, y_batch_test.unsqueeze(1)).item()/ len(y_test_other)
    
    e_end = e
    print(f'Epoch {e+0:03}: | Loss: {epoch_loss[e-1]:.5f} | Acc: {epoch_acc[e-1]:.3f}')

#%%
plt.plot(acc_test[0:e_end])
plt.plot()
plt.plot(epoch_acc[0:e_end])
    #%%
y_pred_list = []
model.eval()
with torch.no_grad():
    for X_batch_test in test_loader:
        X_batch_test = X_batch_test.to(device)
        y_test_pred = model(X_batch_test)
        y_test_pred = torch.sigmoid(y_test_pred)
        y_pred_tag = torch.round(y_test_pred)
        y_pred_list.append(y_pred_tag.cpu().numpy())
y_pred_list = [a.squeeze().tolist() for a in y_pred_list]
print("confusion_in:")
print(confusion_matrix(y_test, y_pred_list,normalize='true'))
error_in= np.sum(np.abs(y_pred_list-y_test))/y_test.shape[0]
#%%
model.eval()
with torch.no_grad():
    for X_batch,y_batch in train_loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch_test.to(device)
        y_test_pred = model(X_batch)
        y_test_pred = torch.sigmoid(y_test_pred)
        y_pred_tag = torch.round(y_test_pred)
        #print(y_pred_tag)

#%%
print("Confusion_matrix-in: ")
print(confusion_matrix(y_test, y_pred_list,normalize='true'))
print("Error-in: " + str(error_in))
print("Confusion_matrix-test: ")
print(confusion_matrix(y_test_other, y_pred_list_test,normalize='true'))
print("Error-test: " + str(error_out))
#%%
from sklearn.linear_model import LogisticRegression
import sklearn.preprocessing as preprocessing
X_standardizer = preprocessing.StandardScaler()
X_norm = X_standardizer.fit_transform(X)
X_test_norm = X_standardizer.transform(X_test_other)
Cval = 0.07
model = LogisticRegression(penalty = 'l1', C = Cval, tol = 1e-6, solver='saga', fit_intercept=True,class_weight = 'balanced',max_iter=600)
model = model.fit(X_norm,y)
confusion_in = confusion_matrix(y,model.predict(X_norm),normalize='true')
error= np.sum(np.abs(model.predict(X_test_norm)-y_test_other))/y_test_other.shape[0]
confusion = confusion_matrix(y_test_other,model.predict(X_test_norm),normalize='true')
print("Error:")
print(error)
print("Confusion-out:")
print(confusion)
print("Confusion-in:")
print(confusion_in)
#%%
Cval = 0.07
X_standardizer = preprocessing.StandardScaler()
X_norm = X_standardizer.fit_transform(X_artificial_comb)
X_test_norm = X_standardizer.transform(X_test_other)
model = LogisticRegression(penalty = 'l1', C = Cval, tol = 1e-6, solver='saga', fit_intercept=True,class_weight = 'balanced',max_iter=1200)
model = model.fit(X_norm,y_artificial_comb)
confusion_in_art = confusion_matrix(y_artificial_comb,model.predict(X_norm),normalize='true')
error_art= np.sum(np.abs(model.predict(X_test_norm)-y_test_other))/y_test_other.shape[0]
confusion_art = confusion_matrix(y_test_other,model.predict(X_test_norm),normalize='true')
print("Error-art:")
print(error_art)
print("Confusion-art-out:")
print(confusion_art)
print("Confusion-art-in: ")
print(confusion_in_art)
#%%
X_norm = X_standardizer.fit_transform(X)
X_test_norm = X_standardizer.transform(X_test_other)
X_norm = X_standardizer.fit_transform(X_artificial_comb)
X_test_norm_art = X_standardizer.transform(X_test_other)

#%%
import classification_models
classification_models.print_logistic_regression(X_artificial_comb,y_artificial_comb, 5 ,'artificial_test',7,0.01,0.5)
#%%
ch =50
seg = 4
orig = '04'
other = '06'
#%%
plt.plot(subjects_normal_dict[orig][seg,ch,:],label = orig)
plt.plot(subjects_normal_dict_test[other][seg,ch,:],label = other)
plt.plot(data_artificial_normal_red[orig + '-to-' + other][seg,ch,:],label = orig + '-to-' + other)
plt.legend()
#%%
plt.plot(np.mean(data_artificial_normal_red[orig + '-to-' + other][:,ch,:],axis=0),label=orig + "-to-" +other)
plt.plot(np.mean(subjects_normal_dict[orig][:,ch,:],axis=0),label=orig)
plt.plot(np.mean(subjects_normal_dict_test[other][:,ch,:],axis=0),label=other)
plt.legend()
#%%
plt.plot(np.mean(data_artificial_normal_red[orig +'-to-' + other][:,ch,:],axis=0)-np.mean(np.mean(data_artificial_normal_red[orig +'-to-'+ other][:,ch,:],axis=0)),label=orig + "-to-" + other)
plt.plot(np.mean(subjects_normal_dict[orig][:,ch,:],axis=0)-np.mean(np.mean(subjects_normal_dict[orig][:,ch,:],axis=0)),label=orig)
plt.plot(np.mean(subjects_normal_dict_test[other][:,ch,:],axis=0)-np.mean(np.mean(subjects_normal_dict_test[other][:,ch,:],axis=0)),label=other)
plt.legend()

#%%
data_artificial_normal_mfocuss = {}
data_artificial_scrambled_mfocuss = {}
for subject in subjects:
    data_normal = np.load(subject_dir + '/data/EEG/sub-' + subject + '/normal.npy')[:,b,:]
    data_scrambled = np.load(subject_dir + '/data/EEG/sub-' + subject + '/scrambled.npy')[:,b,:]
    for subject2 in subjects:
        if subject != subject2 and subject != '06':
            print("Current: " + subject + " to " + subject2)
            data_artificial_normal_mfocuss[subject + '-to-' + subject2] = create_morph(data_normal[1:10,:,:], subject_dir, fs, subject, subject2, fwd_list[subject], fwd_list[subject2], fwd_list[subject]['sol']['data'][b,:], fwd_list[subject2]['sol']['data'][b,:])
            #data_artificial_scrambled_mfocuss[subject + '-to-' + subject2] = create_morph(data_scrambled[1:10,:,:], subject_dir, fs, subject, subject2, fwd_list[subject], fwd_list[subject2], fwd_list[subject]['sol']['data'][b,:], fwd_list[subject2]['sol']['data'][b,:])
            
#%%
seg = 3
plt.plot(data_artificial_normal_red[orig + '-to-' + other][seg,ch,:],label = orig + '-to-' + other)
plt.plot(data_artificial_normal_mfocuss[orig + '-to-' + other][seg,ch,:],label = orig + '-mfocuss-' + other)
plt.plot(subjects_normal_dict[orig][seg,ch,:],label = orig)
plt.plot(subjects_normal_dict_test[other][seg,ch,:],label = other)
plt.legend()
#%%
#plt.plot(np.mean(data_artificial_normal_mfocuss[orig +'-to-' + other][:,ch,:],axis=0)-np.mean(np.mean(data_artificial_normal_mfocuss[orig +'-to-'+ other][:,ch,:],axis=0)),label=orig + "-to-" + other)
plt.plot(np.mean(subjects_normal_dict[orig][:,ch,:],axis=0)-np.mean(np.mean(subjects_normal_dict[orig][:,ch,:],axis=0)),label=orig)
plt.plot(np.mean(subjects_normal_dict_test[other][:,ch,:],axis=0)-np.mean(np.mean(subjects_normal_dict_test[other][:,ch,:],axis=0)),label=other)
plt.legend()
#%%
plt.plot(np.mean(subjects_normal_dict['04'][:,ch,:],axis=0))
plt.plot(np.mean(subjects_scrambled_dict['04'][:,ch,:],axis=0))
#%%
plt.plot(np.mean(subjects_normal_dict['05'][:,ch,:],axis=0))
plt.plot(np.mean(subjects_scrambled_dict['05'][:,ch,:],axis=0))
#%%
plt.plot(np.mean(subjects_normal_dict['06'][:,ch,:],axis=0))
plt.plot(np.mean(subjects_scrambled_dict['06'][:,ch,:],axis=0))