import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import sklearn.metrics
import os
os.environ['ETS_TOOLKIT'] = 'qt4'
os.environ['QT_API'] = 'pyqt5'

global subject_dir
subject_dir = 'D:/'
import mne
from mne.minimum_norm import make_inverse_operator, prepare_inverse_operator
face_normal_event = [5,6,7,13,14,15]
face_scrambled_event = [17,18,19]
from mne.minimum_norm.inverse import _assemble_kernel
from helper_functions import *
from neural_networks import *


#%%
# -*- coding: utf-8 -*-

all_subjects = ['01','02','03','04','05','06','07','08','09','10','11','12','13','14','15','16']
data = gen_data_dict(all_subjects,6)
save = 0    
if save == 1:
    for subject1 in all_subjects:
        for j in range(1,7):
            raw = get_raw(subject1,j)
            data[subject1][j]['events']  = get_events(raw)
            data[subject1][j]['normal'], data[subject1][j]['cov'] = get_epochs(raw,data[subject1][j]['events'],10)
            data[subject1][j]['scrambled'] = get_epochs(raw,data[subject1][j]['events'],11)[0]

    pickle.dump( data, open( "data_dsfactor9.pkl", "wb" ) )
else:
    data = pickle.load( open("data_dsfactor9.pkl","rb" ))

save_fwd = 0
if save_fwd == 1:
    conductivity = [0.3,0.01,0.3]   
    fwd_list = gen_fwd_list(subject_dir,all_subjects,conductivity,data['01'][1]['normal'].info)
    pickle.dump( fwd_list, open( "fwd_list.pkl", "wb" ) )
else:
    fwd_list = pickle.load( open( "fwd_list.pkl", "rb" ) )

#%%

#%%
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

depth = 0.0
loose = 0.0
pick_ori = None
save_fwd = 0
if save_fwd == 1:
    conductivity = [0.3,0.01,0.3]   
    fwd_list = gen_fwd_list(subject_dir,all_subjects,conductivity,data['01'][1]['normal'].info)
    pickle.dump( fwd_list, open( "fwd_list.pkl", "wb" ) )
else:
    fwd_list = pickle.load( open( "fwd_list.pkl", "rb" ) )
runs =6
subjects_train_list = []
subjects_test_list = []
pinv = lambda x : np.linalg.pinv(x)
for i in range(0,1):
    data = pickle.load( open("data_dsfactor9.pkl","rb" ))
    mean = 1 
    num_this = 6
    num_train = 4
    num_aug = 2
    np.random.seed(i)
    this_subjects = np.random.choice(all_subjects,num_this,replace=False)
    subjects_train = np.random.choice(this_subjects,num_train,replace=False)
    subjects_test = list(set(this_subjects)-set(subjects_train))
    data_train,data_val = split_data(data,subjects_train,0.8,i,runs)
    data_other = {}
    for subject in subjects_test:
        data_other[subject] = data[subject]
    subjects_train_list.append(subjects_train)
    subjects_test_list.append(subjects_test)
    data_artificial_MNE = gen_artdata_dict(subjects_train,subjects_test,runs)
    snr = 0.5
    lambda2 = 1/snr
    method = "MNE" 
    
        
    for subject1 in subjects_train:
        evoked_tmp = data_train[subject1][1]['normal']
        fwd1 = fwd_list[subject1]
        fwd_fixed1 = mne.convert_forward_solution(fwd1,force_fixed=True)
        G1 = (np.eye(71)-(np.ones((71,1)) @ np.ones((71,1)).T) /(np.ones((71,1)).T @ np.ones((71,1)))) @ fwd_fixed1['sol']['data']
        noise_cov = data_train[subject1][1]['cov']
        inverse_operator = make_inverse_operator(evoked_tmp.info,fwd_fixed1,noise_cov,loose=loose,depth=depth)
        inverse_operator = prepare_inverse_operator(inverse_operator,1,lambda2,method,None,False)
        K,noise_norm,vertno,source_nn = _assemble_kernel(inverse_operator,None,method,pick_ori = pick_ori,use_cps=True)
        for subject2 in np.random.choice(subjects_test,num_aug,replace=False):
            fwd2 = fwd_list[subject2]
            fwd_fixed2 = mne.convert_forward_solution(fwd2,force_fixed=True)
            G2 = (np.eye(71)-(np.ones((71,1)) @ np.ones((71,1)).T) /(np.ones((71,1)).T @ np.ones((71,1)))) @ fwd_fixed2['sol']['data']
            src = mne.SourceEstimate(pinv(G1) @ evoked_tmp.average().data,[fwd1['src'][0]['vertno'],fwd1['src'][1]['vertno']],tmin=0,tstep=1/(1100*9),subject = 'sub-' + subject1 + '_free')
            morph =mne.compute_source_morph(src,subject_from ='sub-' +  subject1 + '_free', subject_to ='sub-' +  subject2 + '_free',subjects_dir = subject_dir, spacing = [fwd2['src'][0]['vertno'],fwd2['src'][1]['vertno']])
            for j in range(1,runs+1):
                evoked = data_train[subject1][j]['normal'] 
               
                if(j  == 4):
                    noise_cov = data_train[subject1][4]['cov']
                    inverse_operator = make_inverse_operator(evoked.info,fwd_fixed1,noise_cov,loose=0,depth=0)
                    inverse_operator = prepare_inverse_operator(inverse_operator,1,lambda2,method,None,False)
                    K,noise_norm,vertno,source_nn = _assemble_kernel(inverse_operator,None,method,pick_ori =pick_ori,use_cps=True)
                data_artificial_MNE[subject1][subject2][j]['normal'] =  G2 @ morph.morph_mat @ K @ evoked.get_data()
                data_artificial_MNE[subject1][subject2][j]['normal'] *= norm_scale(evoked,data_artificial_MNE[subject1][subject2][j]['normal'],mean)
                
                
                evoked = data_train[subject1][j]['scrambled'] 
        
                data_artificial_MNE[subject1][subject2][j]['scrambled'] = G2 @ morph.morph_mat @ K @ evoked.get_data()
                data_artificial_MNE[subject1][subject2][j]['scrambled'] *= norm_scale(evoked,data_artificial_MNE[subject1][subject2][j]['scrambled'],mean)
    
    for aug in range(0,1):
        X_train = dict_concat(data_train,subjects_train,runs)
           
        X_train, y_train = dict_to_data(X_train)
        X_val = dict_concat(data_val,subjects_train,runs)
        X_val, y_val = dict_to_data(X_val)
        X_other = dict_concat(data_other,subjects_test,runs)
        X_other,y_other = dict_to_data(X_other)
        
        
        if aug == 1:
            X_art2 = dict_art_concat(data_artificial_MNE,subjects_train,subjects_test,runs,0)
            X_art2,y_art2 = dict_to_data(X_art2)
            X_train = np.concatenate((X_train,X_art2))
            y_train = np.concatenate((y_train,y_art2))
            del data
        X_train = X_train - np.mean(X_train,axis=2)[:,:,None]
        
        X_val= X_val - np.mean(X_val,axis=2)[:,:,None]
        X_other = X_other -np.mean(X_other,axis=2)[:,:,None]
        
            
        
        
        def binary_acc(y_pred, y_test):
            y_pred_tag = torch.round(torch.sigmoid(y_pred))
            
            correct_results_sum = (y_pred_tag == y_test).sum().float()
            acc = correct_results_sum/y_test.shape[0]
            acc = torch.round(acc * 100)
            
            return acc
        seed = 0
        torch.manual_seed(seed)
        my_model = 'generic'
        if my_model == 'generic':
            chans_in1 = 71
            chans_out1 = 5
            #LEARNING_RATE = 0.0000015
           # LEARNING_RATE = 0.00008
            LEARNING_RATE = 0.0000002
            kernel_size1 = 50
            l1 = (134-kernel_size1+1)*chans_out1
            l2 = 100
            l3 = 10
            model = binaryClassification(chans_in1,chans_out1,kernel_size1,l1,l2,l3)
        
            model.to(device)
            weight_decay = 0.05
            if aug == 0:
                EPOCHS = 1000
               #EPOCHS = 1000
            else:
                EPOCHS = 2000
            print(model)
        else:
            f1 = 4
            d = 3
            f2 = f1*d
            k1 = 60
            chs = 71
            model = EEGNet(f1,f2,k1,d,chs).cuda(0)
            X_train = np.transpose(X_train[:,:,:,None],(0,3,1,2))
            X_val = np.transpose(X_val[:,:,:,None],(0,3,1,2))
            X_other = np.transpose(X_other[:,:,:,None],(0,3,1,2))
            LEARNING_RATE = 0.00003
            weight_decay = 0.5
            EPOCHS = 2000
            if aug == 1:
                EPOCHS = 1000
        BATCH_SIZE = 512
        
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE,weight_decay = weight_decay)
        criterion = torch.nn.BCEWithLogitsLoss(weight = torch.tensor(len(y_train)/np.sum(y_train)))
        train_data = trainData(torch.FloatTensor(X_train), 
                               torch.FloatTensor(y_train))        
        
        test_data = testData(torch.FloatTensor(X_val),torch.FloatTensor(y_val))
        test_data_other = testData(torch.FloatTensor(X_other),torch.FloatTensor(y_other))
        train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True,drop_last=True)
        test_loader = DataLoader(dataset=test_data, batch_size=BATCH_SIZE)   
        test_loader_other = DataLoader(dataset=test_data_other,batch_size=BATCH_SIZE)
        
    
        model.train()
    
        epoch_loss = np.zeros(EPOCHS)
        epoch_acc = np.zeros(EPOCHS)
        acc_test_val = np.zeros(EPOCHS)
        acc_test_other = np.zeros(EPOCHS)
        e_end = 1

        for e in range(e_end, EPOCHS):
            y_predtot = torch.empty(0).to(device)
            y_batchtot = torch.empty(0).to(device)
            for X_batch, y_batch in train_loader:
                model.train()
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                optimizer.zero_grad()
                y_pred = model(X_batch)
                loss = criterion(y_pred, y_batch.unsqueeze(1))
                loss.backward()
                optimizer.step()
                model.eval()
                y_predtot = torch.cat((y_pred,y_predtot))
                y_batchtot = torch.cat((y_batch,y_batchtot))
                epoch_loss[e-1] += loss.item()/len(train_loader)
            epoch_acc[e] = sklearn.metrics.balanced_accuracy_score(y_batchtot.cpu().numpy(),torch.round(torch.sigmoid(y_predtot.detach())).cpu().numpy())
        #
             
        
            y_predtot = torch.empty(0).to(device)
            y_batchtot = torch.empty(0).to(device)
                
            model.eval()
            with torch.no_grad():
                for X_batch_test,y_batch_test in test_loader_other:
                    X_batch_test = X_batch_test.to(device)
                    y_batch_test = y_batch_test.to(device)
                    y_pred = model(X_batch_test)
                    y_predtot = torch.cat((y_pred,y_predtot))
                    y_batchtot = torch.cat((y_batch_test,y_batchtot))
                acc_test_other[e] =  sklearn.metrics.balanced_accuracy_score(y_batchtot.cpu().numpy(),torch.round(torch.sigmoid(y_predtot.detach())).cpu().numpy())
            
            y_predtot = torch.empty(0).to(device)
            y_batchtot = torch.empty(0).to(device)
            model.eval()
            with torch.no_grad():
                for X_batch_test,y_batch_test in test_loader:
                    X_batch_test = X_batch_test.to(device)
                    y_batch_test = y_batch_test.to(device)
                    y_pred = model(X_batch_test)
                    y_predtot = torch.cat((y_pred,y_predtot))
                    y_batchtot = torch.cat((y_batch_test,y_batchtot))
                acc_test_val[e] =  sklearn.metrics.balanced_accuracy_score(y_batchtot.cpu().numpy(),torch.round(torch.sigmoid(y_predtot.detach())).cpu().numpy())
            
            e_end  = e
  
            print(f'Epoch {e+0:03}: | Loss: {epoch_loss[e-1]:.5f} | Acc_train: {epoch_acc[e-1]:.3f}| Acc_val: {acc_test_val[e-1]:.3f} | Acc_other: {acc_test_other[e-1]:.3f}' ) 