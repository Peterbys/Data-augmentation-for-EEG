# -*- coding: utf-8 -*-

import os
os.environ['ETS_TOOLKIT'] = 'qt4'
os.environ['QT_API'] = 'pyqt5'
import numpy as np
global subject_dir
subject_dir = 'D:/'
import mne
face_normal_event = [5,6,7,13,14,15]
face_scrambled_event = [17,18,19]


def get_raw(subject,ind):
    root = 'D:/ds000117-download/sub-' +subject + '/ses-meg/meg/'
    raw_fname = root  + 'sub-' + subject +'_ses-meg_task-facerecognition_run-0'+ str(ind) +'_meg.fif'
    raw = mne.io.read_raw_fif(raw_fname,preload=True)  # already has an average reference
    raw.info['bads'] = ["EEG061","EEG062","EEG063"]
    raw.set_eeg_reference('average', projection=True)  # set average reference.
    raw.apply_proj()
    raw.filter(0.01,40)

    return raw
def get_events(raw):
    face_normal_event = [5,6,7,13,14,15]
    face_scrambled_event = [17,18,19]
    events = mne.find_events(raw,stim_channel="STI101",shortest_event=1)
    for i in range(events.shape[0]):
        if events[i,2] in face_normal_event:
            events[i,2] = 10
        elif events[i,2] in face_scrambled_event:
            events[i,2] = 11
    return events
def get_epochs(raw,events,event_id):  
    event_id_ = dict(normal = event_id)
    tmin = -0.3
    tmax = 0.8

    picks = mne.pick_types(raw.info, meg=False, eeg=True, eog=True,
                       exclude='bads')
    baseline = (tmin,0)  # means from the first instant to t = 0
    
    epochs = mne.Epochs(raw, events, event_id_, tmin, tmax, proj=True, picks=picks,
                    baseline=baseline,preload=True)
    epochs.decimate(9)
    
    tmax =  (events[0,0]-10000)/raw.info['sfreq']
    return epochs, mne.compute_raw_covariance(raw, tmin=0, tmax=tmax, tstep=1, reject=None, flat=None, picks=picks, method='shrunk', method_params=None, cv=3, scalings=None, n_jobs=1, return_estimators=False, reject_by_annotation=True, rank=None, verbose=None)

def split_events_traintest(raw,subject,seed,p):
    np.random.seed(seed)
    events_ind = {}

    
    events = get_events(raw)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       
    k = int(len(np.argwhere(events[:] == 10)[:,0])*p)
    ind = np.random.choice(np.argwhere(events[:] == 10)[:,0],k,False)
    ind_test = list(set(np.argwhere(events[:] == 10)[:,0])-set(ind))
    events_ind['train_norm'] = np.sort(events[ind,:],axis=0)
    events_ind['test_norm'] = np.sort(events[ind_test,:],axis=0)
    k = int(len(np.argwhere(events[:] == 11)[:,0])*p)
    ind = np.random.choice(np.argwhere(events[:] == 11)[:,0],k,False)
    ind_test = list(set(np.argwhere(events[:] == 11)[:,0])-set(ind))
    events_ind['train_scrambled'] = np.sort(events[ind,:],axis=0)
    events_ind['test_scrambled'] = np.sort(events[ind_test,:],axis=0)   
    return events_ind
def gen_fwd(subject,conductivity,info):
    
    src = mne.read_source_spaces(subject_dir + '/sub-' + subject + '_free' + '/sub-' + subject + '_free-src.fif')
    trans = subject_dir + '/sub-' + subject + '_free' + '/sub-' + subject + '_free-trans.fif'
    model = mne.make_bem_model(subject = 'sub-' + subject + '_free', ico=4,
                           conductivity=conductivity,
                           subjects_dir=subject_dir)
    bem = mne.make_bem_solution(model)
    return mne.make_forward_solution(info, trans=trans, src=src, bem=bem,
                                eeg=True,meg=False, mindist=5.0, n_jobs=2)


def gen_data_dict(subjects,runs):
    data = {}
    for sub in subjects: 
        data[sub] = {}
        for run in range(1,runs+1):
            data[sub][run] = {}
            data[sub][run]['normal'] = {}
            data[sub][run]['scrambled'] = {}
        
    return data

def gen_artdata_dict(subjects_train,subjects_test,runs):
    data = {}
    for sub_train in subjects_train: 
        data[sub_train] = {}
        for sub_test in subjects_test:
            data[sub_train][sub_test] = {}
            for run in range(1,runs+1):
                data[sub_train][sub_test][run] = {}
                data[sub_train][sub_test][run]['normal'] = {}
                data[sub_train][sub_test][run]['scrambled'] = {}
        
    return data
    
def dict_concat(dict_,subjects,runs,type_=True):
    if type_:
        shape_ = dict_[subjects[0]][runs]['normal'].get_data().shape
    else:
        shape_ = dict_[subjects[0]][runs]['normal'].shape
    dict_o = {}
    data_normal = np.zeros((0,shape_[1],shape_[2]))
    data_scrambled = np.zeros((0,shape_[1],shape_[2]))
    if type_:
        for sub in subjects:
            for run in range(1,runs+1):
                data_normal = np.concatenate((data_normal,dict_[sub][run]['normal'].get_data()))
                data_scrambled = np.concatenate((data_scrambled,dict_[sub][run]['scrambled'].get_data()))
    else:
        for sub in subjects:
            for run in range(1,runs+1):
                data_normal = np.concatenate((data_normal,dict_[sub][run]['normal']))
                data_scrambled = np.concatenate((data_scrambled,dict_[sub][run]['scrambled']))
    dict_o['normal'] = data_normal
    dict_o['scrambled'] = data_scrambled
    return dict_o

def dict_art_concat(dict_,subjects_train,subjects_test,runs,type_=True):
    if type_:
        shape_ = dict_[subjects_train[0]][subjects_test[0]][runs]['normal'].get_data().shape
    else:
        not_found = True
        i= 0
        while not_found:
            try:
                shape_ = dict_[subjects_train[0]][subjects_test[i]][runs]['normal'].shape
                not_found = False
            except:
                i+=1
        
    dict_o = {}
    data_normal = np.zeros((0,shape_[1],shape_[2]))
    data_scrambled = np.zeros((0,shape_[1],shape_[2]))
    if type_:
        for sub1 in subjects_train:
            for sub2 in subjects_test:
                for run in range(1,runs+1):
                    if len((dict_[sub1][sub2][run]['normal'].get_data())) > 0:
                        data_normal = np.concatenate((data_normal,dict_[sub1][sub2][run]['normal'].get_data()))
                    if len((dict_[sub1][sub2][run]['scrambled'].get_data())) > 0:
                        data_scrambled = np.concatenate((data_scrambled,dict_[sub1][sub2][run]['scrambled'].get_data()))
    else:
        for sub1 in subjects_train:
            for sub2 in subjects_test:
                for run in range(1,runs+1):
                    if len(dict_[sub1][sub2][run]['normal']) > 0 :
                        data_normal = np.concatenate((data_normal,dict_[sub1][sub2][run]['normal']))
                    if len(dict_[sub1][sub2][run]['scrambled']) > 0 :
                        data_scrambled = np.concatenate((data_scrambled,dict_[sub1][sub2][run]['scrambled']))
    dict_o['normal'] = data_normal
    dict_o['scrambled'] = data_scrambled
    return dict_o
def dict_to_data(dict_):
 
   y = np.concatenate((np.zeros(dict_['normal'].shape[0]),np.ones(dict_['scrambled'].shape[0])))
   X = np.concatenate((dict_['normal'],dict_['scrambled'])) 
   return X,y

def gen_fwd_list(subject_dir,subjects,conductivity,info):
    fwd_list = {}
    for subject in subjects:
        src = mne.read_source_spaces(subject_dir + '/sub-' + subject + '_free' + '/sub-' + subject + '_free-src.fif')
        trans = subject_dir + '/sub-' + subject + '_free' + '/sub-' + subject + '_free-trans.fif'
        model = mne.make_bem_model(subject = 'sub-' + subject + '_free', ico=4,
                               conductivity=conductivity,
                               subjects_dir=subject_dir)
        bem = mne.make_bem_solution(model)
        fwd_list[subject]= mne.make_forward_solution(info, trans=trans, src=src, bem=bem,
                                    eeg=True,meg=False, mindist=5.0, n_jobs=2)
    return fwd_list
def split_data(data,subjects,p,seed,runs):
    data_train = gen_data_dict(subjects,runs)
    data_val  = gen_data_dict(subjects,runs)
    for subject in subjects:
        for run in range(1,runs+1):
            train_ind,test_ind = gen_split(data[subject][run]['normal'].events.shape[0],p,seed)
            data_train[subject][run]['normal'] = data[subject][run]['normal'][train_ind]
            data_val[subject][run]['normal'] = data[subject][run]['normal'][test_ind]
            data_train[subject][run]['cov'] = data[subject][run]['cov']
            train_ind,test_ind = gen_split(data[subject][run]['scrambled'].events.shape[0],p,seed)
            data_train[subject][run]['scrambled'] = data[subject][run]['scrambled'][train_ind]
            data_val[subject][run]['scrambled'] = data[subject][run]['scrambled'][test_ind]
    return data_train,data_val
def gen_split(number,p,seed):
    np.random.seed(seed)
    indexes = [i for i in range(number)]
    train_ind = np.random.choice(indexes,int(len(indexes)*p),replace=False)
    test_ind = list(set(indexes)-set(train_ind))
    return train_ind,test_ind
def gen_experiment_dict(num_experiments,aug_list):
    dict_ = {}
    for i in range(num_experiments):
        dict_[i] = {}
        for j in aug_list:
            dict_[i][j] = {}
    return dict_
def norm_scale(evoked,data,mean):
     if mean == 0:
        return np.linalg.norm(evoked.get_data())/np.linalg.norm(data)
     elif mean == 1:
        return np.linalg.norm(np.mean(evoked.get_data(),axis=0))/np.linalg.norm(np.mean(data,axis=0))
