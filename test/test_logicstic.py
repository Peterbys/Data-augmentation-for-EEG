#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 20 13:29:54 2020

@author: peter
"""
import get_data
import mne
import numpy as np
from sklearn.model_selection import train_test_split
faces_normal = get_data.gen_vectorized_data(subjects_normal_dict)
faces_scrambled = get_data.gen_vectorized_data(subjects_scrambled_dict)
X = np.concatenate((faces_normal,faces_scrambled))
y = np.concatenate((np.ones(faces_normal.shape[0]),np.zeros(faces_scrambled.shape[0])))
x_train, x_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)
#%%
from pathlib import Path
import requests
import pickle
import gzip
import torch
x_train, y_train, x_valid, y_valid = map(torch.tensor, (x_train,  y_train, x_valid, y_valid))
#%%
import math
weights = torch.randn(x_train.shape[1], 2) / math.sqrt(x_train.shape[1])
weights.requires_grad_()
bias = torch.zeros(2, requires_grad=True)
def log_softmax(x):
    return x - x.exp().sum(-1).log().unsqueeze(-1)

def model(xb):
    return log_softmax(xb @ weights.double() + bias)
def nll(input, target):
    return -input[range(target.shape[0]), target].mean()
#%%
bs = 64  # batch size
loss_func = nll
lr = 0.5  # learning rate
epochs = 2  # how many epochs to train for

for epoch in range(epochs):
    for i in range((x_train.shape[0] - 1) // bs + 1):
        start_i = i * bs
        end_i = start_i + bs
        xb = x_train[start_i:end_i]
        yb = y_train[start_i:end_i]
        pred = model(xb.double())
        loss = loss_func(pred, yb.bool())

        loss.backward()
        with torch.no_grad():
            weights -= weights.grad * lr
            bias -= bias.grad * lr
            weights.grad.zero_()
            bias.grad.zero_()