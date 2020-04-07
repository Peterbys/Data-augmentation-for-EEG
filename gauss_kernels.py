# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 10:14:51 2020

@author: P
"""
import numpy as np
def gauss_kernel(H,x_train,x_test):
    det = np.linalg.det(H)**(-0.5)
    Hinv = np.linalg.inv(H)
    probs = np.zeros(x_test.shape[0])
    for i in range(x_test.shape[0]):
        x = 0
        for j in range(x_train.shape[0]):
           tt = -0.5 * (x_train[j,:][:,None]-x_test[i,:][:,None]).T @ Hinv @ (x_train[j,:][:,None]-x_test[i,:][:,None])
           #print(tt)
           x += (2*np.pi)**(-x_train.shape[1]/2)*det*np.exp(tt[0])
          # print(x)
          # print(x)
       # print(x)
        probs[i] = x * (1/x_train.shape[0])
    return probs
def gen_kernel(scalar,dim):
    return np.eye(dim) * scalar

def evaluate_density(train_pca,test_pca):
    deltas = np.logspace(-40,-20,100)
   # H =gen_kernel(deltas[k]train_pca.shape[0])
    lik = np.zeros(len(deltas))
    for k in range(len(deltas)):
        H = gen_kernel(deltas[k],train_pca.shape[1])
        q = gauss_kernel(H,train_pca,test_pca)
        lik[k] = np.sum(q)
    return lik,deltas
#print(np.sum(q))

