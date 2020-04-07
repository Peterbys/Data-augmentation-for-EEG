# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 14:40:18 2020

@author: P
"""
from gauss_kernels import gen_kernel
from scipy.cluster.vq import whiten
x_train = whiten(train)
x_test = whiten(train)
H = gen_kernel(0.1,train.shape[1])
det = np.linalg.det(H)**(-0.5)
Hinv = np.linalg.inv(H)
probs = np.zeros(x_test.shape[0])
for i in range(x_test.shape[0]):
    x = 0
    for j in range(x_train.shape[0]):
        tt = -0.5 * (x_train[j,:][:,None]-x_test[i,:][:,None]).T @ Hinv @ (x_train[j,:][:,None]-x_test[i,:][:,None])
        print("tt: " + str(tt))
        x += (2*np.pi)**(-200)*det*np.exp(tt)
        print(x)
    probs[i] = x * (1/x_train.shape[0])