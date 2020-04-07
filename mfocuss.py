# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 14:16:15 2020

@author: P
"""
from scipy import signal
import numpy as np


def Mfocuss(A,B,p,max_iter,lambda_,r):
    X = np.linalg.pinv(A) @ B
    k = 0
    while(k<max_iter):
        W = calc_w(X,p)
       # print(W.shape)
        A = A @ W
        #Q = np.linalg.pinv(A) @ B
        #print(Q.shape)
        Xold = X
        X = W @ (A.T @ np.linalg.inv((A @ A.T + np.eye(A.shape[0])*lambda_+1e-16)) @ B)
        delta = np.linalg.norm(X-Xold)/np.linalg.norm(Xold)
        print(delta)
        k += 1
    q = calc_w(X,0)
    ind = (q @ np.ones((q.shape[1]))).argsort()[-r:][::-1]
    return X,A,ind,q

def calc_w(X,p):
    W = np.zeros((X.shape[0],X.shape[0]))
    for i in range(X.shape[0]):
        W[i,i] = np.sqrt(np.mean(X[i,:]**2))**(1-p/2)
    return W