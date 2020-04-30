#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 14:41:28 2020

@author: peter
"""
from __future__ import division
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import os
from sklearn import preprocessing
import warnings # to silence convergence warnings





def print_logistic_regression(X,y,K,filename,nr_regs):
    with warnings.catch_warnings(): # done to disable all the convergence warnings from elastic net
        warnings.simplefilter("ignore")
        # choose regularization strength
        Cvals =np.linspace(1e-6, 1, nr_regs)
        n = len(X)
        # perform a 5-fold cross-validation - there are so few samples that K = 10 gives worse results
        CV = KFold(K,shuffle=True)
        errorList = []
    
        X_standardizer = preprocessing.StandardScaler()
    
        error = np.zeros((K, nr_regs))
        # perform a logistic regression with L1 (Lasso) penalty and iterate C over the lambda values
        # we train on the training sets and estimate on the testing set
        for i, (train_index, test_index) in enumerate(CV.split(X)):
    
            X_train = X[train_index, :]
            y_train = y[train_index]       
            y_test = y[test_index]
            X_test = X[test_index]
            
            X_train = X_standardizer.fit_transform(X_train)
            X_test = X_standardizer.transform(X_test)
    
            for k, Cval in enumerate(Cvals):
                model = LogisticRegression(penalty = 'l1', C = Cval, tol = 1e-6, solver='saga', fit_intercept=False,class_weight = 'balanced')
                model = model.fit(X_train, y_train)
                y_est = model.predict(X_test)
                # the error is the difference between the estimate and the test
                error[i,k] = np.sum(np.abs(y_est-y_test))/len(y_test)

    # we take the mean error from every lambda value
    meanError = list(np.mean(error, axis=0))
    # we take the standard deviation from every lambda value
    std = np.std(error, axis=0)
    # this is the index of the smallest error
    minError = meanError.index(min(meanError))

    # We want to find the simplest model that is only one standard error away from the smallest error
    # We start by finding all indices that are less than one standard error away from the minimum error
    J = np.where(meanError[minError] + std[minError] > meanError)[0]
    # then we take the simplest model (furthest to the right)
    if (len(J) > 0):
        j = int(J[-1::])
    else:
        j = minError

    Lambda_CV_1StdRule = Cvals[j]
    print("CV lambda 1 std rule %0.2f" % Lambda_CV_1StdRule)
    
    
    ### plot ###
 
    # we plot the mean errors with their std values
    plt.errorbar(Cvals, meanError, std, marker='.', color='orange', markersize=10)
    plt.plot(Cvals, meanError)
    # and the locations of the smallest error and the simplest model according to the one standard error rule
    plt.plot(Cvals[minError], meanError[minError], marker='o', markersize=8, color="red")
    plt.plot(Cvals[j], meanError[j], marker='o', markersize=8, color="blue")
    xposition = [Cvals[minError], Cvals[j]]
    for xc in xposition:
        plt.axvline(x=xc, color='k', linestyle='--')
    plt.xlabel("Lambda")
    plt.ylabel("Deviance")
    plt.title("Cross-validated deviance of Lasso fit")
    plt.show()
    plt.savefig(filename)
    
    return Cvals, meanError