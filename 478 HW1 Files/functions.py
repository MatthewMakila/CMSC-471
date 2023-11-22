#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  6 23:48:52 2022

@author: abhishek.umrawal
"""

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import timeit

def reading_and_scaling(filename):

    df = pd.read_csv (filename)    
    X = df.iloc[:, 1:].copy()
    X = (X-X.mean())/X.std()
    X['x0'] = 1
    cols = X.columns.tolist()
    cols = cols[-1:] + cols[:-1]
    X = X[cols]
    
    X = np.matrix(X)
    y = np.transpose(np.matrix(df['mpg']))
    
    return X, y

def normal_equations(X,y):
    return np.matmul(np.linalg.inv(np.matmul(np.transpose(X),X)),np.matmul(np.transpose(X),y))

def gradient_descent(X,y,alpha):
    
    start = timeit.default_timer()
    
    m = X.shape[0]
    n = X.shape[1]
    
    e = 10**-5
    theta = np.zeros((n,1))
    h = np.zeros((m,1))

    convergence = False
    iter_count = 0
    costs = []
    while (convergence == False):
        iter_count+=1
        h = np.matmul(X,theta)
        
        grad = np.zeros((n,1))
    
        for j in range(n):
            grad[j] = (-1/m) * np.matmul(np.transpose(y-h), X[:,j])
    
        theta_new = theta - alpha*grad
    
        if np.all(abs(theta-theta_new) < e):
            convergence = True
    
        theta = theta_new
        costs.append((1/2*m) * np.matmul(np.transpose(y-h), (y-h)).item())
        
    end = timeit.default_timer()

    run_time = np.round(end - start,4)
    
    return theta, costs, run_time
  
def plotting(X,y):
    _, costs_10, run_time_10 = gradient_descent(X,y,0.10)
    _, costs_15, run_time_15 = gradient_descent(X,y,0.15)
    _, costs_20, run_time_20 = gradient_descent(X,y,0.20)
    _, costs_25, run_time_25 = gradient_descent(X,y,0.25)
    _, costs_30, run_time_30 = gradient_descent(X,y,0.30)

    plt.figure(1)
    plt.xlabel('iteration')
    plt.ylabel('cost')
    plt.plot(list(range(200,len(costs_10))), costs_10[200:], label = 'learning rate = 0.10')
    plt.plot(list(range(200,len(costs_15))), costs_15[200:], label = 'learning rate = 0.15')
    plt.plot(list(range(200,len(costs_20))), costs_20[200:], label = 'learning rate = 0.20')
    plt.plot(list(range(200,len(costs_25))), costs_25[200:], label = 'learning rate = 0.25')
    plt.plot(list(range(200,len(costs_30))), costs_30[200:], label = 'learning rate = 0.30')
    plt.legend(loc="upper right")
    plt.grid()

    plt.figure(2)
    alphas = [0.10, 0.15, 0.20, 0.25, 0.30]
    run_times = [run_time_10, run_time_15, run_time_20, run_time_25, run_time_30]
    plt.xlabel('learning rate')
    plt.ylabel('run time')
    plt.plot(alphas, run_times, 's')
    plt.xticks(alphas)
    plt.grid()