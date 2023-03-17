# -*- coding: utf-8 -*-
"""
Created on Sat Aug 27 11:27:08 2022

@author: njau
"""

import matplotlib.pyplot as plt
import numpy as np

x = np.array([2.4,5.0,1.5,3.8,8.7,3.6,1.2,8.1,2.5,5,1.6,1.6,2.4,3.9,5.4])
y = np.array([2.1,4.7,1.7,3.6,8.7,3.2,1.0,8.0,2.4,6,1.1,1.3,2.4,3.9,4.8])

n = len(x)  # number of training samples

# f(w,w_0)=wx+w_0

# 初始化参数
w_old = np.random.rand(1)
w0_old = np.random.rand(1)

# learning rate 
eta = 0.001 # can be modified 

T = 50  #  maximum iteration

# original loss function value
Loss_values = [0.5*np.sum((w_old*x+w0_old-y)**2)]

for i in range(T):
    # compute the gradient dL/dw, dL/dw0
    dL_w  = 0
    dL_w0 = 0
    for j in range(n):
        dL_w = dL_w + (w_old*x[j]+w0_old-y[j])*x[j]
        dL_w0 = dL_w0 + (w_old*x[j]+w0_old-y[j])
    # end of gradient computation
    
    w_new = w_old - eta* dL_w 
    w0_new = w0_old - eta * dL_w0 
    # compute the loss function at current step
    Loss_values.append(0.5*np.sum((w_new*x+w0_new-y)**2))
    
    w_old = w_new
    w0_old = w0_new
    
plt.figure(1,facecolor ='g')
plt.plot(range(0,T+1),Loss_values,'r-')
plt.xlabel('number of iteration')
plt.ylabel('Loss function values')    

#  plot the original data and fitted curve
# plot x-y data 
plt.figure(2,facecolor='y')
plt.show()
plt.plot(x,y,'r*')
plt.xlabel('x')
plt.ylabel('y')
plt.title("linear regression")

#  fitted curve
x = np.array(range(1,10))
y = w_new*x + w0_new
plt.plot(x,y,'k-')
plt.legend('original data','regression line')
    
    
    
    













