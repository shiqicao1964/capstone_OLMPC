#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import numpy
import matplotlib.pyplot as plt
import casadi as cs
from casadi.casadi import *
import math 
from scipy.optimize import minimize
from math import sqrt


# In[2]:

predict = np.genfromtxt("model_predict_record.out", delimiter=",")
measurement = np.genfromtxt("measurements_record.out", delimiter=",")
control = np.genfromtxt("controlinput_record.out", delimiter=",")
ind = 200
measurement = measurement[:,ind:-ind]
predict = predict[:,ind:-ind]
control = control[:,ind:-ind]

x1 = (measurement[:,0:-1])
x2 = (control[:,0:-1])
print('x1',x1.shape)
print('x2',x2.shape)
input_state = np.concatenate(( x1 , x2 ), axis=0)
error_y = (measurement[:,1:] - predict[:,1:])[6:9,:]
input_state = input_state[:,::4]
error_y = error_y[:,::4]
print('train set size:',error_y.shape)
print(input_state.shape)





def kernel(x1, x2,sigf,l):
    x1 = x1.T
    x2 = x2.T
    dist_matrix = np.sum(x1**2, 1).reshape(-1, 1) + np.sum(x2**2, 1) - 2 * np.dot(x1, x2.T)
    return sigf ** 2 * np.exp(-0.5 / l ** 2 * dist_matrix)

def y_hat(X,y,query_slot_index,sigf,l) :
    query_point = X[:,query_slot_index]
    traian_setX = np.delete(X, query_slot_index, axis=1)
    train_sety = np.delete(y, query_slot_index, axis=1)
    
    Ks = kernel(query_point,traian_setX,sigf,l)
    Kxxinv = np.linalg.inv( kernel(traian_setX,traian_setX,sigf,l))
    
    y_hat = np.dot ( np.dot(  Ks ,Kxxinv ), train_sety.T ).T
    return y_hat


def error_square(X,y,query_slot_index,sigf,l):
    Validation_y = y[:,query_slot_index]
    error_square = sum (sum ((y_hat(X,y,query_slot_index,sigf,l) - Validation_y)**2))
    return error_square

def RMS (X,y,sigf,l,slot_number):
    totalRMS = 0
    slot_size = int(X.shape[1]/slot_number)
    for i in range(slot_number):
        query_slot_index = np.array(range(i*slot_size,(i+1)*slot_size  ))
        i_RMS = error_square(X,y,query_slot_index,sigf,l)
        totalRMS += i_RMS
    return totalRMS

def negative_log_likelihood_loss(X,y,sigf,l):
    K = kernel(X,X,sigf,l)
    loss = 0.5 * y.dot(np.linalg.inv(K)).dot(y.T) + 0.5 * np.linalg.slogdet(K)[1] + 0.5 * X.shape[1] * np.log(2 * np.pi)
    return loss.ravel()

from tqdm import trange
slot_number = 20
errorRMS_1 = np.zeros(60)
length_scales = np.zeros(60)
for k in trange(60):
    length_scale = math.exp(k/10 -3)
    length_scales[k] = length_scale
    errorRMS_1[k] = RMS(input_state,error_y,np.std(error_y),length_scale,slot_number)
minindex_1 = numpy.where(errorRMS_1 == numpy.amin(errorRMS_1))[0]
errorRMS_2 = np.zeros(14)
length_scales_2 = np.zeros(14)

for k in trange(14):
    length_scale = math.exp(k/50 -3.14 + minindex_1/10)
    length_scales_2[k] = length_scale
    errorRMS_2[k] = RMS(input_state,error_y,np.std(error_y),length_scale,slot_number)
minindex = numpy.where(errorRMS_2 == numpy.amin(errorRMS_2))
L = length_scales_2[minindex[0]]
print('the optimal L : ',L)
np.savetxt('Final_train_set/length_scale.out',np.array([L]),delimiter=',')
fig,axs = plt.subplots(2)
axs[0].plot(length_scales,errorRMS_1)
axs[0].set_title('length_scale VS RMS value')
axs[1].plot(length_scales_2,errorRMS_2)
plt.show()

np.savetxt('Final_train_set/trainset_statesX.out',input_state,delimiter=',')
np.savetxt('Final_train_set/trainset_error_y.out',error_y,delimiter=',')

fig = plt.figure(figsize = (18,18))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(measurement[0,:],measurement[1,:],measurement[2,:])
plt.show()

def fit(X,y,l,sig_f):
    
    L = np.diag(np.ones(  X.shape[0]  ) * l)
    L = L**2

    K = kernel(X,X,sig_f,l)
    error = cs.MX.sym('error',y.shape[0],1)
    x = cs.MX.sym('x',X.shape[0],1)

    x1 = x.T
    x2 = X.T
    dist_matrix = cs.sum2(x1**2) + cs.sum2(x2**2) - ((cs.mtimes(x1, x2.T))*2).T
    Kstar = (sig_f ** 2 * np.exp(-0.5 / l ** 2 * dist_matrix)).T
    error = cs.mtimes ( cs.mtimes(Kstar,np.linalg.inv(K) ), y.T ).T
    
    return error,x

sig_f = np.std(error_y)
length_scale = L
optimal,x = fit(input_state,error_y,length_scale,sig_f)
optimal_func = cs.Function('f',[x],[optimal])


