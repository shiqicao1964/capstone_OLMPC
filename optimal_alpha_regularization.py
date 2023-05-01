#!/usr/bin/env python
# coding: utf-8

# In[41]:


import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from skopt import gp_minimize
from sklearn.model_selection import cross_val_score
from skopt.space import Real
import warnings
warnings.filterwarnings('ignore')
import time
from tqdm import trange
from scipy.optimize import minimize

predict = np.genfromtxt('predict_mismatch_16.out', delimiter=",")
measurement = np.genfromtxt('measurement_mismatch_16.out', delimiter=",")
controls = np.genfromtxt('controls_mismatch_16.out', delimiter=",")
x1 = (measurement[:,0:-1])
x2 = (controls[:,0:-1])
input_state = np.concatenate(( x1 , x2 ), axis=0)[:,::2]
error_y = (measurement[:,1:] - predict[:,1:])[:,::2]
print('input_state',input_state.shape)
print('error_y',error_y.shape)


# In[42]:


class GPR:
    def __init__(self, optimize=True):
        self.is_fit = False
        self.train_X, self.train_y = None, None
        self.params = {"l": 0.3, "sigma_f": 0.2}
        self.optimize = optimize
    def kernel(self, x1, x2):
        dist_matrix = np.sum(x1**2, 1).reshape(-1, 1) + np.sum(x2**2, 1) - 2 * np.dot(x1, x2.T)
        return self.params["sigma_f"] ** 2 * np.exp(-0.5 / self.params["l"] ** 2 * dist_matrix)
    
    def fit(self, X, y,bounds_):
        # store train data
        self.train_X = np.asarray(X)
        self.train_y = np.asarray(y)
        self.params['sigma_f'] = np.std(y)
         # hyper parameters optimization
        def negative_log_likelihood_loss(params):
            self.params["l"]= params
            Kyy = self.kernel(self.train_X, self.train_X) #+ 1e-8 * np.eye(len(self.train_X))
            loss = 0.5 * self.train_y.T.dot(np.linalg.inv(Kyy)).dot(self.train_y) + 0.5 * np.linalg.slogdet(Kyy)[1] + 0.5 * len(self.train_X) * np.log(2 * np.pi)
            return np.sum(loss.ravel())

        if self.optimize:
            res = minimize(negative_log_likelihood_loss, [self.params["l"]],
                   bounds=bounds_,
                   method='L-BFGS-B')
            self.params["l"] = res.x[0]
        self.is_fit = True


# In[49]:


gpr1 = GPR()
gpr1.fit(input_state.T,error_y[[2],:].T , [(0.1, 10)] )
l_1 = gpr1.params['l']
sig_f_1 = gpr1.params['sigma_f']
print(f'l_1:',l_1,'sig_f_1:',sig_f_1)

search_list_alpha = np.array([1e-1,5e-2,1e-2,5e-3,1e-3,5e-4,1e-4,5e-5,1e-5,5e-6,1e-6,5e-7,1e-7,5e-8,1e-8])
recode_score=np.array([])
alpha_lst = np.array([])
for i in trange(15):
    alpha = search_list_alpha[i]
    print('alpha',alpha)
    kernel = ConstantKernel(0.1) * RBF(1)
    gp = GaussianProcessRegressor(kernel=kernel, alpha=alpha)
    cv_scores = cross_val_score(gp, X, y, cv=5, scoring='neg_mean_squared_error')
    print('score:',np.mean(cv_scores))
    recode_score = np.append(recode_score,np.mean(cv_scores))
    alpha_lst = np.append(alpha_lst,alpha)
kernel = sig_f_1**2 * RBF(l_1)
gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-6)
gp.fit(X, y)
print(recode_score)
print(alpha_lst)
print('optimal_alpha',alpha_lst[recode_score == np.max(recode_score)])


# In[ ]:





# In[ ]:




