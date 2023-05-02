#!/usr/bin/env python
# coding: utf-8

# In[9]:


import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from skopt import gp_minimize
from skopt.space import Real
from skopt.utils import use_named_args
from sklearn.metrics import mean_squared_error
from math import sqrt
import numpy as np
import time
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
import warnings
warnings.filterwarnings('ignore')

t0 = time.time()
# 加载数据集
predict = np.genfromtxt('predict_mismatch_16.out', delimiter=",")
measurement = np.genfromtxt('measurement_mismatch_16.out', delimiter=",")
controls = np.genfromtxt('controls_mismatch_16.out', delimiter=",")
x1 = (measurement[:,0:-1])
x2 = (controls[:,0:-1])
input_state = np.concatenate(( x1 , x2 ), axis=0)
noise = np.random.normal(0, 0.001, input_state.shape)
input_state = input_state + noise
error_y = (measurement[:,1:] - predict[:,1:])

input_state_0 = input_state[:,::10].T
error_y_0 = error_y[:,::10].T

input_state_1 = input_state[:,1::10].T
error_y_1 = error_y[:,1::10].T

print('input_state_0',input_state_0.shape)

# 定义高斯过程回归模型
kernel = C(1.0, (1e-3, 1e2)) * RBF(1.0, (1e-2, 10))

# 定义超参数搜索空间
search_space = [Real(1e-7, 1e-1, name='alpha')]

# 定义高斯过程回归模型及其优化目标
@use_named_args(search_space)
def objective(alpha):
    model = GaussianProcessRegressor(kernel=kernel, alpha=alpha)
    model.fit(input_state_0, error_y_0)
    y_pred = model.predict(input_state_1)
    rmse = sqrt(mean_squared_error(error_y_1, y_pred))
    return rmse

# 定义初始超参数
x0 = [1e-3]

# 运行贝叶斯优化
res = gp_minimize(objective, search_space, x0=x0, n_calls=25, random_state=0)

# 输出最佳超参数
best_alpha = res.x[0]
print("Best alpha:", best_alpha,'time used',time.time()-t0)


# In[ ]:




