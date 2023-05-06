import numpy as np
from math import sqrt
import matplotlib.pyplot as plt
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time

if __name__ == "__main__":
   

    noise = np.array([0,5e-3,1e-2,1.5e-2])

    control_group_avg = np.array([0.151,0.157,0.2,0.209])
    control_group_max = np.array([0.39,0.35,0.56,0.50])
    control_group_ratio = np.array([0.02,0.14,0.32,0.24])*100

    MLL_avg = np.array([0.135,0.157,0.169,0.24])
    MLL_max = np.array([0.28,0.39,0.335,0.6])
    MLL_ratio = np.array([0.017,0.03,0.14,0.342])*100    

    CROSS_avg = np.array([0.144,0.158,0.166,0.156])
    CROSS_max = np.array([0.33,0.39,0.46,0.40])
    CROSS_ratio = np.array([0.014,0.062,0.089,0.047])*100

    bayes_avg = np.array([0.141,0.149,0.162,0.147])
    bayes_max = np.array([0.34,0.314,0.35,0.39])
    bayes_ratio = np.array([0.012,0.02,0.0414,0.0214])*100

    wind = noise
    fig2D = plt.figure(figsize = (12,5))
    ax = fig2D.add_subplot()
    ax.plot(wind,control_group_avg,color='red',label='No regularization')
    ax.scatter(wind,control_group_avg,color='red')
    ax.plot(wind,MLL_avg,color='green',label='Method A (Maximum marginal likelihood)')
    ax.scatter(wind,MLL_avg,color='green')
    ax.plot(wind,CROSS_avg,color='yellow',label='Method B (Grid search)')
    ax.scatter(wind,CROSS_avg,color='yellow')
    ax.plot(wind,bayes_avg,color='blue',label='Method C (Bayesian Optimization)')
    ax.scatter(wind,bayes_avg,color='blue')
    ax.set_xlabel('added noise level (std)')
    ax.set_ylabel('RMSE [m]')
    ax.grid(True)
    ax.legend()
    ax.set_title('noise level vs RMSE')
    

    fig2D2 = plt.figure(figsize = (12,5))
    ax = fig2D2.add_subplot()
    ax.plot(wind,control_group_max,color='red',label='No regularization')
    ax.scatter(wind,control_group_max,color='red')
    ax.plot(wind,MLL_max,color='green',label='Method A (Maximum marginal likelihood)')
    ax.scatter(wind,MLL_max,color='green')
    ax.plot(wind,CROSS_max,color='yellow',label='Method B (Grid search)')
    ax.scatter(wind,CROSS_max,color='yellow')
    ax.plot(wind,bayes_max,color='blue',label='Method C (Bayesian Optimization)')
    ax.scatter(wind,bayes_max,color='blue')
    ax.set_xlabel('added noise level (std)')
    ax.set_ylabel('Max Error [m]')
    ax.grid(True)
    ax.legend()
    ax.set_title('noise level vs Max Error')
    

    fig2D3 = plt.figure(figsize = (12,5))
    ax = fig2D3.add_subplot()
    ax.plot(wind,control_group_ratio,color='red',label='No regularization')
    ax.scatter(wind,control_group_ratio,color='red')
    ax.plot(wind,MLL_ratio,color='green',label='Method A (Maximum marginal likelihood)')
    ax.scatter(wind,MLL_ratio,color='green')
    ax.plot(wind,CROSS_ratio,color='yellow',label='Method B (Grid search)')
    ax.scatter(wind,CROSS_ratio,color='yellow')
    ax.plot(wind,bayes_ratio,color='blue',label='Method C (Bayesian Optimization)')
    ax.scatter(wind,bayes_ratio,color='blue')
    ax.set_xlabel('added noise level (std)')
    ax.set_ylabel('Error>0.25 ratio (%)')
    ax.grid(True)
    ax.legend()
    ax.set_title('noise level vs Error>0.25 ratio')
    plt.show()

    
