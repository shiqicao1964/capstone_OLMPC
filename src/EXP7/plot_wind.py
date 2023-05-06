import numpy as np
from math import sqrt
import matplotlib.pyplot as plt
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time

if __name__ == "__main__":
   

    wind = np.array([0,4,8,12])

    control_group_avg = np.array([0.151,0.152,0.167,0.171])
    control_group_max = np.array([0.39,0.42,0.57,0.67])
    control_group_ratio = np.array([0.02,0.026,0.0405,0.065])*100

    MLL_avg = np.array([0.135,0.151,0.172,0.173])
    MLL_max = np.array([0.28,0.41,0.46,0.53])
    MLL_ratio = np.array([0.017,0.03,0.052,0.064])*100

    CROSS_avg = np.array([0.144,0.151,0.158,0.154])
    CROSS_max = np.array([0.33,0.41,0.36,0.34])
    CROSS_ratio = np.array([0.014,0.023,0.019,0.029])*100

    bayes_avg = np.array([0.141,0.137,0.138,0.144])
    bayes_max = np.array([0.34,0.36,0.32,0.36])
    bayes_ratio = np.array([0.012,0.0114,0.0147,0.016])*100


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
    ax.set_xlabel('wind [m/s]')
    ax.set_ylabel('RMSE [m]')
    ax.grid(True)
    ax.legend()
    ax.set_title('wind speed vs RMSE')
    

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
    ax.set_xlabel('wind [m/s]')
    ax.set_ylabel('Max Error [m]')
    ax.grid(True)
    ax.legend()
    ax.set_title('wind speed vs Max Error')
    

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
    ax.set_xlabel('wind [m/s]')
    ax.set_ylabel('Error>0.25 ratio (%)')
    ax.grid(True)
    ax.legend()
    ax.set_title('wind speed vs Error>0.25 ratio')
    plt.show()

    
