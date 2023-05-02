import numpy as np
from math import sqrt
import matplotlib.pyplot as plt
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
def eight_trag(speed = 3,x_w = 3,y_w = 4,z_w = 0,H = 5,dT = 0.05,sim_t = 100):
    t_abl = [0]
    t = np.linspace(0, 2*np.pi, num=10000)
    x = x_w * np.cos(t) / (1 + np.sin(t)**2)
    y = y_w * np.sin(t) * np.cos(t) / (1 + np.sin(t)**2)
    z = H + z_w*np.cos(t)
    x0 = x_w * np.cos(0) / (1 + np.sin(0)**2)
    y0 = y_w * np.sin(0) * np.cos(0) / (1 + np.sin(0)**2)
    z0 = H + z_w*np.cos(0)

    for n in range(len(t)):
        # calculate if dist == dT * speed
        dist = math.sqrt((x[n]-x0)**2 + (y[n]-y0)**2 + (z[n]-z0)**2)
        if dist > dT * speed:
            t_abl.append(t[n])
            x0 = x[n]
            y0 = y[n]
            z0 = z[n]
    t_new = np.array(t_abl)
    T_onecircle = len(t_new)*dT
    circles = int(np.ceil(sim_t/T_onecircle))
  
    t_abl = [0]
    vx = []
    vy = []
    vz = []
    t = np.linspace(0, circles*2*np.pi, num=10000*circles)
    x = x_w * np.cos(t) / (1 + np.sin(t)**2)
    y = y_w * np.sin(t) * np.cos(t) / (1 + np.sin(t)**2)
    z = H + z_w*np.cos(t)
    x0 = x_w * np.cos(0) / (1 + np.sin(0)**2)
    y0 = y_w * np.sin(0) * np.cos(0) / (1 + np.sin(0)**2)
    z0 = H + z_w*np.cos(0)
    for n in range(len(t)):
        # calculate if dist == dT * speed
        dx = x[n]-x0
        dy = y[n]-y0
        dz = z[n]-z0
        dist = math.sqrt((dx)**2 + (dy)**2 + (dz)**2)
        if dist > dT * speed:
            t_abl.append(t[n])
            vx.append( dx * speed / dist )
            vy.append( dy * speed / dist )
            vz.append( dz * speed / dist )
            x0 = x[n]
            y0 = y[n]
            z0 = z[n]
    t_new = np.array(t_abl)
    x = x_w * np.cos(t_new) / (1 + np.sin(t_new)**2)
    y = y_w * np.sin(t_new) * np.cos(t_new) / (1 + np.sin(t_new)**2)
    z =  H + z_w*np.cos(t_new)
    vx = np.array(vx)
    vy = np.array(vy)
    vz = np.array(vz)
    return x[:-1], y[:-1], z[:-1],t_new,vx,vy,vz

if __name__ == "__main__":
   
    N = 20
    # set traj 
    discretization_dt = 0.05
    z = 5
    v_average = 3
    sim_t = 120
    x_w = 4
    y_w = 4
    z_w = 0
    pos_traj_x,pos_traj_y,pos_traj_z,t,vel_traj_x,vel_traj_y,vel_traj_z = eight_trag(speed = v_average,x_w = x_w,y_w = y_w,z_w = z_w,H = z,dT = discretization_dt,sim_t = sim_t)
    ref = np.zeros((pos_traj_x.shape[0],12))

    ref[::,0] = pos_traj_x
    ref[::,1] = pos_traj_y
    ref[::,2] = pos_traj_z
    #ref[::,6] = vel_traj_x
    #ref[::,7] = vel_traj_y
    #ref[::,8] = vel_traj_z

    #PREpredict.out
    #PREmeasurement.out
    #PREcontrols.out
    #PREgp_predict.out

    #measurement.out
    #predict.out
    #gp_predict.out

    mes = np.genfromtxt('measurement.out',delimiter=',')[:,:]
    pre = np.genfromtxt('predict.out',delimiter=',')[:,:]
    gppre = np.genfromtxt('gp_predict.out',delimiter=',')[:,:]

    print('gp_predict',gppre.shape)
    print('predict',pre.shape)
    print('measurement',mes.shape)

    error_DT2real = (mes-pre)**2
    error_GP2real = (mes-gppre)**2

    DT_err_sq = np.mean(error_DT2real,1) * 1000
    GP_err_sq = np.mean(error_GP2real,1) * 1000
    print('x     err:(DT ,GP)',DT_err_sq[0],GP_err_sq[0])
    print('y     err:(DT ,GP)',DT_err_sq[1],GP_err_sq[1])
    print('z     err:(DT ,GP)',DT_err_sq[2],GP_err_sq[2])
    #print('roll  err:(DT ,GP)',DT_err_sq[3],GP_err_sq[3])
    #print('pitch err:(DT ,GP)',DT_err_sq[4],GP_err_sq[4])
    #print('yaw   err:(DT ,GP)',DT_err_sq[5],GP_err_sq[5])
    print('Vx    err:(DT ,GP)',DT_err_sq[6],GP_err_sq[6])
    print('Vy    err:(DT ,GP)',DT_err_sq[7],GP_err_sq[7])
    print('Vz    err:(DT ,GP)',DT_err_sq[8],GP_err_sq[8])
    #print('p     err:(DT ,GP)',DT_err_sq[9],GP_err_sq[9])
    #print('q     err:(DT ,GP)',DT_err_sq[10],GP_err_sq[10])
    #print('r     err:(DT ,GP)',DT_err_sq[11],GP_err_sq[11])


    #time.sleep(1000)

    fig2D = plt.figure(figsize = (12,12))
    ax5 = fig2D.add_subplot(4,1,1)
    ax6 = fig2D.add_subplot(4,1,2)
    ax7 = fig2D.add_subplot(4,1,3)
    ax8 = fig2D.add_subplot(4,1,4)
    t = np.linspace(0, mes.shape[1] * 0.05, num=mes.shape[1])
    ax5.plot(t,mes[2],color='red')
    ax5.plot(t,pre[2],color='blue')
    ax5.plot(t,gppre[2],color='green')
    ax5.set_title('z')

    ax6.plot(t,mes[6],color='red')
    ax6.plot(t,pre[6],color='blue')
    ax6.plot(t,gppre[6],color='green')
    ax6.set_title('Vx')

    ax7.plot(t,mes[7],color='red')
    ax7.plot(t,pre[7],color='blue')
    ax7.plot(t,gppre[7],color='green')
    ax7.set_title('Vy')

    ax8.plot(t,mes[8],color='red')
    ax8.plot(t,pre[8],color='blue')
    ax8.plot(t,gppre[8],color='green')
    ax8.set_title('Vz')
    
    fig2D = plt.figure(figsize = (12,12))
    ax9 = fig2D.add_subplot(2,1,1)
    ax10 = fig2D.add_subplot(2,1,2)
    ax9.plot(t,mes[0],color='red')
    ax9.plot(t,pre[0],color='blue')
    ax9.plot(t,gppre[0],color='green')
    ax9.set_title('x')

    ax10.plot(t,mes[1],color='red')
    ax10.plot(t,pre[1],color='blue')
    ax10.plot(t,gppre[1],color='green')
    ax10.set_title('y')
    plt.show()
    import random
