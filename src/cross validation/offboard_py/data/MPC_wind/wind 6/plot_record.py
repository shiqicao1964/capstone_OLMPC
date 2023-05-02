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

    start_point = 500
    stop_point = 4000
    error_record_dyn = np.genfromtxt('pos_error_record_dyn.out',delimiter=',')[:,start_point:]

    print(error_record_dyn.shape)

    print('mean square error of dynamic model : (x , y , z)', np.mean(error_record_dyn,1))
    print('average distance error of dynamic model :', np.mean(np.sqrt(np.sum(error_record_dyn,0))))


    ref_mes_dyn = np.genfromtxt('ref_mes_dyn.out',delimiter=',')[:,start_point:]
    #ref_mes_dyn = np.concatenate((ref_mes_dyn, np.genfromtxt('ref_mes_1.out',delimiter=',') ), axis = 1)

    iindex = -1
    print('pos_record dyn',ref_mes_dyn.shape)
    print('error_record_dyn dyn',error_record_dyn.shape)

    x_w = 4
    H = 3

    fig = plt.figure(figsize = (18,6))
    fig.subplots_adjust(wspace=0.1, hspace=0.1)
    ax1 = fig.add_subplot(1, 3, 1, projection='3d')



    ax1.plot(ref_mes_dyn[3],ref_mes_dyn[4],ref_mes_dyn[5],color='blue')
    ax1.plot(ref_mes_dyn[0],ref_mes_dyn[1],ref_mes_dyn[2],color='red')
    ax1.set_xlim(-x_w,x_w)
    ax1.set_ylim(-x_w,x_w)
    ax1.set_zlim(H-1,H+1)


    plt.show()
    time.sleep(1)

    fig2 = plt.figure(figsize = (12,12))
    fig2.subplots_adjust(wspace=0.05, hspace=0.05)
    ax4 = fig2.add_subplot(111, projection='3d')
    plt.ion()

    ref_mes = ref_mes_dyn 
    for i in range(500):
        t_n = i
        ax4.plot(ref_mes[3],ref_mes[4],ref_mes[5],color='blue')
        ax4.plot(ref_mes[0],ref_mes[1],ref_mes[2],color='red')
        ax4.scatter(ref_mes[3,t_n],ref_mes[4,t_n],ref_mes[5,t_n],color='green')
        ax4.scatter(ref_mes[0,t_n],ref_mes[1,t_n],ref_mes[2,t_n],color='black')
        ax4.set_xlim(-10,10)
        ax4.set_ylim(-10,10)
        ax4.set_zlim(2,4)
        dx = ref_mes[0,t_n] - ref_mes[3,t_n]
        dy = ref_mes[1,t_n] - ref_mes[4,t_n]
        dz = ref_mes[2,t_n] - ref_mes[5,t_n]
        dist = math.sqrt(dx**2 + dy**2 + dz**2)
        print('dx dy dz',dx,dy,dz)
        print(dist,' m')
        plt.show()
        plt.pause(0.001)
        ax4.clear()


    
