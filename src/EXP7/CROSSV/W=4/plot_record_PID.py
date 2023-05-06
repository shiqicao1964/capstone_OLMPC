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

    error_recordPID = np.genfromtxt('pos_error_record_PID.out',delimiter=',')[:,:]

    print(error_recordPID.shape)

    print('mean square error of dynamic model : (x , y , z)', np.mean(error_recordPID,1))
    print('average distance error of dynamic model :', np.mean(np.sqrt(np.sum(error_recordPID,0))))

    ref_mesPID = np.genfromtxt('ref_mes_PID.out',delimiter=',')[:,:]


    x_w = 4
    H = 1.5

    fig = plt.figure(figsize = (18,6))
    fig.subplots_adjust(wspace=0.1, hspace=0.1)
    ax1 = fig.add_subplot(111, projection='3d')

    ax1.plot(ref_mesPID[3],ref_mesPID[4],ref_mesPID[5],color='green')
    ax1.plot(ref_mesPID[0],ref_mesPID[1],ref_mesPID[2],color='red')
    ax1.set_ylim(-x_w,x_w)
    ax1.set_zlim(H-1,H+1)

    plt.show()
    time.sleep(1)


    fig2D = plt.figure(figsize = (12,12))
    ax5 = fig2D.add_subplot()
    plt.ion()
    for i in range(500):
        t_n = i
        ax5.plot(ref_mesPID[3],ref_mesPID[4],color='green')
        ax5.plot(ref_mesPID[0],ref_mesPID[1],color='red')
        ax5.scatter(ref_mesPID[3,t_n],ref_mesPID[4,t_n],s = 80,color='green')
        ax5.scatter(ref_mesPID[0,t_n],ref_mesPID[1,t_n],s = 80,color='black')
        ax5.set_xlim(-4,4)
        ax5.set_ylim(-4,4)
        dx = ref_mesPID[0,t_n] - ref_mesPID[3,t_n]
        dy = ref_mesPID[1,t_n] - ref_mesPID[4,t_n]
        dz = ref_mesPID[2,t_n] - ref_mesPID[5,t_n]
        dist = math.sqrt(dx**2 + dy**2 + dz**2)
        print('dx dy dz',dx,dy,dz)
        print(dist,' m')
        plt.show()
        plt.pause(0.001)
        ax5.clear()


    
