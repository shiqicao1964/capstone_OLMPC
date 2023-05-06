
import os
import sys
import shutil
import casadi as cs
import numpy as np
from copy import copy
import matplotlib
import matplotlib.pyplot as plt
from acados_template import AcadosOcp, AcadosOcpSolver, AcadosModel
from math import sqrt
import math 
from casadi.casadi import *
import time
from tqdm import trange
from scipy.optimize import minimize
import numpy as np
from skopt import gp_minimize
from sklearn.model_selection import cross_val_score
from skopt.space import Real
import warnings
warnings.filterwarnings('ignore')
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C



class px4_quad:
    def __init__(self):
        # Quadrotor intrinsic parameters
        self.J = np.array([.03, .03, .06])  # N m s^2 = kg m^2
        self.mass = 1.5  # kg

        # Length of motor to CoG segment
        self.length = 0.47 / 2  # m
        self.max_thrust = 15
        self.g = np.array([[0], [0], [9.81]])  # m s^-2
        h = np.cos(np.pi / 4) * self.length
        self.x_f = np.array([h, -h, -h, h])
        self.y_f = np.array([-h, -h, h, h])
        self.c = 0.013  # m   (z torque generated by each motor)
        self.z_l_tau = np.array([-self.c, self.c, -self.c, self.c])

        # Input constraints
        self.max_input_value = 1  # Motors at full thrust
        self.min_input_value = 0  # Motors turned off
        self.min_u = self.min_input_value
        self.max_u = self.max_input_value

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
    return x[:-1], y[:-1], z[:-1],t_new,vx,vy,vz,T_onecircle

def ellipse_trag(speed = 3,x_w = 3,y_w = 4,z_w = 0,H = 5,dT = 0.05,sim_t = 100):
    t_abl = [0]
    t = np.linspace(0, 2*np.pi, num=10000)
    x = x_w * np.cos(t) 
    y = y_w * np.sin(t)
    z = H + z_w*np.cos(t)
    x0 = x_w * np.cos(0) 
    y0 = y_w * np.sin(0) 
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
    x = x_w * np.cos(t) 
    y = y_w * np.sin(t) 
    z = H + z_w*np.cos(t)
    x0 = x_w * np.cos(0) 
    y0 = y_w * np.sin(0) 
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
    x = x_w * np.cos(t_new) 
    y = y_w * np.sin(t_new)
    z =  H + z_w*np.cos(t_new)
    vx = np.array(vx)
    vy = np.array(vy)
    vz = np.array(vz)
    return x[:-1], y[:-1], z[:-1],t_new,vx,vy,vz,T_onecircle


def linear_quad_model():

    # Declare model variables
    roll = cs.MX.sym('roll')  # position
    pitch = cs.MX.sym('pitch')
    yaw = cs.MX.sym('yaw')

    x_ = cs.MX.sym('x_')
    y_ = cs.MX.sym('y_')
    z_ = cs.MX.sym('z_')

    p = cs.MX.sym('p')
    q = cs.MX.sym('q')
    r = cs.MX.sym('r')

    vx = cs.MX.sym('vx')
    vy = cs.MX.sym('vy')
    vz = cs.MX.sym('vz')
    # Full state vector (12-dimensional)
    x = cs.vertcat(x_,y_,z_,roll,pitch,yaw,vx,vy,vz,p,q,r)
    state_dim = 12

    # Control input vector
    u1 = cs.MX.sym('u1')
    u2 = cs.MX.sym('u2')
    u3 = cs.MX.sym('u3')
    u4 = cs.MX.sym('u4')
    u = cs.vertcat(u1, u2, u3, u4)

    my_quad = px4_quad()
    # p_dynamics
    pos_dynamics = cs.vertcat(vx,vy,vz)

    #q_dynamics
    angle_dynamics = cs.vertcat(
        p+r*pitch+q*roll*pitch,
        q-r*roll,
        r+q*roll)

    # v_dynamics
    g = -9.8
    ft = (u1 + u2 + u3 + u4)*my_quad.max_thrust
    taux = (u3 - u1)*my_quad.max_thrust*my_quad.length
    tauy = (u4 - u2)*my_quad.max_thrust*my_quad.length
    tauz = (u2 + u4 - u1 - u3)*my_quad.max_thrust*my_quad.length
    v_dynamics = cs.vertcat(
        r*vy-q*vz-g*pitch,
        p*vz-r*vx+g*roll,
        q*vx-p*vy + g + ft/my_quad.mass)
    #w_dynamics 
    w_dynamics = cs.vertcat(
            (my_quad.J[1] - my_quad.J[2])/my_quad.J[0] * r * q + taux/my_quad.J[0],
            (my_quad.J[2] - my_quad.J[0])/my_quad.J[1] * r * p + tauy/my_quad.J[1],
            (my_quad.J[0] - my_quad.J[1])/my_quad.J[2] * p * q + tauz/my_quad.J[2])

    
    pdot = cs.MX.sym('pdot', 3)  # position
    qdot = cs.MX.sym('adot', 3)  # angle roll pitch yaw
    vdot = cs.MX.sym('vdot', 3)  # velocity
    rdot = cs.MX.sym('rdot', 3)  # angle rate
    xdot = cs.vertcat(pdot, qdot, vdot, rdot)

    normails = cs.vertcat(pos_dynamics, angle_dynamics, v_dynamics, w_dynamics)
    f_impl = xdot - normails

    model_name = 'px4_quad_linear_model'

    # Dynamics model
    model = AcadosModel()
    model.f_expl_expr = normails
    model.f_impl_expr = f_impl
    model.x = x
    model.xdot = xdot
    model.u = u
    model.p = []
    model.name = 'px4_quad_linear_model'

    return model

def DT_linear_model(dT,name):
    model = linear_quad_model()
    x = model.x
    u = model.u
    model.name = f'px4_quad_linear_model_{name}'
    ode = cs.Function('ode',[x, u], [model.f_expl_expr])
    # set up Rk4
    k1 = ode(x,       u)
    k2 = ode(x+dT/2*k1,u)
    k3 = ode(x+dT/2*k2,u)
    k4 = ode(x+dT*k3,  u)
    xf = x + dT/6 * (k1 + 2*k2 + 2*k3 + k4)

    model.disc_dyn_expr = xf
    return model

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


class GPR:
    def __init__(self, optimize=True):
        self.is_fit = False
        self.train_X, self.train_y = None, None
        self.params = {"l": 0.3, "sigma_f": 0.2}
        self.optimize = optimize
    def kernel(self, x1, x2):
        dist_matrix = np.sum(x1**2, 1).reshape(-1, 1) + np.sum(x2**2, 1) - 2 * np.dot(x1, x2.T)
        return self.params["sigma_f"] ** 2 * np.exp(-0.5 / self.params["l"] ** 2 * dist_matrix)
    
    def fit(self, X, y,bounds_,alpha):
        # store train data
        self.train_X = np.asarray(X)
        self.train_y = np.asarray(y)
        self.params['sigma_f'] = np.std(y)
         # hyper parameters optimization
        def negative_log_likelihood_loss(params):
            self.params["l"]= params
            Kyy = self.kernel(self.train_X, self.train_X) + alpha * np.eye(len(self.train_X))
            loss = 0.5 * self.train_y.T.dot(np.linalg.inv(Kyy)).dot(self.train_y) + 0.5 * np.linalg.slogdet(Kyy)[1] + 0.5 * len(self.train_X) * np.log(2 * np.pi)
            return np.sum(loss.ravel())

        if self.optimize:
            res = minimize(negative_log_likelihood_loss, [self.params["l"]],
                   bounds=bounds_,
                   method='L-BFGS-B')
            self.params["l"] = res.x[0]
        self.is_fit = True


def DT_gp_model(dT,name,predict,measurement,control,total_buff_size):
    model = linear_quad_model()
    x = model.x
    u = model.u
    model.name = f'px4_quad_linear_model_{name}'
    ode = cs.Function('ode',[x, u], [model.f_expl_expr])
    # set up Rk4
    k1 = ode(x,       u)
    k2 = ode(x+dT/2*k1,u)
    k3 = ode(x+dT/2*k2,u)
    k4 = ode(x+dT*k3,  u)
    xf = x + dT/6 * (k1 + 2*k2 + 2*k3 + k4)
    
    # fit gp model
    t0 = time.time()

    x1 = (measurement[:,0:-1])
    x2 = (control[:,0:-1])
    input_state = np.concatenate(( x1 , x2 ), axis=0)
    error_y = (measurement[:,1:] - predict[:,1:])

    # take 60 points for optimal regarlization
    down_sample_factor_R = int(total_buff_size / 60 )
    input_state_R = input_state[:,::down_sample_factor_R]
    error_y_R = error_y[:,::down_sample_factor_R]

    # use 100 points in GP 
    down_sample_factor = int(total_buff_size / 30 )
    input_state = input_state[:,::down_sample_factor]
    error_y = error_y[:,::down_sample_factor]


    # find optimal reguarlization\
    optimal_alpha_1 = find_optimal_alpha(input_state_R.T,error_y_R[[2,6,7,8],:].T)
    
    # find optimal L 
    gpr1 = GPR()
    gpr1.fit(input_state.T,error_y[[6],:].T , [(0.1, 15)] ,optimal_alpha_1)


    gpr2 = GPR()
    gpr2.fit(input_state.T,error_y[[7],:].T , [(0.1, 15)] ,optimal_alpha_1)


    gpr3 = GPR()
    gpr3.fit(input_state.T,error_y[[2,8],:].T ,[(0.1, 15)] ,optimal_alpha_1)






    # use optimal L train GP error model
    l_1 = gpr1.params['l']
    sig_f_1 = gpr1.params['sigma_f']
    X = input_state
    Y = error_y[[6],:]
    error_1 = fit_gp(sig_f_1,l_1,X,Y,x,u,optimal_alpha_1)

    l_2 = gpr2.params['l']
    sig_f_2 = gpr2.params['sigma_f']
    X = input_state
    Y = error_y[[7],:]
    error_2 = fit_gp(sig_f_2,l_2,X,Y,x,u,optimal_alpha_1)

    l_3 = gpr3.params['l']
    sig_f_3 = gpr3.params['sigma_f']
    X = input_state
    Y = error_y[[2,8],:]
    error_3 = fit_gp(sig_f_3,l_3,X,Y,x,u,optimal_alpha_1)





    error_1 = cs.vertcat(cs.MX([0,0,0,0,0,0]),error_1,cs.MX([0,0]),cs.MX([0,0,0]))
    error_2 = cs.vertcat(cs.MX([0,0,0,0,0,0,0]),error_2,cs.MX([0]),cs.MX([0,0,0]))
    error_3 = cs.vertcat(cs.MX([0,0]),error_3[0],cs.MX([0,0,0,0,0]),error_3[1],cs.MX([0,0,0]))


    print('=======================================================================')
    print('============================fit result ================================')
    print('=======================================================================')
    print(f'l_1:',l_1,'sig_f_1:',sig_f_1 )
    print(f'l_2:',l_2,'sig_f_2:',sig_f_2 )
    print(f'l_3:',l_3,'sig_f_3:',sig_f_3 )
    print('optimal_alpha_1:',optimal_alpha_1)
    print('input_state.shape',input_state.shape)
    
    print('down_sample_factor',down_sample_factor)
    print('size of train set:',error_y.shape)
    print('time for fit GP model',time.time() - t0)
    print('=======================================================================')
    print('=======================================================================')
    print('=======================================================================')

    model.disc_dyn_expr = xf + error_1 + error_2 + error_3
    return model


def find_optimal_alpha(X,y):
    search_list_alpha = np.array([1e-3,5e-4,1e-4,5e-5,1e-5,5e-6,1e-6])
    recode_score=np.array([])
    alpha_lst = np.array([])
    t0 = time.time()
    print('X',X.shape,'y',y.shape)
    for i in range(len(search_list_alpha)):
        alpha = search_list_alpha[i]
        kernel = C(0.2, (1e-3, 1e2)) * RBF(1.0, (1e-2, 1e1))
        gp = GaussianProcessRegressor(kernel=kernel, alpha=alpha)
        cv_scores = cross_val_score(gp, X, y, cv=3, scoring='neg_mean_squared_error')
        recode_score = np.append(recode_score,np.mean(cv_scores))
        alpha_lst = np.append(alpha_lst,alpha)
    
    optimal_alpha = alpha_lst[recode_score == np.max(recode_score)]
    print('optimal_alpha',optimal_alpha,'time used',time.time()-t0)
    return optimal_alpha

def fit_gp(sig_f,l,X,Y,x,u,alpha):
    
    K = kernel(X,X,sig_f,l)
    x1 = cs.vertcat(x,u).T
    x2 = X.T
    dist_matrix = cs.sum2(x1**2) + cs.sum2(x2**2) - ((cs.mtimes(x1, x2.T))*2).T
    Kstar = (sig_f ** 2 * np.exp(-0.5 / l ** 2 * dist_matrix)).T
    error = cs.mtimes ( cs.mtimes(Kstar,np.linalg.inv(K + alpha * np.eye(X.shape[1])) ), Y.T ).T
    return error
    
def acados_settinngs(acados_models,solver_options = None,t_horizon = 1,N = 20,build=True, generate=True):
    
    my_quad = px4_quad()
    
    q_cost = np.array([20, 20, 22, 2, 2, 2, 1, 1, 1, 1, 1, 1])
    r_cost = np.array([0.1, 0.1, 0.1, 0.1])
     
    nx = acados_models.x.size()[0]
    nu = acados_models.u.size()[0]
    ny = nx + nu
    ny_e = nx

    n_param = acados_models.p.size()[0] if isinstance(acados_models.p, cs.MX) else 0
    ocp = AcadosOcp()
    ocp.model = acados_models
    ocp.dims.N = N
    ocp.solver_options.tf = t_horizon

    # Initialize parameters

    ocp.dims.np = n_param
    ocp.parameter_values = np.zeros(n_param)

    ocp.cost.cost_type = 'NONLINEAR_LS'
    ocp.cost.cost_type_e = 'NONLINEAR_LS'
    
    ocp.model.cost_y_expr = cs.vertcat(ocp.model.x, ocp.model.u)
    ocp.model.cost_y_expr_e = ocp.model.x


    ocp.cost.W = np.diag(np.concatenate((q_cost, r_cost)))
    ocp.cost.W_e = np.diag(q_cost)
    ocp.cost.W_e *= 0


    # Initial reference trajectory (will be overwritten)
    x_ref = np.zeros(nx)
    ocp.cost.yref = np.concatenate((x_ref, np.array([0.0, 0.0, 0.0, 0.0])))
    ocp.cost.yref_e = x_ref
    ocp.constraints.x0 = x_ref

    # Set constraints
    ocp.constraints.lbu = np.array([my_quad.min_u] * 4)
    ocp.constraints.ubu = np.array([my_quad.max_u] * 4)



    ocp.constraints.idxbu = np.array([0,1,2,3])
    # Solver options
    ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
    ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
    ocp.solver_options.hpipm_mode =  'BALANCE'           #'SPEED'
    ocp.solver_options.integrator_type = 'DISCRETE'
    ocp.solver_options.print_level = 0
    ocp.solver_options.nlp_solver_type = 'SQP_RTI'
    ocp.solver_options.regularize_method = 'PROJECT'
    ocp.solver_options.qp_solver_warm_start = 1
    ocp.solver_options.nlp_solver_max_iter = 10
    t1 = time.time()
    acados_solver = AcadosOcpSolver(ocp, json_file='acados_ocp.json',build=build, generate=generate)
    print('=============================FINISHED nonlinear_LS ACADOS SOLVER SETTINGS ===========================')
    print('time for build the solver:',time.time()-t1)
    return acados_solver



def run_solver(N,model,acados_solver,initial_state,ref):
    
    u_target = np.zeros((N+1,4))
    ref = np.concatenate((ref, u_target),axis = 1)
    for j in range(N):
        acados_solver.set(j, "yref", ref[j])
    acados_solver.set(N, "yref", ref[N][:-4])
    
    # Set initial state.
    x_init = initial_state
    x_init = np.stack(x_init)
    # Set initial condition, equality constraint
    acados_solver.set(0, 'lbx', x_init)
    acados_solver.set(0, 'ubx', x_init)

    # Solve OCP
    acados_solver.solve()
    # get vx vy vz 
    x_next = acados_solver.get(8, "x")
    vx_next = x_next[6]
    vy_next = x_next[7]
    vz_next = x_next[8]
    p_next = x_next[9]
    q_next = x_next[10]
    r_next = x_next[11]	
    control = acados_solver.get(0, "u")
    time_tot = acados_solver.get_stats('time_tot')
    sqp_iter = acados_solver.get_stats('sqp_iter')
    #print('cpu time',time_tot)
    #print('sqp_iter',sqp_iter)

    return vx_next,vy_next,vz_next,p_next,q_next,r_next,control

def solve_DT_nextState(model,input_u ,current_x):
    x = model.x
    u = model.u
    xf = model.disc_dyn_expr
    DTsolution = cs.Function('f',[u,x],[xf])
    result = DTsolution(input_u,current_x)

    return result

def rotation_matrix(yaw, pitch, roll):
    cy = np.cos(yaw)
    sy = np.sin(yaw)
    cp = np.cos(pitch)
    sp = np.sin(pitch)
    cr = np.cos(roll)
    sr = np.sin(roll)

    R_yaw = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    R_pitch = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    R_roll = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])

    R_world_to_body = np.dot(R_yaw, np.dot(R_pitch, R_roll))

    return R_world_to_body



