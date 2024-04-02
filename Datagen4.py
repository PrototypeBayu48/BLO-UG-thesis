import numpy as np
import scipy as sc
import random as rand
from sklearn import preprocessing, linear_model
import matplotlib.pyplot as plt
import dill
import os
import torch

from core.controllers import PDController
from core.dynamics import LinearSystemDynamics, ConfigurationDynamics

from koopman_core.controllers import OpenLoopController, MPCController,BilinearFBLinController, PerturbedController, LinearLiftedController
from koopman_core.dynamics import LinearLiftedDynamics, BilinearLiftedDynamics
from koopman_core.learning import Edmd, BilinearEdmd
from koopman_core.basis_functions import PlanarQuadBasis
from koopman_core.learning.utils import differentiate_vec
from koopman_core.systems import PlanarQuadrotorForceInput

from koopman_core.learning import KoopDnn_modified, KoopmanNetCtrl
from koopman_core.util import fit_standardizer

from koopman_core.util import evaluate_ol_pred
from tabulate import tabulate

from koopman_core.controllers import MPCController, NonlinearMPCControllerNb, BilinearMPCControllerNb

from matplotlib.ticker import MaxNLocator

from koopman_core.controllers import PerturbedController

from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from scipy import ndimage

import do_mpc
from numpy import sin, cos, tan
import random as rand

from time import time

def quadmpc(xinitial,xfinal,xnext,u0,cpoint_tol,maxiter,trajgen = 0):
    model_type = 'continuous' # either 'discrete' or 'continuous'
    model = do_mpc.model.Model(model_type)

    x = model.set_variable(var_type='_x', var_name='x', shape=(1,1))
    y = model.set_variable(var_type='_x', var_name='y', shape=(1,1))
    z = model.set_variable(var_type='_x', var_name='z', shape=(1,1))
    phi = model.set_variable(var_type='_x', var_name='phi', shape=(1,1))
    theta = model.set_variable(var_type='_x', var_name='theta', shape=(1,1))
    psi = model.set_variable(var_type='_x', var_name='psi', shape=(1,1))
    dx = model.set_variable(var_type='_x', var_name='dx', shape=(1,1))
    dy = model.set_variable(var_type='_x', var_name='dy', shape=(1,1))
    dz = model.set_variable(var_type='_x', var_name='dz', shape=(1,1))
    dphi = model.set_variable(var_type='_x', var_name='dphi', shape=(1,1))
    dtheta = model.set_variable(var_type='_x', var_name='dtheta', shape=(1,1))
    dpsi = model.set_variable(var_type='_x', var_name='dpsi', shape=(1,1))
    w1 = model.set_variable(var_type='_u', var_name='w1')
    w2 = model.set_variable(var_type='_u', var_name='w2')
    w3 = model.set_variable(var_type='_u', var_name='w3')
    w4 = model.set_variable(var_type='_u', var_name='w4')

    mass = 0.0336
    Ixx = 1.7e-5
    Iyy = 1.7e-5
    Izz = 2.9e-5
    gravity = 9.81
    l = 0.04
    ct = 3.1582e-10
    cq = 7.9379e-12

    hover_thrust = mass*gravity
    omega_thrust = (mass*gravity)/(4*ct)
    max_omega_thrust = (0.057*gravity)/(4*ct)

    c1 = (Iyy - Izz)/Ixx
    c2 = (Izz - Ixx)/Iyy
    c3 = (Iyy - Ixx)/Izz

    model.set_rhs('x', dx)
    model.set_rhs('y', dy)
    model.set_rhs('z', dz)
    model.set_rhs('phi', dphi)
    model.set_rhs('theta', dtheta)
    model.set_rhs('psi', dpsi)

    dx_next = ((cos(psi)*sin(theta)*cos(phi) + sin(psi)*sin(phi))/mass)*((ct)*((w1) + (w2) + (w3) + (w4)))
    dy_next = ((sin(psi)*sin(theta)*cos(phi) - cos(psi)*sin(phi))/mass)*((ct)*((w1) + (w2) + (w3) + (w4)))
    dz_next = -gravity + ((cos(theta)*cos(phi))/mass)*((ct)*((w1) + (w2) + (w3) + (w4)))
    dphi_next = dtheta*dpsi*c1 + ((ct*l)*((w2) - (w4)))/Ixx
    dtheta_next = dphi*dpsi*c2 + ((ct*l)*(- (w1) + (w3)))/Iyy
    dpsi_next = dphi*dtheta*c3 + ((cq)*(- (w1) + (w2) - (w3) + (w4)))/Izz

    model.set_rhs('dx', dx_next)
    model.set_rhs('dy', dy_next)
    model.set_rhs('dz', dz_next)
    model.set_rhs('dphi', dphi_next)
    model.set_rhs('dtheta', dtheta_next)
    model.set_rhs('dpsi', dpsi_next)

    # Build the model
    model.setup()

    mpc = do_mpc.controller.MPC(model)

    setup_mpc = {
        'n_horizon': 25,
        't_step': 0.02,
        'n_robust': 0,
        'store_full_solution': True,
        'nlpsol_opts': {'ipopt.linear_solver': 'MA27'}
    }

    mpc.set_param(**setup_mpc)

    mpc.settings.supress_ipopt_output()

    xf = xfinal[0]
    yf = xfinal[1]
    zf = xfinal[2]
    dx_t = xf - xnext[0]
    dy_t = yf - xnext[1]
    dz_t = zf - xnext[2]

    mterm = 10*(x - xf)**2 + 10*(y - yf)**2 + 10*(z - zf)**2 + (dx - dx_t)**2 + (dy - dy_t)**2 + (dz - dz_t)**2
    lterm = 10*(x - xf)**2 + 10*(y - yf)**2 + 10*(z - zf)**2 + (dx - dx_t)**2 + (dy - dy_t)**2 + (dz - dz_t)**2
    
    mpc.set_objective(mterm=mterm, lterm=lterm)

    mpc.set_rterm(w1=1e-16,w2=1e-16,w3=1e-16,w4=1e-16) # input penalty

    # lower bounds of the states
    mpc.bounds['lower','_x','x'] = -20.0
    mpc.bounds['lower','_x','y'] = -20.0
    mpc.bounds['lower','_x','z'] = 0.0
    mpc.bounds['lower','_x','phi'] = -np.pi/2
    mpc.bounds['lower','_x','theta'] = -np.pi/2
    mpc.bounds['lower','_x','psi'] = -np.pi/2
    mpc.bounds['lower','_x','dx'] = -20.0
    mpc.bounds['lower','_x','dy'] = -20.0
    mpc.bounds['lower','_x','dz'] = -20.0
    mpc.bounds['lower','_x','dphi'] = -20.0
    mpc.bounds['lower','_x','dtheta'] = -20.0
    mpc.bounds['lower','_x','dpsi'] = -20.0

    # upper bounds of the states
    mpc.bounds['upper','_x','x'] = 20.0
    mpc.bounds['upper','_x','y'] = 20.0
    mpc.bounds['upper','_x','z'] = 40.0
    mpc.bounds['upper','_x','phi'] = np.pi/2
    mpc.bounds['upper','_x','theta'] = np.pi/2
    mpc.bounds['upper','_x','psi'] = np.pi/2
    mpc.bounds['upper','_x','dx'] = 20.0
    mpc.bounds['upper','_x','dy'] = 20.0
    mpc.bounds['upper','_x','dz'] = 20.0
    mpc.bounds['upper','_x','dphi'] = 20.0
    mpc.bounds['upper','_x','dtheta'] = 20.0
    mpc.bounds['upper','_x','dpsi'] = 20.0

    # lower bounds of the input
    mpc.bounds['lower','_u','w1'] = 0.0
    mpc.bounds['lower','_u','w2'] = 0.0
    mpc.bounds['lower','_u','w3'] = 0.0
    mpc.bounds['lower','_u','w4'] = 0.0

    # upper bounds of the input
    mpc.bounds['upper','_u','w1'] = max_omega_thrust
    mpc.bounds['upper','_u','w2'] = max_omega_thrust
    mpc.bounds['upper','_u','w3'] = max_omega_thrust
    mpc.bounds['upper','_u','w4'] = max_omega_thrust

    mpc.setup()

    simulator = do_mpc.simulator.Simulator(model)

    simulator.set_param(t_step = 0.02)

    simulator.setup()

    # Initial state
    x0 = np.array([[xinitial[0]], [xinitial[1]], [xinitial[2]], [xinitial[3]], [xinitial[4]], [xinitial[5]], [xinitial[6]], [xinitial[7]], [xinitial[8]], [xinitial[9]], [xinitial[10]], [xinitial[11]]])
    mpc.x0 = x0
    simulator.x0 = x0
    mpc.u0 = u0
    simulator.u0 = u0

    # Use initial state to set the initial guess.
    mpc.set_initial_guess()

    iter = 0
    cpoint = np.sqrt((x0[0] - xf)**2 + (x0[1] - yf)**2 + (x0[2] - zf)**2)
    t1 = time()
    if trajgen == 0:
        if np.sqrt(dx_t**2 + dy_t**2 + dz_t**2) > 0:
            while cpoint > cpoint_tol:
                iter = iter + 1
                print(iter)
                u0 = mpc.make_step(x0)
                x0 = simulator.make_step(u0)
                cpoint = np.sqrt((x0[0] - xf)**2 + (x0[1] - yf)**2 + (x0[2] - zf)**2)
                if iter > maxiter:
                    break
        else:
            while cpoint > 0.0:
                iter = iter + 1
                print(iter)
                u0 = mpc.make_step(x0)
                x0 = simulator.make_step(u0)
                cpoint = np.sqrt((x0[0] - xf)**2 + (x0[1] - yf)**2 + (x0[2] - zf)**2)
                if iter > maxiter:
                    break
    else:
        for n in range(maxiter + 1):
            iter = iter + 1
            print(iter)
            u0 = mpc.make_step(x0)
            x0 = simulator.make_step(u0)
    
    print(np.sqrt((x0[0] - xf)**2 + (x0[1] - yf)**2 + (x0[2] - zf)**2))
    print(time() - t1)

    results = mpc.data
    t_f = results['_time']
    x_f = results['_x']
    u_f = results['_u']

    t_f = np.squeeze(t_f,axis=1)

    return [t_f,x_f,u_f]

def quadmpc_koop(xinitial,xfinal,xnext,u0,cpoint_tol,maxiter,n,A_array,B_array,C_array,basis,trajgen = 0):
    model_type = 'discrete' # either 'discrete' or 'continuous'
    model = do_mpc.model.Model(model_type)

    z = model.set_variable(var_type='_x', var_name='z', shape=(n,1))
    w1 = model.set_variable(var_type='_u', var_name='w1')
    w2 = model.set_variable(var_type='_u', var_name='w2')
    w3 = model.set_variable(var_type='_u', var_name='w3')
    w4 = model.set_variable(var_type='_u', var_name='w4')

    mass = 0.0336
    Ixx = 1.7e-5
    Iyy = 1.7e-5
    Izz = 2.9e-5
    gravity = 9.81
    l = 0.04
    ct = 3.1582e-10
    cq = 7.9379e-12

    hover_thrust = mass*gravity
    omega_thrust = (mass*gravity)/(4*ct)
    max_omega_thrust = (0.057*gravity)/(4*ct)

    G1 = B_array[0,:,:]
    G2 = B_array[1,:,:]
    G3 = B_array[2,:,:]
    G4 = B_array[3,:,:]

    x_next = A_array@z + (G1@z)*w1 + (G2@z)*w2 + (G3@z)*w3 + (G4@z)*w4

    model.set_rhs('z', x_next)

    # Build the model
    model.setup()

    mpc = do_mpc.controller.MPC(model)

    setup_mpc = {
        'n_horizon': 25,
        't_step': 0.02,
        'n_robust': 0,
        'state_discretization': 'discrete',
        'store_full_solution': True,
        'nlpsol_opts': {'ipopt.linear_solver': 'MA27'}
    }

    mpc.set_param(**setup_mpc)

    mpc.settings.supress_ipopt_output()

    xf = xfinal[0]
    yf = xfinal[1]
    zf = xfinal[2]
    dx_t = xf - xnext[0]
    dy_t = yf - xnext[1]
    dz_t = zf - xnext[2]

    mterm = 10*(z[1] - xf)**2 + 10*(z[2] - yf)**2 + 10*(z[3] - zf)**2 + (z[7] - dx_t)**2 + (z[8] - dy_t)**2 + (z[9] - dz_t)**2
    lterm = 10*(z[1] - xf)**2 + 10*(z[2] - yf)**2 + 10*(z[3] - zf)**2 + (z[7] - dx_t)**2 + (z[8] - dy_t)**2 + (z[9] - dz_t)**2

    mpc.set_objective(mterm=mterm, lterm=lterm)

    mpc.set_rterm(w1=1.0,w2=1.0,w3=1.0,w4=1.0) # input penalty

    max_x = np.array([10.0, 10.0, 40.0, np.pi/2, np.pi/2, np.pi/2, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0])
    max_z = (np.inf)*(np.ones(n))
    max_z[1] = max_x[0]
    max_z[2] = max_x[1]
    max_z[3] = max_x[2]
    max_z[4] = max_x[3]
    max_z[5] = max_x[4]
    max_z[6] = max_x[5]
    max_z[7] = max_x[6]
    max_z[8] = max_x[7]
    max_z[9] = max_x[8]
    max_z[10] = max_x[9]
    max_z[11] = max_x[10]
    max_z[12] = max_x[11]
    min_z = -max_z
    min_z[3] = 0.0

    # lower bounds of the states
    mpc.bounds['lower','_x','z'] = min_z

    # upper bounds of the states
    mpc.bounds['upper','_x','z'] = max_z

    # lower bounds of the input
    mpc.bounds['lower','_u','w1'] = 0.0
    mpc.bounds['lower','_u','w2'] = 0.0
    mpc.bounds['lower','_u','w3'] = 0.0
    mpc.bounds['lower','_u','w4'] = 0.0

    # upper bounds of the input
    mpc.bounds['upper','_u','w1'] = 1.0
    mpc.bounds['upper','_u','w2'] = 1.0
    mpc.bounds['upper','_u','w3'] = 1.0
    mpc.bounds['upper','_u','w4'] = 1.0

    mpc.setup()

    simulator = do_mpc.simulator.Simulator(model)

    simulator.set_param(t_step = 0.02)

    simulator.setup()

    # Initial state
    x0 = np.array([xinitial[0], xinitial[1], xinitial[2], xinitial[3], xinitial[4], xinitial[5], xinitial[6], xinitial[7], xinitial[8], xinitial[9], xinitial[10], xinitial[11]])
    z0 = np.transpose(basis(x0))
    mpc.x0 = z0
    simulator.x0 = z0
    mpc.u0 = u0
    simulator.u0 = u0

    # Use initial state to set the initial guess.
    mpc.set_initial_guess()

    iter = 0
    cpoint = np.sqrt((z0[1] - xf)**2 + (z0[2] - yf)**2 + (z0[3] - zf)**2)
    t1 = time()
    if trajgen == 0:
        if np.sqrt(dx_t**2 + dy_t**2 + dz_t**2) > 0:
            while cpoint > cpoint_tol:
                iter = iter + 1
                print(iter)
                u0 = mpc.make_step(z0)
                z0 = simulator.make_step(u0)
                cpoint = np.sqrt((z0[1] - xf)**2 + (z0[2] - yf)**2 + (z0[3] - zf)**2)
                if iter > maxiter:
                    break
        else:
            while cpoint > 0.0:
                iter = iter + 1
                print(iter)
                u0 = mpc.make_step(z0)
                z0 = simulator.make_step(u0)
                cpoint = np.sqrt((z0[1] - xf)**2 + (z0[2] - yf)**2 + (z0[3] - zf)**2)
                if iter > maxiter:
                    break
    else:
        for n in range(maxiter + 1):
            iter = iter + 1
            print(iter)
            u0 = mpc.make_step(z0)
            z0 = simulator.make_step(u0)

    print(np.sqrt((z0[1] - xf)**2 + (z0[2] - yf)**2 + (z0[3] - zf)**2))
    print(time() - t1)

    results = mpc.data
    t_f = results['_time']
    z_f = results['_x','z']
    u_f = results['_u']

    z_f = np.transpose(z_f)
    x_f = np.matmul(C_array,z_f)
    x_f = np.transpose(x_f)

    t_f = np.squeeze(t_f,axis=1)

    return [t_f,x_f,u_f]

def quadsim(xset,uset,maxiter):
    model_type = 'continuous' # either 'discrete' or 'continuous'
    model = do_mpc.model.Model(model_type)

    x = model.set_variable(var_type='_x', var_name='x', shape=(1,1))
    y = model.set_variable(var_type='_x', var_name='y', shape=(1,1))
    z = model.set_variable(var_type='_x', var_name='z', shape=(1,1))
    phi = model.set_variable(var_type='_x', var_name='phi', shape=(1,1))
    theta = model.set_variable(var_type='_x', var_name='theta', shape=(1,1))
    psi = model.set_variable(var_type='_x', var_name='psi', shape=(1,1))
    dx = model.set_variable(var_type='_x', var_name='dx', shape=(1,1))
    dy = model.set_variable(var_type='_x', var_name='dy', shape=(1,1))
    dz = model.set_variable(var_type='_x', var_name='dz', shape=(1,1))
    dphi = model.set_variable(var_type='_x', var_name='dphi', shape=(1,1))
    dtheta = model.set_variable(var_type='_x', var_name='dtheta', shape=(1,1))
    dpsi = model.set_variable(var_type='_x', var_name='dpsi', shape=(1,1))
    w1 = model.set_variable(var_type='_u', var_name='w1')
    w2 = model.set_variable(var_type='_u', var_name='w2')
    w3 = model.set_variable(var_type='_u', var_name='w3')
    w4 = model.set_variable(var_type='_u', var_name='w4')

    mass = 0.0336
    Ixx = 1.7e-5
    Iyy = 1.7e-5
    Izz = 2.9e-5
    gravity = 9.81
    l = 0.04
    ct = 3.1582e-10
    cq = 7.9379e-12

    hover_thrust = mass*gravity
    omega_thrust = (mass*gravity)/(4*ct)
    max_omega_thrust = (0.057*gravity)/(4*ct)

    c1 = (Iyy - Izz)/Ixx
    c2 = (Izz - Ixx)/Iyy
    c3 = (Iyy - Ixx)/Izz

    model.set_rhs('x', dx)
    model.set_rhs('y', dy)
    model.set_rhs('z', dz)
    model.set_rhs('phi', dphi)
    model.set_rhs('theta', dtheta)
    model.set_rhs('psi', dpsi)

    dx_next = ((cos(psi)*sin(theta)*cos(phi) + sin(psi)*sin(phi))/mass)*((ct)*((w1) + (w2) + (w3) + (w4)))
    dy_next = ((sin(psi)*sin(theta)*cos(phi) - cos(psi)*sin(phi))/mass)*((ct)*((w1) + (w2) + (w3) + (w4)))
    dz_next = -gravity + ((cos(theta)*cos(phi))/mass)*((ct)*((w1) + (w2) + (w3) + (w4)))
    dphi_next = dtheta*dpsi*c1 + ((ct*l)*((w2) - (w4)))/Ixx
    dtheta_next = dphi*dpsi*c2 + ((ct*l)*(- (w1) + (w3)))/Iyy
    dpsi_next = dphi*dtheta*c3 + ((cq)*(- (w1) + (w2) - (w3) + (w4)))/Izz

    model.set_rhs('dx', dx_next)
    model.set_rhs('dy', dy_next)
    model.set_rhs('dz', dz_next)
    model.set_rhs('dphi', dphi_next)
    model.set_rhs('dtheta', dtheta_next)
    model.set_rhs('dpsi', dpsi_next)

    # Build the model
    model.setup()

    simulator = do_mpc.simulator.Simulator(model)

    simulator.set_param(t_step = 0.02)

    simulator.setup()

    # Initial state
    x0 = np.array([[xset[0,0]], [xset[0,1]], [xset[0,2]], [xset[0,3]], [xset[0,4]], [xset[0,5]], [xset[0,6]], [xset[0,7]], [xset[0,8]], [xset[0,9]], [xset[0,10]], [xset[0,11]]])
    u0 = np.array([[uset[0,0]], [uset[0,1]], [uset[0,2]], [uset[0,3]]])
    simulator.x0 = x0
    simulator.u0 = u0
    
    iter = 0
    for n in range(maxiter):
            iter = iter + 1
            print(iter)
            u0 = np.array([[uset[n,0]], [uset[n,1]], [uset[n,2]], [uset[n,3]]])
            simulator.make_step(u0)

    results = simulator.data
    t_f = results['_time']
    x_f = results['_x']

    t_f = np.squeeze(t_f,axis=1)

    return [t_f,x_f]

def trajgen(tspan,xspan,yspan,zspan,ustart,cpoint_tol_trajgen,maxiter_trajgen,A_array,B_array,C_array,basis,koop = 0):
    trajgen_iter = 1
    print('Trajgen iteration =')
    print(trajgen_iter)
    x1 = np.array([xspan[0], yspan[0], zspan[0], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    x2 = np.array([xspan[1], yspan[1], zspan[1], 0.0, 0.0, 0.0])
    x3 = np.array([xspan[2], yspan[2], zspan[2], 0.0, 0.0, 0.0])
    t_targ = np.array([tspan[0]])
    if koop == 0:
        [tfinal,xfinal,ufinal] = quadmpc(x1,x2,x3,ustart,cpoint_tol_trajgen,maxiter_trajgen)
    else:
        [tfinal,xfinal,ufinal] = quadmpc_koop(x1,x2,x3,ustart,cpoint_tol_trajgen,maxiter_trajgen,23,A_array,B_array,C_array,basis)
    t_targ = np.concatenate((t_targ,[tfinal[len(tfinal)-1]]))
    ustep = np.array([[ufinal[len(tfinal)-1,0]], [ufinal[len(tfinal)-1,1]], [ufinal[len(tfinal)-1,2]], [ufinal[len(tfinal)-1,3]]])
    for n1 in range(2,len(tspan)):
        trajgen_iter = trajgen_iter + 1
        print('Trajgen iteration =')
        print(trajgen_iter)
        x1 = xfinal[len(tfinal)-1,:]
        x2 = np.array([xspan[n1], yspan[n1], zspan[n1], 0.0, 0.0, 0.0])
        if n1 == len(tspan)-1:
            x3 = np.array([xspan[n1], yspan[n1], zspan[n1], 0.0, 0.0, 0.0])
        else:
            x3 = np.array([xspan[n1+1], yspan[n1+1], zspan[n1+1], 0.0, 0.0, 0.0])
        if koop == 0:
            [tstep,xstep,ustep] = quadmpc(x1,x2,x3,ustep,cpoint_tol_trajgen,maxiter_trajgen)
        else:
            [tstep,xstep,ustep] = quadmpc_koop(x1,x2,x3,ustep,cpoint_tol_trajgen,maxiter_trajgen,23,A_array,B_array,C_array,basis)
        tstep = tfinal[len(tfinal)-1] + tstep
        xfinal = np.concatenate((xfinal[0:len(tfinal)-1,:],xstep))
        ufinal = np.concatenate((ufinal[0:len(tfinal)-1,:],ustep))
        tfinal = np.concatenate((tfinal[0:len(tfinal)-1],tstep))
        ustep = np.array([[ustep[len(tstep)-1,0]], [ustep[len(tstep)-1,1]], [ustep[len(tstep)-1,2]], [ustep[len(tstep)-1,3]]])
        t_targ = np.concatenate((t_targ,[tfinal[len(tfinal)-1]]))
    
    return [tfinal,t_targ,xfinal,ufinal]

n = 12
m = 4
dt = 0.02

mass = 0.0336
gravity = 9.81
ct = 3.1582e-10
hover_thrust = (mass*gravity)/(4*ct)
max_thrust = (0.057*gravity)/(4*ct)
uhover = np.array([[hover_thrust], [hover_thrust], [hover_thrust], [hover_thrust]])

print(hover_thrust)
print(max_thrust)

n_traj_train = 8
n_traj_val = int(0.25*n_traj_train)

datagen_iter = 12000

x_train = np.empty((n_traj_train, datagen_iter + 1, 12))
u_train = np.empty((n_traj_train, datagen_iter, 4))
x_val = np.empty((n_traj_val, datagen_iter + 1, 12))
u_val = np.empty((n_traj_val, datagen_iter, 4))

datagen_iter = 0

for n1 in range(n_traj_train):
    t_dataset_train = np.arange(0,240.0 + 2.0,2.0)
    x_dataset_train = np.zeros(len(t_dataset_train))
    y_dataset_train = np.zeros(len(t_dataset_train))
    z_dataset_train = np.zeros(len(t_dataset_train))
    for i in range(len(t_dataset_train)):
        x_dataset_train[i] = -0.3 + rand.random()*0.6
        y_dataset_train[i] = -0.3 + rand.random()*0.6
        z_dataset_train[i] = 0.1 + rand.random()*0.9
    [t_iter,t_iter_targ,x_iter,u_iter] = trajgen(t_dataset_train,x_dataset_train,y_dataset_train,z_dataset_train,uhover,0.0,100,1,1,1,1)
    u_iter = u_iter[0:len(t_iter)-1,:]
    x_train[n1,:,:] = x_iter
    u_train[n1,:,:] = u_iter
    t_train = t_iter
    datagen_iter = datagen_iter + 1
    print('iter =')
    print(datagen_iter)

for n2 in range(n_traj_val):
    t_dataset_val = np.arange(0,240.0 + 2.0,2.0)
    x_dataset_val = np.zeros(len(t_dataset_val))
    y_dataset_val = np.zeros(len(t_dataset_val))
    z_dataset_val = np.zeros(len(t_dataset_val))
    for i in range(len(t_dataset_val)):
        x_dataset_val[i] = -0.3 + rand.random()*0.6
        y_dataset_val[i] = -0.3 + rand.random()*0.6
        z_dataset_val[i] = 0.1 + rand.random()*0.9
    [t_iter,t_iter_targ,x_iter,u_iter] = trajgen(t_dataset_val,x_dataset_val,y_dataset_val,z_dataset_val,uhover,0.0,100,1,1,1,1)
    u_iter = u_iter[0:len(t_iter)-1,:]
    x_val[n2,:,:] = x_iter
    u_val[n2,:,:] = u_iter
    t_val = t_iter
    datagen_iter = datagen_iter + 1
    print('iter =')
    print(datagen_iter)

np.save('x_train_4',x_train)
np.save('u_train_4',u_train)
np.save('x_val_4',x_val)
np.save('u_val_4',u_val)

plt.figure()
plt.plot(t_val,x_val[n_traj_val - 1,:,0],label = 'x val')
plt.plot(t_val,x_val[n_traj_val - 1,:,1],label = 'y val')
plt.plot(t_val,x_val[n_traj_val - 1,:,2],label = 'z val')
plt.plot(t_dataset_val,x_dataset_val,'o',label = 'x val target')
plt.plot(t_dataset_val,y_dataset_val,'o',label = 'y val target')
plt.plot(t_dataset_val,z_dataset_val,'o',label = 'z val target')
plt.legend()
plt.show()

plt.figure()
ax = plt.axes(projection='3d')
ax.plot3D(x_val[n_traj_val - 1,:,0],x_val[n_traj_val - 1,:,1],x_val[n_traj_val - 1,:,2],label='val trajectory')
ax.plot3D(x_dataset_val,y_dataset_val,z_dataset_val,'o',label='val target trajectory')
plt.legend()
plt.show()

