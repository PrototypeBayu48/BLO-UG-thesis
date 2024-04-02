import numpy as np
import scipy as sc
import random as rand
from sklearn import preprocessing, linear_model
import matplotlib.pyplot as plt
import dill
import os
import torch

import do_mpc
from numpy import sin, cos, tan
import random as rand

from time import time

from KoopmanNetworks import BLOfunc_save

import pypose as pp

import pickle

def quadmpc(xstart,xfinal,datalen,u0,geffect = 0):

    itertime = np.zeros(datalen)
    cpointdata = np.zeros(datalen)
    cpointdiff = np.zeros(datalen)

    mass = 0.0336
    Ixx = 1.7e-5
    Iyy = 1.7e-5
    Izz = 2.9e-5
    gravity = 9.81
    l = 0.04
    ct = 3.1582e-10
    cq = 7.9379e-12

    R = (45e-3)/2
    vh = np.sqrt(((mass*gravity)/4)/(2*1.225*(np.pi)*(R**2)))

    hover_w2 = (mass*gravity)/(4*ct)
    max_w2 = (0.057*gravity)/(4*ct)
    hover_thrust = (mass*gravity)
    max_thrust = (0.057*gravity)
    max_tx = ct*l*max_w2
    max_ty = ct*l*max_w2
    max_tz = 2*cq*max_w2
    uhover = np.array([[hover_thrust], [0.0], [0.0], [0.0]])
    uhover_norm = uhover/(0.1*max_thrust)

    u0_norm = np.zeros((4,1))
    u0_norm[0,0] = u0[0,0]/(0.1*max_thrust)
    u0_norm[1,0] = u0[1,0]/(0.2*max_tx)
    u0_norm[2,0] = u0[2,0]/(0.2*max_ty)
    u0_norm[3,0] = u0[3,0]/(0.2*max_tz)

    Tarray = np.zeros((4,4))
    Tarray[0,:] = ct
    Tarray[1,1] = ct*l
    Tarray[1,3] = -ct*l
    Tarray[2,0] = -ct*l
    Tarray[2,2] = ct*l
    Tarray[3,0] = -cq
    Tarray[3,1] = cq
    Tarray[3,2] = -cq
    Tarray[3,3] = cq

    c1 = (Iyy - Izz)/Ixx
    c2 = (Izz - Ixx)/Iyy
    c3 = (Iyy - Ixx)/Izz

    geffectmodel_type = 'continuous' # either 'discrete' or 'continuous'
    geffectmodel = do_mpc.model.Model(geffectmodel_type)

    x = geffectmodel.set_variable(var_type='_x', var_name='x', shape=(1,1))
    y = geffectmodel.set_variable(var_type='_x', var_name='y', shape=(1,1))
    z = geffectmodel.set_variable(var_type='_x', var_name='z', shape=(1,1))
    phi = geffectmodel.set_variable(var_type='_x', var_name='phi', shape=(1,1))
    theta = geffectmodel.set_variable(var_type='_x', var_name='theta', shape=(1,1))
    psi = geffectmodel.set_variable(var_type='_x', var_name='psi', shape=(1,1))
    dx = geffectmodel.set_variable(var_type='_x', var_name='dx', shape=(1,1))
    dy = geffectmodel.set_variable(var_type='_x', var_name='dy', shape=(1,1))
    dz = geffectmodel.set_variable(var_type='_x', var_name='dz', shape=(1,1))
    dphi = geffectmodel.set_variable(var_type='_x', var_name='dphi', shape=(1,1))
    dtheta = geffectmodel.set_variable(var_type='_x', var_name='dtheta', shape=(1,1))
    dpsi = geffectmodel.set_variable(var_type='_x', var_name='dpsi', shape=(1,1))
    Th = geffectmodel.set_variable(var_type='_u', var_name='Th')
    tx = geffectmodel.set_variable(var_type='_u', var_name='tx')
    ty = geffectmodel.set_variable(var_type='_u', var_name='ty')
    tz = geffectmodel.set_variable(var_type='_u', var_name='tz')

    geffectmodel.set_rhs('x', dx)
    geffectmodel.set_rhs('y', dy)
    geffectmodel.set_rhs('z', dz)
    geffectmodel.set_rhs('phi', dphi)
    geffectmodel.set_rhs('theta', dtheta)
    geffectmodel.set_rhs('psi', dpsi)

    dx_next = ((cos(psi)*sin(theta)*cos(phi) + sin(psi)*sin(phi))/mass)*(Th)
    dy_next = ((sin(psi)*sin(theta)*cos(phi) - cos(psi)*sin(phi))/mass)*(Th)
    dz_next = -gravity + ((cos(theta)*cos(phi))/mass)*(Th)
    dphi_next = dtheta*dpsi*c1 + (tx)/Ixx
    dtheta_next = dphi*dpsi*c2 + (ty)/Iyy
    dpsi_next = dphi*dtheta*c3 + (tz)/Izz

    geffectmodel.set_rhs('dx', dx_next)
    geffectmodel.set_rhs('dy', dy_next)
    geffectmodel.set_rhs('dz', dz_next)
    geffectmodel.set_rhs('dphi', dphi_next)
    geffectmodel.set_rhs('dtheta', dtheta_next)
    geffectmodel.set_rhs('dpsi', dpsi_next)

    # Build the model
    geffectmodel.setup()

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
    Th = model.set_variable(var_type='_u', var_name='Th')
    tx = model.set_variable(var_type='_u', var_name='tx')
    ty = model.set_variable(var_type='_u', var_name='ty')
    tz = model.set_variable(var_type='_u', var_name='tz')

    model.set_rhs('x', dx)
    model.set_rhs('y', dy)
    model.set_rhs('z', dz)
    model.set_rhs('phi', dphi)
    model.set_rhs('theta', dtheta)
    model.set_rhs('psi', dpsi)

    dx_next = ((cos(psi)*sin(theta)*cos(phi) + sin(psi)*sin(phi))/mass)*(Th*0.1*max_thrust)
    dy_next = ((sin(psi)*sin(theta)*cos(phi) - cos(psi)*sin(phi))/mass)*(Th*0.1*max_thrust)
    dz_next = -gravity + ((cos(theta)*cos(phi))/mass)*(Th*0.1*max_thrust)
    dphi_next = dtheta*dpsi*c1 + (tx*0.2*max_tx)/Ixx
    dtheta_next = dphi*dpsi*c2 + (ty*0.2*max_ty)/Iyy
    dpsi_next = dphi*dtheta*c3 + (tz*0.2*max_tz)/Izz

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
        #'n_horizon': 25,
        'n_horizon': 50,
        't_step': 0.02,
        'n_robust': 0,
        'store_full_solution': True,
        'nlpsol_opts': {'ipopt.linear_solver': 'MA27'}
    }

    mpc.set_param(**setup_mpc)

    mpc.settings.supress_ipopt_output()

    errorpen = 1.0

    pospen = 10.0
    xpen = 1.0
    ypen = 1.0
    zpen = 1.0

    angpen = 1e-3
    phipen = 1.0
    thetapen = 1.0
    psipen = 1.0

    velpen = 1.0
    dxpen = 1.0
    dypen = 1.0
    dzpen = 1.0

    angvelpen = 1e-3
    dphipen = 1.0
    dthetapen = 1.0
    dpsipen = 1.0

    cost = pospen*(xpen*(x - xfinal[0])**2 + ypen*(y - xfinal[1])**2 + zpen*(z - xfinal[2])**2) + angpen*(phipen*(phi - xfinal[3])**2 + thetapen*(theta - xfinal[4])**2 + psipen*(psi - xfinal[5])**2) + velpen*(dxpen*(dx - xfinal[6])**2 + dypen*(dy - xfinal[7])**2 + dzpen*(dz - xfinal[8])**2) + angvelpen*(dphipen*(dphi - xfinal[9])**2 + dthetapen*(dtheta - xfinal[10])**2 + dpsipen*(dpsi - xfinal[11])**2)
    #cost = 10*(x - xfinal[0])**2 + 10*(y - xfinal[1])**2 + 10*(z - xfinal[2])**2

    mpc.set_objective(mterm=0*cost, lterm=cost)

    mpc.set_rterm(Th=1.0,tx=1.0,ty=1.0,tz=1.0) # input penalty

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
    mpc.bounds['lower','_u','Th'] = 0.0
    mpc.bounds['lower','_u','tx'] = -5.0
    mpc.bounds['lower','_u','ty'] = -5.0
    mpc.bounds['lower','_u','tz'] = -5.0

    # upper bounds of the input
    mpc.bounds['upper','_u','Th'] = 10.0
    mpc.bounds['upper','_u','tx'] = 5.0
    mpc.bounds['upper','_u','ty'] = 5.0
    mpc.bounds['upper','_u','tz'] = 5.0

    mpc.setup()

    simulator = do_mpc.simulator.Simulator(geffectmodel)

    simulator.set_param(t_step = 0.02)

    tvp_template_sim = simulator.get_tvp_template()

    def tvp_fun_sim(t_now):
        return tvp_template_sim

    simulator.set_tvp_fun(tvp_fun_sim)

    simulator.setup()

    # Initial state
    x0 = xstart
    mpc.x0 = x0
    simulator.x0 = x0
    mpc.u0 = u0_norm
    simulator.u0 = u0

    # Use initial state to set the initial guess.
    mpc.set_initial_guess()

    iter = 0
    for n in range(datalen):
        t2 = time()
        iter = iter + 1
        print(iter)
        cpoint1 = np.sqrt((x0[0] - xfinal[0])**2 + (x0[1] - xfinal[1])**2 + (x0[2] - xfinal[2])**2)
        print(cpoint1)
        u0 = mpc.make_step(x0)
        if geffect == 1:
            u0[0,0] = u0[0,0]*0.1*max_thrust
            u0[1,0] = u0[1,0]*0.2*max_tx
            u0[2,0] = u0[2,0]*0.2*max_ty
            u0[3,0] = u0[3,0]*0.2*max_tz
            fge = (0.104*(R/x0[2]) - 0.0952)*(np.sqrt(x0[6]**2 + x0[7]**2)/vh)**2 - 0.171*(R/x0[2]) + 1.02
            if fge > 1:
                fge = np.ones(1)
            u0[0,0] = u0[0,0]/fge
        else:
            u0[0,0] = u0[0,0]*0.1*max_thrust
            u0[1,0] = u0[1,0]*0.2*max_tx
            u0[2,0] = u0[2,0]*0.2*max_ty
            u0[3,0] = u0[3,0]*0.2*max_tz
        x0 = simulator.make_step(u0)
        cpoint2 = np.sqrt((x0[0] - xfinal[0])**2 + (x0[1] - xfinal[1])**2 + (x0[2] - xfinal[2])**2)
        print(cpoint2)
        print(cpoint2-cpoint1)
        itertime[iter-1] = time() - t2
        cpointdata[iter-1] = cpoint2
        cpointdiff[iter-1] = cpoint2-cpoint1

    results = mpc.data
    t_f = results['_time']
    x_f = results['_x']
    u_f = results['_u']

    t_f = np.squeeze(t_f,axis=1)

    u_f[:,0] = u_f[:,0]*(0.1*max_thrust)
    u_f[:,1] = u_f[:,1]*(0.2*max_tx)
    u_f[:,2] = u_f[:,2]*(0.2*max_ty)
    u_f[:,3] = u_f[:,3]*(0.2*max_tz)

    return [t_f,x_f,u_f,itertime,cpointdata,cpointdiff]

def quadmpc_koop(xstart,zstart,zfinal,datalen,u0,n,A_array,B_array,modeltype,basis,xscal,geffect = 0):
    t1 = time()

    def funcxnorm(x,scal):
        xnorm = np.zeros(12)
        for n1 in range(12):
            xnorm[n1] = x[n1]/scal[n1]
        return xnorm
    
    def funcxscal(xnorm,scal):
        x = np.zeros(12)
        for n1 in range(12):
            x[n1] = xnorm[n1]*scal[n1]
        return x

    itertime = np.zeros(datalen)
    cpointdata = np.zeros(datalen)
    cpointdiff = np.zeros(datalen)

    mass = 0.0336
    Ixx = 1.7e-5
    Iyy = 1.7e-5
    Izz = 2.9e-5
    gravity = 9.81
    l = 0.04
    ct = 3.1582e-10
    cq = 7.9379e-12

    R = (45e-3)/2
    vh = np.sqrt(((mass*gravity)/4)/(2*1.225*(np.pi)*(R**2)))

    Tarray = np.zeros((4,4))
    Tarray[0,:] = ct
    Tarray[1,1] = ct*l
    Tarray[1,3] = -ct*l
    Tarray[2,0] = -ct*l
    Tarray[2,2] = ct*l
    Tarray[3,0] = -cq
    Tarray[3,1] = cq
    Tarray[3,2] = -cq
    Tarray[3,3] = cq

    hover_w2 = (mass*gravity)/(4*ct)
    max_w2 = (0.057*gravity)/(4*ct)
    hover_thrust = (mass*gravity)
    max_thrust = (0.057*gravity)
    max_tx = ct*l*max_w2
    max_ty = ct*l*max_w2
    max_tz = 2*cq*max_w2
    uhover = np.array([[hover_thrust], [0.0], [0.0], [0.0]])
    uhover_norm = uhover/(0.1*max_thrust)

    u0_norm = np.zeros((4,1))
    u0_norm[0,0] = u0[0,0]/(0.1*max_thrust)
    u0_norm[1,0] = u0[1,0]/(0.2*max_tx)
    u0_norm[2,0] = u0[2,0]/(0.2*max_ty)
    u0_norm[3,0] = u0[3,0]/(0.2*max_tz)

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
    Th = model.set_variable(var_type='_u', var_name='Th')
    tx = model.set_variable(var_type='_u', var_name='tx')
    ty = model.set_variable(var_type='_u', var_name='ty')
    tz = model.set_variable(var_type='_u', var_name='tz')

    c1 = (Iyy - Izz)/Ixx
    c2 = (Izz - Ixx)/Iyy
    c3 = (Iyy - Ixx)/Izz

    model.set_rhs('x', dx)
    model.set_rhs('y', dy)
    model.set_rhs('z', dz)
    model.set_rhs('phi', dphi)
    model.set_rhs('theta', dtheta)
    model.set_rhs('psi', dpsi)

    dx_next = ((cos(psi)*sin(theta)*cos(phi) + sin(psi)*sin(phi))/mass)*(Th)
    dy_next = ((sin(psi)*sin(theta)*cos(phi) - cos(psi)*sin(phi))/mass)*(Th)
    dz_next = -gravity + ((cos(theta)*cos(phi))/mass)*(Th)
    dphi_next = dtheta*dpsi*c1 + (tx)/Ixx
    dtheta_next = dphi*dpsi*c2 + (ty)/Iyy
    dpsi_next = dphi*dtheta*c3 + (tz)/Izz

    model.set_rhs('dx', dx_next)
    model.set_rhs('dy', dy_next)
    model.set_rhs('dz', dz_next)
    model.set_rhs('dphi', dphi_next)
    model.set_rhs('dtheta', dtheta_next)
    model.set_rhs('dpsi', dpsi_next)

    # Build the model
    model.setup()

    model_type_koop = modeltype # either 'discrete' or 'continuous'
    model_koop = do_mpc.model.Model(model_type_koop)

    z = model_koop.set_variable(var_type='_x', var_name='z', shape=(n,1))
    Th = model_koop.set_variable(var_type='_u', var_name='Th')
    tx = model_koop.set_variable(var_type='_u', var_name='tx')
    ty = model_koop.set_variable(var_type='_u', var_name='ty')
    tz = model_koop.set_variable(var_type='_u', var_name='tz')

    G1 = B_array[0,:,:]
    G2 = B_array[1,:,:]
    G3 = B_array[2,:,:]
    G4 = B_array[3,:,:]

    x_next = (A_array + ((G1)*(Th) + (G2)*(tx) + (G3)*(ty) + (G4)*(tz)))@z

    model_koop.set_rhs('z', x_next)

    # Build the model
    model_koop.setup()

    mpc = do_mpc.controller.MPC(model_koop)

    if modeltype == 'discrete':
        setup_mpc = {
            #'n_horizon': 25,
            'n_horizon': 10,
            't_step': 0.02,
            #'n_robust': 0,
            'n_robust': 4,
            'state_discretization': 'discrete',
            'store_full_solution': True,
            'nlpsol_opts': {'ipopt.linear_solver': 'MA27'}
        }
    else:
        setup_mpc = {
        #'n_horizon': 25,
        'n_horizon': 10,
        't_step': 0.02,
        #'n_robust': 0,
        'n_robust': 4,
        'state_discretization': 'collocation',
        'collocation_type': 'radau',
        'collocation_deg': 2,
        'collocation_ni': 1,
        'store_full_solution': True,
        'nlpsol_opts': {'ipopt.linear_solver': 'MA27'}
        }

    mpc.set_param(**setup_mpc)

    mpc.settings.supress_ipopt_output()

    errorpen = 1.0

    pospen = 2.0
    xpen = 1.0
    ypen = 1.0
    zpen = 1.0

    angpen = 1e-1
    phipen = 1.0
    thetapen = 1.0
    psipen = 1.0

    velpen = 2.0
    dxpen = 1.0
    dypen = 1.0
    dzpen = 1.0

    angvelpen = 1e-1
    dphipen = 1.0
    dthetapen = 1.0
    dpsipen = 1.0

    #cost = errorpen*((z[0] - zfinal[0])**2 + pospen*(xpen*(z[1] - zfinal[1])**2 + ypen*(z[2] - zfinal[2])**2 + zpen*(z[3] - zfinal[3])**2) + angpen*(phipen*(z[4] - zfinal[4])**2 + thetapen*(z[5] - zfinal[5])**2 + psipen*(z[6] - zfinal[6])**2) + velpen*(dxpen*(z[7] - zfinal[7])**2 + dypen*(z[8] - zfinal[8])**2 + dzpen*(z[9] - zfinal[9])**2) + angvelpen*(dphipen*(z[10] - zfinal[10])**2 + dthetapen*(z[11] - zfinal[11])**2 + dpsipen*(z[12] - zfinal[12])**2) + (z[13] - zfinal[13])**2 + (z[14] - zfinal[14])**2 + (z[15] - zfinal[15])**2 + (z[16] - zfinal[16])**2 + (z[17] - zfinal[17])**2 + (z[18] - zfinal[18])**2 + (z[19] - zfinal[19])**2 + (z[20] - zfinal[20])**2 + (z[21] - zfinal[21])**2 + (z[22] - zfinal[22])**2)
    cost = errorpen*((z[0] - zfinal[0])**2 + pospen*(xpen*(z[1] - zfinal[1])**2 + ypen*(z[2] - zfinal[2])**2 + zpen*(z[3] - zfinal[3])**2) + angpen*(phipen*(z[4] - zfinal[4])**2 + thetapen*(z[5] - zfinal[5])**2 + psipen*(z[6] - zfinal[6])**2) + velpen*(dxpen*(z[7] - zfinal[7])**2 + dypen*(z[8] - zfinal[8])**2 + dzpen*(z[9] - zfinal[9])**2) + angvelpen*(dphipen*(z[10] - zfinal[10])**2 + dthetapen*(z[11] - zfinal[11])**2 + dpsipen*(z[12] - zfinal[12])**2))
    #cost = errorpen*(pospen*(xpen*(z[1] - zfinal[1])**2 + ypen*(z[2] - zfinal[2])**2 + zpen*(z[3] - zfinal[3])**2))

    mpc.set_objective(mterm=0*cost, lterm=cost)

    inputpen = 1.0

    mpc.set_rterm(Th=inputpen,tx=inputpen,ty=inputpen,tz=inputpen) # input penalty

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
    mpc.bounds['lower','_u','Th'] = 0
    mpc.bounds['lower','_u','tx'] = -5.0
    mpc.bounds['lower','_u','ty'] = -5.0
    mpc.bounds['lower','_u','tz'] = -5.0

    # upper bounds of the input
    mpc.bounds['upper','_u','Th'] = 10.0
    mpc.bounds['upper','_u','tx'] = 5.0
    mpc.bounds['upper','_u','ty'] = 5.0
    mpc.bounds['upper','_u','tz'] = 5.0

    mpc.setup()

    simulator = do_mpc.simulator.Simulator(model)

    simulator.set_param(t_step = 0.02)

    tvp_template_sim = simulator.get_tvp_template()

    def tvp_fun_sim(t_now):
        return tvp_template_sim

    simulator.set_tvp_fun(tvp_fun_sim)

    simulator.setup()

    simulator.setup()

    # Initial state
    x0 = xstart
    z0 = zstart
    mpc.x0 = z0
    simulator.x0 = x0
    mpc.u0 = u0_norm
    simulator.u0 = u0

    # Use initial state to set the initial guess.
    mpc.set_initial_guess()

    iter = 0
    cpoint1 = np.sqrt((z0[1]*xscal[0] - zfinal[1]*xscal[0])**2 + (z0[2]*xscal[1] - zfinal[2]*xscal[1])**2 + (z0[3]*xscal[2] - zfinal[3]*xscal[2])**2)
    for n in range(datalen):
        t2 = time()
        iter = iter + 1
        print(iter)
        cpoint1 = np.sqrt((z0[1]*xscal[0] - zfinal[1]*xscal[0])**2 + (z0[2]*xscal[1] - zfinal[2]*xscal[1])**2 + (z0[3]*xscal[2] - zfinal[3]*xscal[2])**2)
        print(cpoint1)
        u0 = mpc.make_step(z0)
        u0[0,0] = u0[0,0]*0.1*max_thrust
        u0[1,0] = u0[1,0]*0.2*max_tx
        u0[2,0] = u0[2,0]*0.2*max_ty
        u0[3,0] = u0[3,0]*0.2*max_tz
        if geffect == 1:
            fge = (0.104*(R/z0[3]) - 0.0952)*(np.sqrt(z0[7]**2 + z0[8]**2)/vh)**2 - 0.171*(R/z0[3]) + 1.02
            if fge > 1:
                fge = np.ones(1)
            u0[0,0] = u0[0,0]/fge
        x0 = simulator.make_step(u0)
        x0 = np.squeeze(x0)
        z0 = np.transpose(basis(funcxnorm(x0,xscal)))
        cpoint2 = np.sqrt((z0[1]*xscal[0] - zfinal[1]*xscal[0])**2 + (z0[2]*xscal[1] - zfinal[2]*xscal[1])**2 + (z0[3]*xscal[2] - zfinal[3]*xscal[2])**2)
        print(cpoint2)
        print(cpoint2-cpoint1)
        itertime[iter-1] = time() - t2
        cpointdata[iter-1] = cpoint2
        cpointdiff[iter-1] = cpoint2-cpoint1
        if cpointdiff[iter-1] > 0.2:
            break

    results1 = mpc.data
    t_f = results1['_time']
    z_f = results1['_x']
    u_f = results1['_u']

    t_f = np.squeeze(t_f,axis=1)

    results2 = simulator.data
    x_f = results2['_x']

    u_f[:,0] = u_f[:,0]*(0.1*max_thrust)
    u_f[:,1] = u_f[:,1]*(0.2*max_tx)
    u_f[:,2] = u_f[:,2]*(0.2*max_ty)
    u_f[:,3] = u_f[:,3]*(0.2*max_tz)

    return [t_f,x_f,u_f,itertime,cpointdata,cpointdiff]

def dataprocess(x_train,w_train,x_val,w_val):
    u_train = np.empty((n_traj_train, datagen_iter, 4))
    u_val = np.empty((n_traj_val, datagen_iter, 4))

    for n1 in range(8):
        Th = ct*(((w_train[n1,:,0])) + ((w_train[n1,:,1])) + ((w_train[n1,:,2])) + ((w_train[n1,:,3])))
        tx = ct*l*(((w_train[n1,:,1])) - ((w_train[n1,:,3])))
        ty = ct*l*(- ((w_train[n1,:,0])) + ((w_train[n1,:,2])))
        tz = cq*( - ((w_train[n1,:,0])) + ((w_train[n1,:,1])) - ((w_train[n1,:,2])) + ((w_train[n1,:,3])))
        u_train[n1,:,0] = Th
        u_train[n1,:,1] = tx
        u_train[n1,:,2] = ty
        u_train[n1,:,3] = tz

    for n2 in range(2):
        Th = ct*(((w_val[n2,:,0])) + ((w_val[n2,:,1])) + ((w_val[n2,:,2])) + ((w_val[n2,:,3])))
        tx = ct*l*(((w_val[n2,:,1])) - ((w_val[n2,:,3])))
        ty = ct*l*(- ((w_val[n2,:,0])) + ((w_val[n2,:,2])))
        tz = cq*( - ((w_val[n2,:,0])) + ((w_val[n2,:,1])) - ((w_val[n2,:,2])) + ((w_val[n2,:,3])))
        u_val[n2,:,0] = Th
        u_val[n2,:,1] = tx
        u_val[n2,:,2] = ty
        u_val[n2,:,3] = tz

    u_train[:,:,0] = ((u_train[:,:,0])/(0.1*max_thrust))
    u_val[:,:,0] = ((u_val[:,:,0])/(0.1*max_thrust))

    u_train[:,:,1] = (u_train[:,:,1])/(0.2*max_tx)
    u_val[:,:,1] = (u_val[:,:,1])/(0.2*max_tx)

    u_train[:,:,2] = (u_train[:,:,2])/(0.2*max_ty)
    u_val[:,:,2] = (u_val[:,:,2])/(0.2*max_ty)

    u_train[:,:,3] = (u_train[:,:,3])/(0.2*max_tz)
    u_val[:,:,3] = (u_val[:,:,3])/(0.2*max_tz)

    x_train[:,:,3:6] = pp.so3(x_train[:,:,3:6]).tensor()
    x_val[:,:,3:6] = pp.so3(x_val[:,:,3:6]).tensor()

    return x_train,u_train,x_val,u_val

n = 12
m = 4
dt = 0.02

blo = 1
discrete = 1

koopnet = 0
koopnet_train = 0

groundeffect = 0

mass = 0.0336
Ixx = 1.7e-5
Iyy = 1.7e-5
Izz = 2.9e-5
gravity = 9.81
l = 0.04
ct = 3.1582e-10
cq = 7.9379e-12

R = (45e-3)/2
vh = np.sqrt(((mass*gravity)/4)/(2*1.225*(np.pi)*(R**2)))

Tarray = np.zeros((4,4))
Tarray[0,:] = ct
Tarray[1,1] = ct*l
Tarray[1,3] = -ct*l
Tarray[2,0] = -ct*l
Tarray[2,2] = ct*l
Tarray[3,0] = -cq
Tarray[3,1] = cq
Tarray[3,2] = -cq
Tarray[3,3] = cq

hover_w2 = (mass*gravity)/(4*ct)
max_w2 = (0.057*gravity)/(4*ct)
hover_thrust = (mass*gravity)
max_thrust = (0.057*gravity)
max_tx = ct*l*max_w2
max_ty = ct*l*max_w2
max_tz = 2*cq*max_w2
uhover = np.array([[hover_thrust], [0.0], [0.0], [0.0]])
uhover_norm = uhover/(0.1*max_thrust)

print(hover_thrust)
print(max_thrust)
print(uhover)
print(uhover_norm)

n_traj_train = 8
n_traj_val = int(0.25*n_traj_train)

datagen_iter = 12000

if groundeffect == 1:
    x_train = np.load('x_train_7.npy')
    w_train = np.load('u_train_7.npy')
    x_val = np.load('x_val_7.npy')
    w_val = np.load('u_val_7.npy')
else:
    x_train = np.load('x_train_4.npy')
    w_train = np.load('u_train_4.npy')
    x_val = np.load('x_val_4.npy')
    w_val = np.load('u_val_4.npy')

[x_train,u_train,x_val,u_val] = dataprocess(x_train,w_train,x_val,w_val)

data_iter_new = 12000

train_traj_len = 200

x_train_new = np.zeros((10,data_iter_new,n))
x_train_new[:8,:,:] = x_train[:,:data_iter_new,:]
x_train_new[8:10,:,:] = x_val[:,:data_iter_new,:]

u_train_new = np.zeros((10,data_iter_new,m))
u_train_new[:8,:,:] = u_train[:,:data_iter_new,:]
u_train_new[8:10,:,:] = u_val[:,:data_iter_new,:]

x_train_new = np.reshape(x_train_new,(-1,train_traj_len,n))
u_train_new = np.reshape(u_train_new,(-1,train_traj_len,m))

t_val = dt*np.arange(len(x_train_new[0,:,0]))

print(np.shape(x_train_new))
print(np.shape(u_train_new))

x_scal = np.ones(n)

net_params = {}
net_params['state_dim'] = n
net_params['ctrl_dim'] = m
net_params['encoder_hidden_width'] = 100
net_params['encoder_hidden_depth'] = 2
net_params['encoder_output_dim'] = 10
net_params['optimizer'] = 'adam'
net_params['activation_type'] = 'tanh'
net_params['lr'] = 1e-3
net_params['epochs'] = 40
net_params['batch_size'] = 200
net_params['lin_loss_penalty'] = 2e-1
net_params['l2_reg'] = 5e-4
net_params['l1_reg'] = 0.0
net_params['n_fixed_states'] = n
net_params['first_obs_const'] = True
net_params['override_kinematics'] = True
net_params['override_C'] = True
net_params['dt'] = dt

print(net_params)

net_params_blo = net_params
net_params_blo['batch_size'] = 200

if blo == 1:
    [A_blo,B_blo,params_blo] = BLOfunc_save(t_val,x_train_new,u_train_new,net_params_blo,disc=discrete)

np.save('A_datagen4_blo_disc_1',A_blo)
np.save('B_datagen4_blo_disc_1',B_blo)

with open('params_datagen4_blo_disc_1.pkl','wb') as fp:
    pickle.dump(params_blo,fp)



