import numpy as np
import matplotlib.pyplot as plt
import torch
import do_mpc
import pypose as pp
import pickle
import jax
from numpy import sin, cos, tan
from time import time
from Cont_func_1 import CKBF_STK

# Quadcopter NMPC Function
def quadmpc(xarray,u0,contw,geffect = 0,ideal = 0):

    datalen = len(xarray[:,0])

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

    xref1 = model.set_variable('_tvp', 'xref1', shape=(12,1))

    model.set_rhs('x', dx)
    model.set_rhs('y', dy)
    model.set_rhs('z', dz)
    model.set_rhs('phi', dphi)
    model.set_rhs('theta', dtheta)
    model.set_rhs('psi', dpsi)

    if ideal == 1:
        dx_next = ((cos(psi)*sin(theta)*cos(phi) + sin(psi)*sin(phi))/mass)*(Th*0.1*max_thrust)/((0.104*(R/z) - 0.0952)*(np.sqrt(dx**2 + dy**2)/vh)**2 - 0.171*(R/z) + 1.02)
        dy_next = ((sin(psi)*sin(theta)*cos(phi) - cos(psi)*sin(phi))/mass)*(Th*0.1*max_thrust)/((0.104*(R/z) - 0.0952)*(np.sqrt(dx**2 + dy**2)/vh)**2 - 0.171*(R/z) + 1.02)
        dz_next = -gravity + ((cos(theta)*cos(phi))/mass)*(Th*0.1*max_thrust)/((0.104*(R/z) - 0.0952)*(np.sqrt(dx**2 + dy**2)/vh)**2 - 0.171*(R/z) + 1.02)
    else:
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
        'n_horizon': 25,
        't_step': 0.02,
        'n_robust': 0,
        'store_full_solution': True,
        'nlpsol_opts': {'ipopt.linear_solver': 'MA27'}
    }

    mpc.set_param(**setup_mpc)

    mpc.settings.supress_ipopt_output()

    errorpen = 1.0

    #pospen = 40.0
    #pospen = 10.0
    pospen = 1.0
    xpen = contw[0]
    ypen = contw[1]
    zpen = contw[2]

    #angpen = 0.1
    #angpen = 0.2
    angpen = 1.0
    phipen = contw[3]
    thetapen = contw[4]
    psipen = contw[5]

    #velpen = 2.0
    velpen = 1.0
    dxpen = contw[6]
    dypen = contw[7]
    dzpen = contw[8]

    #angvelpen = 0.1
    #angvelpen = 0.2
    angvelpen = 1.0
    dphipen = contw[9]
    dthetapen = contw[10]
    dpsipen = contw[11]

    cost = pospen*(xpen*(x - model.tvp['xref1',0])**2 + ypen*(y - model.tvp['xref1',1])**2 + zpen*(z - model.tvp['xref1',2])**2) + angpen*(phipen*(phi - model.tvp['xref1',3])**2 + thetapen*(theta - model.tvp['xref1',4])**2 + psipen*(psi - model.tvp['xref1',5])**2) + velpen*(dxpen*(dx - model.tvp['xref1',6])**2 + dypen*(dy - model.tvp['xref1',7])**2 + dzpen*(dz - model.tvp['xref1',8])**2) + angvelpen*(dphipen*(dphi - model.tvp['xref1',9])**2 + dthetapen*(dtheta - model.tvp['xref1',10])**2 + dpsipen*(dpsi - model.tvp['xref1',11])**2)
    #cost = 10*(x - model.tvp['xref1',0])**2 + 10*(y - model.tvp['xref1',1])**2 + 10*(z - model.tvp['xref1',2])**2

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

    tvp_template = mpc.get_tvp_template()
    
    def tvp_fun(t_now):
        n2 = int(t_now/0.02) + 1
        #n2 = int(t_now/0.02)
        for n3 in range(12):
            tvp_template['_tvp',:, 'xref1', n3] = xarray[n2,n3]
        return tvp_template

    mpc.set_tvp_fun(tvp_fun)

    mpc.setup()

    simulator = do_mpc.simulator.Simulator(geffectmodel)

    simulator.set_param(t_step = 0.02)

    tvp_template_sim = simulator.get_tvp_template()

    def tvp_fun_sim(t_now):
        return tvp_template_sim

    simulator.set_tvp_fun(tvp_fun_sim)

    simulator.setup()

    # Initial state
    x0 = xarray[0,:]
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
        cpoint1 = np.sqrt((x0[0] - xarray[iter-1,0])**2 + (x0[1] - xarray[iter-1,1])**2 + (x0[2] - xarray[iter-1,2])**2)
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
        cpoint2 = np.sqrt((x0[0] - xarray[iter-1,0])**2 + (x0[1] - xarray[iter-1,1])**2 + (x0[2] - xarray[iter-1,2])**2)
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

# Quadcopter Koopman MPC Function
def quadmpc_koop(xarray,zarray,u0,n,A_array,B_array,modeltype,basis,xscal,contw,geffect = 0):
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

    datalen = len(xarray[:,0])

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

    zref1 = model_koop.set_variable('_tvp', 'zref1', shape=(n,1))

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
            'n_horizon': 25,
            #'n_horizon': 10,
            't_step': 0.02,
            'n_robust': 0,
            #'n_robust': 2,
            'state_discretization': 'discrete',
            'store_full_solution': True,
            'nlpsol_opts': {'ipopt.linear_solver': 'MA27'}
        }
    else:
        setup_mpc = {
        'n_horizon': 25,
        #'n_horizon': 10,
        't_step': 0.02,
        'n_robust': 0,
        #'n_robust': 1,
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

    #pospen = 40.0
    #pospen = 10.0
    pospen = 1.0
    xpen = contw[0]
    ypen = contw[1]
    zpen = contw[2]

    #angpen = 0.1
    #angpen = 0.2
    angpen = 1.0
    phipen = contw[3]
    thetapen = contw[4]
    psipen = contw[5]

    #velpen = 2.0
    velpen = 1.0
    dxpen = contw[6]
    dypen = contw[7]
    dzpen = contw[8]

    #angvelpen = 0.1
    #angvelpen = 0.2
    angvelpen = 1.0
    dphipen = contw[9]
    dthetapen = contw[10]
    dpsipen = contw[11]

    #cost = errorpen*((z[0] - model_koop.tvp['zref1',0])**2 + pospen*(xpen*(z[1] - model_koop.tvp['zref1',1])**2 + ypen*(z[2] - model_koop.tvp['zref1',2])**2 + zpen*(z[3] - model_koop.tvp['zref1',3])**2) + angpen*(phipen*(z[4] - model_koop.tvp['zref1',4])**2 + thetapen*(z[5] - model_koop.tvp['zref1',5])**2 + psipen*(z[6] - model_koop.tvp['zref1',6])**2) + velpen*(dxpen*(z[7] - model_koop.tvp['zref1',7])**2 + dypen*(z[8] - model_koop.tvp['zref1',8])**2 + dzpen*(z[9] - model_koop.tvp['zref1',9])**2) + angvelpen*(dphipen*(z[10] - model_koop.tvp['zref1',10])**2 + dthetapen*(z[11] - model_koop.tvp['zref1',11])**2 + dpsipen*(z[12] - model_koop.tvp['zref1',12])**2) + (z[13] - model_koop.tvp['zref1',13])**2 + (z[14] - model_koop.tvp['zref1',14])**2 + (z[15] - model_koop.tvp['zref1',15])**2 + (z[16] - model_koop.tvp['zref1',16])**2 + (z[17] - model_koop.tvp['zref1',17])**2 + (z[18] - model_koop.tvp['zref1',18])**2 + (z[19] - model_koop.tvp['zref1',19])**2 + (z[20] - model_koop.tvp['zref1',20])**2 + (z[21] - model_koop.tvp['zref1',21])**2 + (z[22] - model_koop.tvp['zref1',22])**2)
    cost = pospen*(xpen*(z[1] - model_koop.tvp['zref1',1])**2 + ypen*(z[2] - model_koop.tvp['zref1',2])**2 + zpen*(z[3] - model_koop.tvp['zref1',3])**2) + angpen*(phipen*(z[4] - model_koop.tvp['zref1',4])**2 + thetapen*(z[5] - model_koop.tvp['zref1',5])**2 + psipen*(z[6] - model_koop.tvp['zref1',6])**2) + velpen*(dxpen*(z[7] - model_koop.tvp['zref1',7])**2 + dypen*(z[8] - model_koop.tvp['zref1',8])**2 + dzpen*(z[9] - model_koop.tvp['zref1',9])**2) + angvelpen*(dphipen*(z[10] - model_koop.tvp['zref1',10])**2 + dthetapen*(z[11] - model_koop.tvp['zref1',11])**2 + dpsipen*(z[12] - model_koop.tvp['zref1',12])**2)
    #cost = pospen*(xpen*(z[1] - model_koop.tvp['zref1',1])**2 + ypen*(z[2] - model_koop.tvp['zref1',2])**2 + zpen*(z[3] - model_koop.tvp['zref1',3])**2) + velpen*(dxpen*(z[7] - model_koop.tvp['zref1',7])**2 + dypen*(z[8] - model_koop.tvp['zref1',8])**2 + dzpen*(z[9] - model_koop.tvp['zref1',9])**2)
    #cost = pospen*(xpen*(z[1] - model_koop.tvp['zref1',1])**2 + ypen*(z[2] - model_koop.tvp['zref1',2])**2 + zpen*(z[3] - model_koop.tvp['zref1',3])**2)

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

    tvp_template = mpc.get_tvp_template()

    def tvp_fun(t_now):
        n2 = int(t_now/0.02) + 1
        #n2 = int(t_now/0.02)
        for n3 in range(23):
            tvp_template['_tvp',:, 'zref1', n3] = zarray[n2,n3]
        return tvp_template

    mpc.set_tvp_fun(tvp_fun)

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
    x0 = xarray[0,:]
    z0 = zarray[0,:]
    mpc.x0 = z0
    simulator.x0 = x0
    mpc.u0 = u0_norm
    simulator.u0 = u0

    # Use initial state to set the initial guess.
    mpc.set_initial_guess()

    iter = 0
    cpoint1 = np.sqrt((z0[1]*xscal[0] - zarray[0,1]*xscal[0])**2 + (z0[2]*xscal[1] - zarray[0,2]*xscal[1])**2 + (z0[3]*xscal[2] - zarray[0,3]*xscal[2])**2)
    for n in range(datalen):
        t2 = time()
        iter = iter + 1
        print(iter)
        cpoint1 = np.sqrt((z0[1]*xscal[0] - zarray[iter-1,1]*xscal[0])**2 + (z0[2]*xscal[1] - zarray[iter-1,2]*xscal[1])**2 + (z0[3]*xscal[2] - zarray[iter-1,3]*xscal[2])**2)
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
        cpoint2 = np.sqrt((z0[1]*xscal[0] - zarray[iter-1,1]*xscal[0])**2 + (z0[2]*xscal[1] - zarray[iter-1,2]*xscal[1])**2 + (z0[3]*xscal[2] - zarray[iter-1,3]*xscal[2])**2)
        print(cpoint2)
        print(cpoint2-cpoint1)
        itertime[iter-1] = time() - t2
        cpointdata[iter-1] = cpoint2
        cpointdiff[iter-1] = cpoint2-cpoint1
        if cpoint2 > 2.0:
            break
        if x0[2] < 0.0:
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

# KBF Function
def defkbf(n,m,net_params):
    return CKBF_STK([n, m, net_params['encoder_output_dim']+1], net_params['encoder_hidden_width']*np.ones(net_params['encoder_hidden_depth']), True, jax.nn.swish)

# Set state space and control input size and timestep
n = 12
m = 4
dt = 0.02

# Set if BLO results are desired and if it should be discretized
blo = 1
discrete = 0

# Set if KoopNet results are desired
koopnet = 1

# Set if ground effect is desired
groundeffect = 1

# Set if Ideal NMPC results are desired
idealcont = 1

# Set system parameters
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

# Define and display control input parameters
hover_w2 = (mass*gravity)/(4*ct)
max_w2 = (0.057*gravity)/(4*ct)
hover_thrust = (mass*gravity)
max_thrust = (0.057*gravity)
max_tx = ct*l*max_w2
max_ty = ct*l*max_w2
max_tz = 2*cq*max_w2
uhover = np.array([[hover_thrust], [0.0], [0.0], [0.0]])
uhover_norm = uhover/(0.1*max_thrust)

print('Hover Thrust, N = ')
print(hover_thrust)
print('Max Thrust, N = ')
print(max_thrust)
print('Hover Control Inputs = ')
print(uhover)
print('Normalized Hover Control Inputs = ')
print(uhover_norm)

# Define scaling for state space x
x_scal = np.ones(n)

# Define BLO Neural Network parameters
net_params_blo = {}
net_params_blo['state_dim'] = n
net_params_blo['ctrl_dim'] = m
net_params_blo['encoder_hidden_width'] = 100
net_params_blo['encoder_hidden_depth'] = 2
net_params_blo['encoder_output_dim'] = 10
net_params_blo['optimizer'] = 'adam'
net_params_blo['activation_type'] = 'tanh'
net_params_blo['lr'] = 1e-3
net_params_blo['epochs'] = 200
net_params_blo['batch_size'] = 200
net_params_blo['lin_loss_penalty'] = 2e-1
net_params_blo['l2_reg'] = 5e-4
net_params_blo['l1_reg'] = 0.0
net_params_blo['n_fixed_states'] = n
net_params_blo['first_obs_const'] = True
net_params_blo['override_kinematics'] = True
net_params_blo['override_C'] = True
net_params_blo['dt'] = dt

# Display BLO Neural Network parameters
print('BLO Neural Network Parameters = ')
print(net_params_blo)

# Load BLO data
if blo == 1:
    if discrete == 1:
        if groundeffect == 1:
            A_blo = np.load('A_datagen7_blo_disc_1.npy')
            B_blo = np.load('B_datagen7_blo_disc_1.npy')
            with open('params_datagen7_blo_disc_1.pkl','rb') as fp:
                params_blo = pickle.load(fp)
        else:
            A_blo = np.load('A_datagen4_blo_disc_1.npy')
            B_blo = np.load('B_datagen4_blo_disc_1.npy')
            with open('params_datagen4_blo_disc_1.pkl','rb') as fp:
                params_blo = pickle.load(fp)
    else:
        if groundeffect == 1:
            A_blo = np.load('A_datagen10_blo_1.npy')
            B_blo = np.load('B_datagen10_blo_1.npy')
            with open('params_datagen10_blo_1.pkl','rb') as fp:
                params_blo = pickle.load(fp)
        else:
            A_blo = np.load('A_datagen4_blo_1.npy')
            B_blo = np.load('B_datagen4_blo_1.npy')
            with open('params_datagen4_blo_1.pkl','rb') as fp:
                params_blo = pickle.load(fp)
    kbf = defkbf(n,m,net_params_blo)
    def basis_blo(x):
        return np.array(kbf.encoder(x,params_blo))

# Load KoopNet data
if koopnet == 1:
    if groundeffect == 1:
        A_koopnet = np.load('A_datagen10_1.npy')
        B_koopnet = np.load('B_datagen10_1.npy')
        net_koopnet = torch.load('net_datagen10_1.pt')
        def basis_koopnet(x):
            return net_koopnet.encode(np.atleast_2d(x))
    else:
        A_koopnet = np.load('A_datagen4_2.npy')
        B_koopnet = np.load('B_datagen4_2.npy')
        net_koopnet = torch.load('net_datagen4_2.pt')
        def basis_koopnet(x):
            return net_koopnet.encode(np.atleast_2d(x))

# Load and display BLO and KoopNet loss history    
if blo == 1:    
    with open('hist_datagen10_blo_1.pkl','rb') as fp:
        hist_blo = pickle.load(fp)
    loss_blo = np.array(hist_blo['L'])

if koopnet == 1:
    loss_koopnet = np.load('loss_datagen10_1.npy')

epochs = np.arange(0,net_params_blo['epochs'])

if blo == 1:  
    print('BLO Final Loss = ')
    print(loss_blo[-1,-1])
if koopnet == 1:
    print('KoopNet Final Training Loss = ')
    print(loss_koopnet[-1,0])
    print('KoopNet Final Validation Loss = ')
    print(loss_koopnet[-1,3])

# Plot BLO and KoopNet loss history
plt.figure()
if blo == 1:  
    plt.plot(epochs,loss_blo[:,-1],'--',label='BLO Loss')
if koopnet == 1:
    plt.plot(epochs,loss_koopnet[:,0],'--',label='KoopNet Training Loss')
    plt.plot(epochs,loss_koopnet[:,3],'--',label='KoopNet Validation Loss')
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.yscale('log')
plt.show()
            
# Define control weights
contw = np.zeros(n)
contw[0:3] = 20.0
contw[3:6] = 0.02
contw[6:9] = 1.0
contw[9:12] = 0.02

# Define test trajectory
traj_dt = dt
traj_tf = 10.0 # Length of trajectory

traj_tlen = 10.0 # Length to complete trajectory

tspan = np.arange(0,traj_tf + traj_dt,traj_dt) # Define test timespan

testtype = 'smallhorfig8' # Specify trajectory used, set as 0 for straight line flight

# Flight trajectory definitions
if testtype == 'horfig8':
    xspan = np.sin((2*(np.pi)/(traj_tlen/2))*tspan)
    xdotspan = (2*(np.pi)/(traj_tlen/2))*np.cos((2*(np.pi)/(traj_tlen/2))*tspan)
    yspan = np.cos((2*(np.pi)/traj_tlen)*tspan)
    ydotspan = -(2*(np.pi)/traj_tlen)*np.sin((2*(np.pi)/traj_tlen)*tspan)
    zspan = 0.05*np.ones(int(len(tspan)))
    zdotspan = np.zeros(int(len(tspan)))
elif testtype == 'smallhorfig8':
    xspan = 0.4*np.sin((2*(np.pi)/(traj_tlen/2))*tspan)
    xdotspan = 0.4*(2*(np.pi)/(traj_tlen/2))*np.cos((2*(np.pi)/(traj_tlen/2))*tspan)
    yspan = 0.4*np.cos((2*(np.pi)/traj_tlen)*tspan)
    ydotspan = -0.4*(2*(np.pi)/traj_tlen)*np.sin((2*(np.pi)/traj_tlen)*tspan)
    zspan = 0.05*np.ones(int(len(tspan)))
    zdotspan = np.zeros(int(len(tspan)))
elif testtype == 'vertfig8':
    xspan = 0.0*np.ones(int(len(tspan)))
    xdotspan = np.zeros(int(len(tspan)))
    yspan = np.cos((2*(np.pi)/traj_tlen)*tspan)
    ydotspan = -(2*(np.pi)/traj_tlen)*np.sin((2*(np.pi)/traj_tlen)*tspan)
    zspan = np.sin((2*(np.pi)/(traj_tlen/2))*tspan) + 1.1
    zdotspan = (2*(np.pi)/(traj_tlen/2))*np.cos((2*(np.pi)/(traj_tlen/2))*tspan)
elif testtype == 'horcirc':
    xspan = np.cos((2*(np.pi)/(traj_tlen))*tspan)
    xdotspan = -(2*(np.pi)/(traj_tlen))*np.sin((2*(np.pi)/(traj_tlen))*tspan)
    yspan = np.sin((2*(np.pi)/(traj_tlen))*tspan)
    ydotspan = (2*(np.pi)/(traj_tlen))*np.cos((2*(np.pi)/(traj_tlen))*tspan)
    zspan = 0.1*np.ones(int(len(tspan)))
    zdotspan = np.zeros(int(len(tspan)))
elif testtype == 'risecirc':
    xspan = np.cos((2*(np.pi)/(traj_tlen/2))*tspan)
    xdotspan = -(2*(np.pi)/(traj_tlen/2))*np.sin((2*(np.pi)/(traj_tlen/2))*tspan)
    yspan = np.sin((2*(np.pi)/(traj_tlen/2))*tspan)
    ydotspan = (2*(np.pi)/(traj_tlen/2))*np.cos((2*(np.pi)/(traj_tlen/2))*tspan)
    zspan = (1/traj_tlen)*tspan + 0.1
    zdotspan = (1/traj_tlen)*np.ones(int(len(tspan)))
elif testtype == 'desccirc':
    xspan = np.cos((2*(np.pi)/(traj_tlen/2))*tspan)
    xdotspan = -(2*(np.pi)/(traj_tlen/2))*np.sin((2*(np.pi)/(traj_tlen/2))*tspan)
    yspan = np.sin((2*(np.pi)/(traj_tlen/2))*tspan)
    ydotspan = (2*(np.pi)/(traj_tlen/2))*np.cos((2*(np.pi)/(traj_tlen/2))*tspan)
    zspan = -(1/traj_tlen)*tspan + 1.1
    zdotspan = -(1/traj_tlen)*np.ones(int(len(tspan)))
elif testtype == 'smalldesccirc':
    xspan = 0.4*np.cos((2*(np.pi)/(traj_tlen/2))*tspan)
    xdotspan = -0.4*(2*(np.pi)/(traj_tlen/2))*np.sin((2*(np.pi)/(traj_tlen/2))*tspan)
    yspan = 0.4*np.sin((2*(np.pi)/(traj_tlen/2))*tspan)
    ydotspan = 0.4*(2*(np.pi)/(traj_tlen/2))*np.cos((2*(np.pi)/(traj_tlen/2))*tspan)
    zspan = -(1/traj_tlen)*tspan + 1.1
    zdotspan = -(1/traj_tlen)*np.ones(int(len(tspan)))
else:
    xspan = (2/traj_tlen)*tspan
    xdotspan = (2/traj_tlen)*np.ones(int(len(tspan)))
    yspan = (2/traj_tlen)*tspan
    ydotspan = (2/traj_tlen)*np.ones(int(len(tspan)))
    zspan = 0.05*np.ones(int(len(tspan)))
    zdotspan = np.zeros(int(len(tspan)))

# Load chosen trajectory to x_ref_real matrix
x_ref_real = np.zeros((len(tspan),12))
x_ref_real[:,0] = xspan
x_ref_real[:,1] = yspan
x_ref_real[:,2] = zspan
x_ref_real[:,6] = xdotspan
x_ref_real[:,7] = ydotspan
x_ref_real[:,8] = zdotspan

# Calculate and display average distance between waypoints for the trajectory
trajdist = np.zeros(len(tspan)-1)
for n1 in range(len(tspan)-2):
    trajdist[n1] = np.sqrt((x_ref_real[n1+1,0]-x_ref_real[n1,0])**2 + (x_ref_real[n1+1,1]-x_ref_real[n1,1])**2 + (x_ref_real[n1+1,2]-x_ref_real[n1,2])**2)
print('Average Distance Between Waypoints = ')
print(np.mean(trajdist))

# Normalize trajectory
x_ref_real_norm = np.zeros((len(tspan),12))
for n1 in range(n):
    x_ref_real_norm[:,n1] = x_ref_real[:,n1]/x_scal[n1]

# Lift trajectory to lifted space z
if blo == 1:
    z_ref_real_blo = basis_blo(x_ref_real_norm)
if koopnet == 1:
    z_ref_real_koopnet = basis_koopnet(x_ref_real_norm)

# Obtain NMPC results
tnmpcstart = time()
[t_nmpc,x_nmpc,u_nmpc,itertime_nmpc,cpoint_nmpc,cpointdiff_nmpc] = quadmpc(x_ref_real,uhover,contw,geffect = groundeffect,ideal = 0)
tnmpcend = time()

# Obtain BLO MPC results
tkooprealsimstart1 = time()
if blo == 1:
    if discrete == 1:
        [t_blo,x_blo,u_blo,itertime_blo,cpoint_blo,cpointdiff_blo] = quadmpc_koop(x_ref_real,z_ref_real_blo,uhover,23,A_blo,B_blo,'discrete',basis_blo,x_scal,contw,geffect = groundeffect)
    else:
        [t_blo,x_blo,u_blo,itertime_blo,cpoint_blo,cpointdiff_blo] = quadmpc_koop(x_ref_real,z_ref_real_blo,uhover,23,A_blo,B_blo,'continuous',basis_blo,x_scal,contw,geffect = groundeffect)
tkooprealsimend1 = time()

# Obtain KoopNet MPC results
if koopnet == 1:
    tkooprealsimstart2 = time()
    [t_koopnet,x_koopnet,u_koopnet,itertime_koopnet,cpoint_koopnet,cpointdiff_koopnet] = quadmpc_koop(x_ref_real,z_ref_real_koopnet,uhover,23,A_koopnet,B_koopnet,'discrete',basis_koopnet,x_scal,contw,geffect = groundeffect)
    tkooprealsimend2 = time()

# Obtain Ideal NMPC results
if idealcont == 1:
    tidealnmpcstart = time()
    [t_idealnmpc,x_idealnmpc,u_idealnmpc,itertime_idealnmpc,cpoint_idealnmpc,cpointdiff_idealnmpc] = quadmpc(x_ref_real,uhover,contw,geffect = groundeffect,ideal = 1)
    tidealnmpcend = time()

# Display testing results

if blo == 1:
    print('BLO MPC Total Time = ')
    print(tkooprealsimend1 - tkooprealsimstart1)
if koopnet == 1:
    print('KoopNet MPC Total Time = ')
    print(tkooprealsimend2 - tkooprealsimstart2)
if idealcont == 1:
    print('Ideal NMPC Total Time = ')
    print(tidealnmpcend - tidealnmpcstart)
print('NMPC Total Time = ')
print(tnmpcend - tnmpcstart)

if blo == 1:
    print('BLO MPC Mean Time = ')
    print(np.mean(itertime_blo))
if koopnet == 1:
    print('KoopNet MPC Mean Time = ')
    print(np.mean(itertime_koopnet))
if idealcont == 1:
    print('Ideal NMPC Mean Time = ')
    print(np.mean(itertime_idealnmpc))
print('NMPC Mean Time = ')
print(np.mean(itertime_nmpc))

if blo == 1:
    print('BLO MPC Mean Error = ')
    print(np.mean(cpoint_blo))
if koopnet == 1:
    print('KoopNet MPC Mean Error = ')
    print(np.mean(cpoint_koopnet))
if idealcont == 1:
    print('Ideal NMPC Mean Error = ')
    print(np.mean(cpoint_idealnmpc))
print('NMPC Mean Error = ')
print(np.mean(cpoint_nmpc))

if blo == 1:
    print('BLO MPC Mean Thrust = ')
    print(np.mean(u_blo[:,0]))
if koopnet == 1:
    print('KoopNet MPC Mean Thrust = ')
    print(np.mean(u_koopnet[:,0]))
if idealcont == 1:
    print('Ideal NMPC Mean Thrust = ')
    print(np.mean(u_idealnmpc[:,0]))
print('NMPC Mean Thrust = ')
print(np.mean(u_nmpc[:,0]))

if blo == 1:
    print('BLO MPC/NMPC Mean Thrust Ratio = ')
    print(np.mean(u_blo[:,0])/np.mean(u_nmpc[:,0]))
if koopnet == 1:
    print('KoopNet MPC/NMPC Mean Thrust Ratio = ')
    print(np.mean(u_koopnet[:,0])/np.mean(u_nmpc[:,0]))
if idealcont == 1:
    print('Ideal NMPC/NMPC Mean Thrust Ratio = ')
    print(np.mean(u_idealnmpc[:,0])/np.mean(u_nmpc[:,0]))
print('NMPC/NMPC Mean Thrust Ratio = ')
print(np.mean(u_nmpc[:,0])/np.mean(u_nmpc[:,0]))

# Plot testing results

plt.figure()
plt.plot(itertime_nmpc,label='NMPC Iteration Time')
if idealcont == 1:
    plt.plot(itertime_idealnmpc,label='Ideal NMPC Iteration Time')
if blo == 1:
    plt.plot(itertime_blo,'--',label='BLO MPC Iteration Time')
if koopnet == 1:
    plt.plot(itertime_koopnet,':',label='KoopNet MPC Iteration Time')
plt.legend()
plt.xlabel('Iterations')
plt.ylabel('Time, s')
plt.yscale('log')
plt.show()

plt.figure()
plt.plot(cpoint_nmpc,label='NMPC Error')
if idealcont == 1:
    plt.plot(cpoint_idealnmpc,label='Ideal NMPC Error')
if blo == 1:
    plt.plot(cpoint_blo,'--',label='BLO MPC Error')
if koopnet == 1:
    plt.plot(cpoint_koopnet,':',label='KoopNet MPC Error')
plt.legend()
plt.xlabel('Waypoints')
plt.ylabel('Error, m')
plt.show()

plt.figure()
plt.plot(cpointdiff_nmpc,label='NMPC Convergence')
if idealcont == 1:
    plt.plot(cpointdiff_idealnmpc,label='Ideal NMPC Convergence')
if blo == 1:
    plt.plot(cpointdiff_blo,'--',label='BLO MPC Convergence')
if koopnet == 1:
    plt.plot(cpointdiff_koopnet,':',label='KoopNet MPC Convergence')
plt.legend()
plt.xlabel('Waypoints')
plt.ylabel('Distance, m')
plt.show()

fig1, ax1 = plt.subplots(3,sharex=True)
for n1 in np.arange(3):
    if n1 == 0:
        ax1[n1].plot(tspan,x_ref_real[:,n1],'b:',label='Target Trajectory')
        ax1[n1].plot(t_nmpc,x_nmpc[:,n1],'c-',label='NMPC Trajectory')
        if idealcont == 1:
            ax1[n1].plot(t_idealnmpc,x_idealnmpc[:,n1],'m-',label='Ideal NMPC Trajectory')
        if blo == 1:
            ax1[n1].plot(t_blo,x_blo[:,n1],'r--',label='BLO MPC Trajectory')
        if koopnet == 1:
            ax1[n1].plot(t_koopnet,x_koopnet[:,n1],'k--',label='KoopNet MPC Trajectory')
    else:
        ax1[n1].plot(tspan,x_ref_real[:,n1],'b:')
        ax1[n1].plot(t_nmpc,x_nmpc[:,n1],'c-')
        if idealcont == 1:
            ax1[n1].plot(t_idealnmpc,x_idealnmpc[:,n1],'m-')
        if blo == 1:
            ax1[n1].plot(t_blo,x_blo[:,n1],'r--')
        if koopnet == 1:
            ax1[n1].plot(t_koopnet,x_koopnet[:,n1],'k--')
ax1[0].set(ylabel='x, m')
ax1[1].set(ylabel='y, m')
ax1[2].set(ylabel='z, m')
fig1.legend()
fig1.supxlabel('Time, s')
plt.show()

plt.figure()
plt.plot(t_nmpc,x_nmpc[:,3],label='$\phi$ NMPC')
plt.plot(t_nmpc,x_nmpc[:,4],label=r'$\theta$ NMPC')
plt.plot(t_nmpc,x_nmpc[:,5],label='$\psi$ NMPC')
if idealcont == 1:
    plt.plot(t_idealnmpc,x_idealnmpc[:,3],label='$\phi$ Ideal NMPC')
    plt.plot(t_idealnmpc,x_idealnmpc[:,4],label=r'$\theta$ Ideal NMPC')
    plt.plot(t_idealnmpc,x_idealnmpc[:,5],label='$\psi$ Ideal NMPC')
if blo == 1:
    plt.plot(t_blo,x_blo[:,3],'--',label='$\phi$ BLO MPC')
    plt.plot(t_blo,x_blo[:,4],'--',label=r'$\theta$ BLO MPC')
    plt.plot(t_blo,x_blo[:,5],'--',label='$\psi$ BLO MPC')
if koopnet == 1:
    plt.plot(t_koopnet,x_koopnet[:,3],'--',label='$\phi$ KoopNet MPC')
    plt.plot(t_koopnet,x_koopnet[:,4],'--',label=r'$\theta$ KoopNet MPC')
    plt.plot(t_koopnet,x_koopnet[:,5],'--',label='$\psi$ KoopNet MPC')
plt.legend()
plt.xlabel('Time, s')
plt.ylabel(r'$\phi$ / $\theta$ / $\psi$, rad')
plt.show()

plt.figure()
plt.plot(t_nmpc,u_nmpc[:,0],label='T NMPC')
if idealcont == 1:
    plt.plot(t_idealnmpc,u_idealnmpc[:,0],label='T Ideal NMPC')
if blo == 1:
    plt.plot(t_blo,u_blo[:,0],'--',label='T BLO MPC')
if koopnet == 1:
    plt.plot(t_koopnet,u_koopnet[:,0],'--',label='T KoopNet MPC')
plt.plot([t_nmpc[0],t_nmpc[-1]],[hover_thrust,hover_thrust],'k:',label='T Hover')
plt.legend()
plt.xlabel('Time, s')
plt.ylabel('Thrust, N')
plt.show()

plt.figure()
plt.plot(t_nmpc,u_nmpc[:,1],label=r'$\tau_{x}$ NMPC')
plt.plot(t_nmpc,u_nmpc[:,2],label=r'$\tau_{y}$ NMPC')
plt.plot(t_nmpc,u_nmpc[:,3],label=r'$\tau_{z}$ NMPC')
if idealcont == 1:
    plt.plot(t_idealnmpc,u_idealnmpc[:,1],label=r'$\tau_{x}$ Ideal NMPC')
    plt.plot(t_idealnmpc,u_idealnmpc[:,2],label=r'$\tau_{y}$ Ideal NMPC')
    plt.plot(t_idealnmpc,u_idealnmpc[:,3],label=r'$\tau_{z}$ Ideal NMPC')
if blo == 1:
    plt.plot(t_blo,u_blo[:,1],'--',label=r'$\tau_{x}$ BLO MPC')
    plt.plot(t_blo,u_blo[:,2],'--',label=r'$\tau_{y}$ BLO MPC')
    plt.plot(t_blo,u_blo[:,3],'--',label=r'$\tau_{z}$ BLO MPC')
if koopnet == 1:
    plt.plot(t_koopnet,u_koopnet[:,1],'--',label=r'$\tau_{x}$ KoopNet MPC')
    plt.plot(t_koopnet,u_koopnet[:,2],'--',label=r'$\tau_{y}$ KoopNet MPC')
    plt.plot(t_koopnet,u_koopnet[:,3],'--',label=r'$\tau_{z}$ KoopNet MPC')
plt.legend()
plt.xlabel('Time, s')
plt.ylabel('Torque, Nm')
plt.show()

plt.figure()
ax = plt.axes(projection='3d')
ax.plot3D(x_nmpc[:,0],x_nmpc[:,1],x_nmpc[:,2],'c-',label='NMPC Trajectory')
if idealcont == 1:
    ax.plot3D(x_idealnmpc[:,0],x_idealnmpc[:,1],x_idealnmpc[:,2],'m-',label='Ideal NMPC Trajectory')
if blo == 1:
    ax.plot3D(x_blo[:,0],x_blo[:,1],x_blo[:,2],'r--',label='BLO MPC Trajectory')
if koopnet == 1:
    ax.plot3D(x_koopnet[:,0],x_koopnet[:,1],x_koopnet[:,2],'k--',label='KoopNet MPC Trajectory')
ax.plot3D(xspan,yspan,zspan,'b:',label='Target Trajectory')
plt.legend()
ax.set_xlabel('x, m')
ax.set_ylabel('y, m')
ax.set_zlabel('z, m')
plt.show()

