import argparse
import jax.numpy as jn
import jax
import matplotlib.pyplot as plt
import numpy as np
import optax
import scipy.integrate as spi

from koopman_core.learning import KoopDnn_modified, KoopmanNetCtrl

from Disc_func_1 import fit_model, plt_hist

from Cont_func_1 import CKBF_STK, make_slo_loss

def KoopNetfunc(t_val,x_train,x_val,u_train,u_val,net_params):
    
    if len(u_train[0,:,0]) == len(x_train[0,:,0]):
        u_train = u_train[:,:-1,:]
        u_val = u_val[:,:-1,:]

    n_traj_train = len(x_train[:,0,0])
    n_traj_val = len(x_val[:,0,0])

    net = KoopmanNetCtrl(net_params)
    model_koop_dnn = KoopDnn_modified(net)
    model_koop_dnn.set_datasets(x_train, np.tile(t_val,(n_traj_train,1)), u_train=u_train, x_val=x_val, u_val=u_val, t_val=np.tile(t_val,(n_traj_val,1)))
    model_koop_dnn.model_pipeline(net_params)
    model_koop_dnn.construct_koopman_model()

    train_loss = [l[0] for l in model_koop_dnn.train_loss_hist]
    train_pred_loss = [l[1] for l in model_koop_dnn.train_loss_hist]
    train_lin_loss = [l[2] for l in model_koop_dnn.train_loss_hist]
    val_loss = [l[0] for l in model_koop_dnn.val_loss_hist]
    val_pred_loss = [l[1] for l in model_koop_dnn.val_loss_hist]
    val_lin_loss = [l[2] for l in model_koop_dnn.val_loss_hist]
    epochs = np.arange(0, net_params['epochs'])

    plt.figure(figsize=(15,8))
    plt.plot(epochs, train_loss, color='tab:orange', label='Training loss')
    plt.plot(epochs, train_pred_loss, '--', color='tab:orange', label='Training prediction loss')
    plt.plot(epochs, train_lin_loss, ':', color='tab:orange', label='Training linear loss')
    plt.plot(epochs, val_loss, color='tab:blue', label='Validation loss')
    plt.plot(epochs, val_pred_loss, '--', color='tab:blue', label='Validation prediction loss')
    plt.plot(epochs, val_lin_loss, ':', color='tab:blue', label='Validation linear loss')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.show()

    A_koop = np.array(model_koop_dnn.A, dtype=float)
    B_koop = np.array(model_koop_dnn.B, dtype=float)
    C_koop = np.array(model_koop_dnn.C, dtype=float)
    basis_koop = model_koop_dnn.basis_encode

    return A_koop,B_koop,basis_koop

def BLOfunc(t_train,x_train,u_train,net_params,disc = 0):
    
    BATCH_SIZE = net_params['batch_size']
    HORIZON_LENGTH = len(t_train)
    if disc == 1:
        INT_ORDER = 0
    else:
        #INT_ORDER = 4
        INT_ORDER = 1
    NUM_TRAIN_STEPS = net_params['epochs']
    suf = f'ci_b{BATCH_SIZE}h{HORIZON_LENGTH}'

    n = net_params['state_dim']
    m = net_params['ctrl_dim']
    nz = net_params['encoder_output_dim'] + net_params['state_dim'] + 1

    cas = 'slo'

    kbf = CKBF_STK([n, m, net_params['encoder_output_dim']+1], net_params['encoder_hidden_width']*np.ones(net_params['encoder_hidden_depth']), True, jax.nn.swish)

    make_loss = make_slo_loss

    if disc == 1:
        new_x_train = np.zeros((len(x_train[:,0,0])*(HORIZON_LENGTH-1),2,n))
        new_u_train = np.zeros((len(u_train[:,0,0])*(HORIZON_LENGTH-1),2,m))
        iter = 0
        for n1 in range(len(x_train[:,0,0])):
            for n2 in range(HORIZON_LENGTH-1):
                new_x_train[iter,0,:] = x_train[n1,n2,:]
                new_x_train[iter,1,:] = x_train[n1,n2+1,:]
                for n3 in range(m):
                    new_u_train[iter,:,n3] = u_train[n1,n2,n3]
                iter = iter + 1
        HORIZON_LENGTH = 2
        t_train = t_train[:2]
        x_train = new_x_train
        u_train = new_u_train

    TRAINING_DATA = np.zeros((len(x_train[:,0,0]),HORIZON_LENGTH,n+m))
    TRAINING_DATA[:,:,0:n] = x_train
    TRAINING_DATA[:,:,n:n+m] = u_train

    TRAINING_DATA = np.reshape(TRAINING_DATA,(len(x_train[:,0,0])//BATCH_SIZE,BATCH_SIZE,HORIZON_LENGTH,n+m))

    print(np.shape(TRAINING_DATA))

    lss, rst = make_loss(kbf, HORIZON_LENGTH, INT_ORDER, net_params['dt'])

    optimizer = optax.adam(learning_rate=net_params['lr'])

    fit = fit_model(f'res/{cas}_{suf}', lss, TRAINING_DATA, NUM_TRAIN_STEPS, 200000, len(TRAINING_DATA[:,0,0,0])-1, reset=rst, init_only=False)
    ini = kbf.init_params()
    params, hist = fit(ini, optimizer)

    f = plt_hist(hist, ['L', 'Lr', 'Ld'], ['b-', 'r--', 'k--'])
    plt.show()

    Atotal = params['As']

    A_blo = np.zeros((nz,nz))
    A_blo[:,:] = Atotal[:,:nz]

    B_blo = np.zeros((m,nz,nz))

    y1 = 0
    for n1 in range(m):
        y1 = y1 + 1
        B_blo[n1,:,:] = Atotal[:,(nz*y1):(nz*(y1+1))]

    if disc == 1:
        A_blo_disc = np.zeros(np.shape(A_blo))
        B_blo_disc = np.zeros(np.shape(B_blo))
        A_blo_disc = np.identity(nz) + A_blo*(net_params['dt'])
        for n1 in range(m):
            B_blo_disc[n1,:,:] = B_blo[n1,:,:]*(net_params['dt'])
        A_blo = A_blo_disc
        B_blo = B_blo_disc

    def basis_blo(x):
        return np.array(kbf.encoder(x,params))

    return A_blo,B_blo,basis_blo

def BLOfunc_save(t_train,x_train,u_train,net_params,twostep = 0):
    
    BATCH_SIZE = net_params['batch_size']
    HORIZON_LENGTH = len(t_train)
    INT_ORDER = 1
    NUM_TRAIN_STEPS = net_params['epochs']
    suf = f'ci_b{BATCH_SIZE}h{HORIZON_LENGTH}'

    n = net_params['state_dim']
    m = net_params['ctrl_dim']
    nz = net_params['encoder_output_dim'] + net_params['state_dim'] + 1

    cas = 'slo'

    kbf = CKBF_STK([n, m, net_params['encoder_output_dim']+1], net_params['encoder_hidden_width']*np.ones(net_params['encoder_hidden_depth']), True, jax.nn.swish)

    make_loss = make_slo_loss

    if twostep == 1:
        new_x_train = np.zeros((len(x_train[:,0,0])*(HORIZON_LENGTH-1),2,n))
        new_u_train = np.zeros((len(u_train[:,0,0])*(HORIZON_LENGTH-1),2,m))
        iter = 0
        for n1 in range(len(x_train[:,0,0])):
            for n2 in range(HORIZON_LENGTH-1):
                new_x_train[iter,0,:] = x_train[n1,n2,:]
                new_x_train[iter,1,:] = x_train[n1,n2+1,:]
                for n3 in range(m):
                    new_u_train[iter,:,n3] = u_train[n1,n2,n3]
                iter = iter + 1
        HORIZON_LENGTH = 2
        t_train = t_train[:2]
        x_train = new_x_train
        u_train = new_u_train

    TRAINING_DATA = np.zeros((len(x_train[:,0,0]),HORIZON_LENGTH,n+m))
    TRAINING_DATA[:,:,0:n] = x_train
    TRAINING_DATA[:,:,n:n+m] = u_train

    TRAINING_DATA = np.reshape(TRAINING_DATA,(len(x_train[:,0,0])//BATCH_SIZE,BATCH_SIZE,HORIZON_LENGTH,n+m))

    print(np.shape(TRAINING_DATA))

    lss, rst = make_loss(kbf, HORIZON_LENGTH, INT_ORDER, net_params['dt'])

    optimizer = optax.adam(learning_rate=net_params['lr'])

    fit = fit_model(f'res/{cas}_{suf}', lss, TRAINING_DATA, NUM_TRAIN_STEPS, 200000, len(TRAINING_DATA[:,0,0,0])-1, reset=rst, init_only=False)
    ini = kbf.init_params()
    params, hist = fit(ini, optimizer)

    f = plt_hist(hist, ['L', 'Lr', 'Ld'], ['b-', 'r--', 'k--'])
    plt.show()

    Atotal = params['As']

    A_blo = np.zeros((nz,nz))
    A_blo[:,:] = Atotal[:,:nz]

    B_blo = np.zeros((m,nz,nz))

    y1 = 0
    for n1 in range(m):
        y1 = y1 + 1
        B_blo[n1,:,:] = Atotal[:,(nz*y1):(nz*(y1+1))]

    return A_blo,B_blo,params,hist

