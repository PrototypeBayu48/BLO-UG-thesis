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

import pypose as pp

n = 12
m = 4
dt = 0.02

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
hover_thrust_norm = hover_thrust/(0.1*max_thrust)

print(hover_thrust)
print(max_thrust)
print(uhover)
print(uhover_norm)

n_traj_train = 8
n_traj_val = int(0.25*n_traj_train)

datagen_iter = 12000

t_val = np.arange(0,240 + 0.02,0.02)

x_train = np.load('x_train_10.npy')
w_train = np.load('u_train_10.npy')
x_val = np.load('x_val_10.npy')
w_val = np.load('u_val_10.npy')

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

plt.figure()
plt.plot(t_val,x_train[n_traj_train - 1,:,0],label = 'x train')
plt.plot(t_val,x_train[n_traj_train - 1,:,1],label = 'y train')
plt.plot(t_val,x_train[n_traj_train - 1,:,2],label = 'z train')
plt.plot(t_val,x_val[n_traj_val - 1,:,0],':',label = 'x val')
plt.plot(t_val,x_val[n_traj_val - 1,:,1],':',label = 'y val')
plt.plot(t_val,x_val[n_traj_val - 1,:,2],':',label = 'z val')
plt.legend()
plt.show()

plt.figure()
plt.plot(t_val,x_train[n_traj_train - 1,:,3],label = 'phi train')
plt.plot(t_val,x_train[n_traj_train - 1,:,4],label = 'theta train')
plt.plot(t_val,x_train[n_traj_train - 1,:,5],label = 'psi train')
plt.plot(t_val,x_val[n_traj_val - 1,:,3],':',label = 'phi val')
plt.plot(t_val,x_val[n_traj_val - 1,:,4],':',label = 'theta val')
plt.plot(t_val,x_val[n_traj_val - 1,:,5],':',label = 'psi val')
plt.legend()
plt.show()

plt.figure()
plt.plot(np.arange(0,240,0.02),u_train[n_traj_train - 1,:,0],label='Th train normalized')
plt.plot(np.arange(0,240,0.02),u_val[n_traj_val - 1,:,0],':',label='Th val normalized')
plt.legend()
plt.show()

plt.figure()
plt.plot(np.arange(0,240,0.02),u_train[n_traj_train - 1,:,1],label='tx train normalized')
plt.plot(np.arange(0,240,0.02),u_train[n_traj_train - 1,:,2],label='ty train normalized')
plt.plot(np.arange(0,240,0.02),u_train[n_traj_train - 1,:,3],label='tz train normalized')
plt.plot(np.arange(0,240,0.02),u_val[n_traj_val - 1,:,1],':',label='tx val normalized')
plt.plot(np.arange(0,240,0.02),u_val[n_traj_val - 1,:,2],':',label='ty val normalized')
plt.plot(np.arange(0,240,0.02),u_val[n_traj_val - 1,:,3],':',label='tz val normalized')
plt.legend()
plt.show()

plt.figure()
ax = plt.axes(projection='3d')
ax.plot3D(x_train[n_traj_train - 1,:,0],x_train[n_traj_train - 1,:,1],x_train[n_traj_train - 1,:,2],label='train trajectory')
ax.plot3D(x_val[n_traj_val - 1,:,0],x_val[n_traj_val - 1,:,1],x_val[n_traj_val - 1,:,2],':',label='val trajectory')
plt.legend()
plt.show()

net_params = {}
net_params['state_dim'] = n
net_params['ctrl_dim'] = m
net_params['encoder_hidden_width'] = 100
net_params['encoder_hidden_depth'] = 2
net_params['encoder_output_dim'] = 10
net_params['optimizer'] = 'adam'
net_params['activation_type'] = 'tanh'
net_params['lr'] = 1e-3
net_params['epochs'] = 200
net_params['batch_size'] = 200
net_params['lin_loss_penalty'] = 2e-1
net_params['l2_reg'] = 5e-4
net_params['l1_reg'] = 0.0
net_params['n_fixed_states'] = 12
net_params['first_obs_const'] = True
net_params['override_kinematics'] = True
net_params['override_C'] = True
net_params['dt'] = dt

print(net_params)

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

kooploss = np.zeros((len(epochs),6))
kooploss[:,0] = train_loss
kooploss[:,1] = train_pred_loss
kooploss[:,2] = train_lin_loss
kooploss[:,3] = val_loss
kooploss[:,4] = val_pred_loss
kooploss[:,5] = val_lin_loss

A_koop = np.array(model_koop_dnn.A, dtype=float)
B_koop = np.array(model_koop_dnn.B, dtype=float)
C_koop = np.array(model_koop_dnn.C, dtype=float)
basis_koop = model_koop_dnn.basis_encode

np.save('A_datagen10_1',A_koop)
np.save('B_datagen10_1',B_koop)

np.save('loss_datagen10_1',kooploss)

torch.save(net,'net_datagen10_1.pt')

