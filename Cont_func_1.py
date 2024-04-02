import argparse
import jax.numpy as jn
import jax
import matplotlib.pyplot as plt
import numpy as np
import optax
import scipy.integrate as spi

jax.config.update("jax_enable_x64", True)

from Disc_func_1 import KBF_ENC, init_mat, fit_model, plt_hist, sample_unif_2d, normalize

G  = 9.8  # acceleration due to gravity, in m/s^2
L1 = 1.0  # length of pendulum 1 in m
L2 = 1.0  # length of pendulum 2 in m
M1 = 1.0  # mass of pendulum 1 in kg
M2 = 1.0  # mass of pendulum 2 in kg

_NC = {
        '1' : [0.5, 0.5],          # Trapezoid
        '2' : [1./3, 4./3, 1./3],  # Simpson 1/3
        '3' : [3./8, 9./8, 9./8, 3./8],  # Simpson 3/8
        '4' : [14./45, 64./45, 24./45, 64./45, 14./45],  # Boole
    }

def dyn_dp(t, state, u=None):
    _u = u(t)
    dydx = np.zeros_like(state)
    dydx[0] = state[1]

    del_ = state[2] - state[0]
    sin_ = np.sin(del_)
    cos_ = np.cos(del_)
    den1 = (M1 + M2)*L1 - M2*L1*cos_*cos_
    dydx[1] = (M2*L1*state[1]*state[1]*sin_*cos_ + M2*G*np.sin(state[2])*cos_ +
               M2*L2*state[3]*state[3]*sin_ - (M1 + M2)*G*np.sin(state[0]) + _u[0])/den1 - state[1]

    dydx[2] = state[3]

    den2 = (L2/L1)*den1
    dydx[3] = (-M2*L2*state[3]*state[3]*sin_*cos_ +
               (M1 + M2)*G*np.sin(state[0])*cos_ -
               (M1 + M2)*L1*state[1]*state[1]*sin_ -
               (M1 + M2)*G*np.sin(state[2]) + _u[1])/den2 - state[3]

    return dydx

def solIVP_dp(ts, x0, u=None):
    if u is None:
        inp = 0.5*(np.random.rand(2) - 0.5)
    else:
        inp = u
    finp = lambda t: (np.ones_like(t).reshape(-1,1)*inp).squeeze()
    sol = spi.solve_ivp(dyn_dp, [0, ts[-1]], x0, t_eval=ts, args=(finp,), method='RK45')
    tmp = np.hstack([sol.y.T, finp(ts)])
    return tmp

class CKBF(KBF_ENC):
    """Full autoencoder form for continuous-time KBF."""
    def init_params(self):
        """Using a flattened system matrix."""
        _p = super().init_params()
        _p.update(As = init_mat([self.Nk, self.Nk*(self.Nu+1)], 415411))
        return _p

    def features(self, x: jn.ndarray, u: jn.ndarray, params: jn.ndarray) -> jn.ndarray:
        """Needed in training."""
        z = self.encoder(x, params)
        w = jn.hstack([[1], u]).reshape(-1,1)
        f = (w*z).reshape(-1)
        return f

    def dynamics(self, z: jn.ndarray, u: jn.ndarray, params: jn.ndarray) -> jn.ndarray:
        """For prediction."""
        w = jn.hstack([[1], u]).reshape(-1,1)
        f = (w*z).reshape(-1)
        dz = jn.dot(params['As'], f)
        return dz

class CKBF_STK(CKBF):
    """Stacked encoder case.  Always assuming 1 on top."""
    def __init__(self, dims, nums, ifone, act):
        self.Nx, self.Nu, self.Nk = dims
        self.Nns   = np.atleast_1d(nums)
        self.Nl    = len(nums)+1
        self.ifone = ifone
        self.act   = act

        self.encoder = self.encoder_stacked

    def init_params(self):
        enc = np.hstack([[self.Nx], self.Nns, [self.Nk]])
        enc[-1] -= 1  # Saving for 1

        _p = {}
        for _i in range(self.Nl):
            _p.update({f'en{_i}': init_mat(enc[_i:_i+2].astype(int), _i)})
            _p.update({f'eb{_i}': init_mat((enc[_i+1].astype(int),), _i+200)})
        _p.update(As = init_mat([self.Nk + self.Nx, (self.Nk + self.Nx)*(self.Nu+1)], 415411))

        return _p

    def encoder_stacked(self, x: jn.ndarray, params: jn.ndarray) -> jn.ndarray:
        _x = jn.atleast_2d(x)
        for _i in range(self.Nl-1):
            x = jn.dot(x, params[f'en{_i}']) + params[f'eb{_i}']
            x = self.act(x)
        x = jn.dot(x, params[f'en{self.Nl-1}']) + params[f'eb{self.Nl-1}']
        Nt = len(_x)
        _z = jn.hstack([np.ones((Nt,1)), _x, jn.atleast_2d(x)])
        return _z.squeeze()

    def decoder(self, z: jn.ndarray, params: jn.ndarray) -> jn.ndarray:
        return z[:,1:self.Nx+1]

def kbf_predict(model, params, x0, ts, ut):
    def f(t, z):
        # z = model.encoder(model.decoder(z, params), params)
        dz = model.dynamics(z, ut(t), params)
        return np.array(dz)
    z0  = model.encoder(x0, params)
    sol = spi.solve_ivp(f, [0, ts[-1]], z0, t_eval=ts, method='RK45')
    xs  = model.decoder(sol.y.T, params)
    return np.array(xs)

def weightNC(N, order, dt):
    if order == 0:
        _W = np.ones(N,)
        _W[-1] = 0.0
        return jn.array(_W*dt)
    assert (N-1) % order == 0  # Check required number of samples
    _w = _NC[str(order)]
    _W = np.zeros(N,)
    _N = (N-1) // order
    for _i in range(_N):
        _W[_i*order:(_i+1)*order+1] += _w
    return jn.array(_W*dt)

def make_slo_loss(KBF: CKBF, ltraj, ordint, dt):
    #W = weightNC(ltraj, ordint, dt) # Old Formulation
    W = jn.zeros(ltraj-1)
    #W = jn.zeros(ltraj-2)
    W = W.at[:].set(dt/2)
    #W = W.at[:].set(dt)
    #W = W.at[0].set(dt/2)
    #W = W.at[-1].set(dt/2)

    def _feat(params: optax.Params, x: jn.ndarray, u: jn.ndarray) -> jn.ndarray:
        return KBF.features(x, u, params)
    vfeat = jax.vmap(_feat, in_axes=(None,0,0))

    def newfeatfunc(z1: jn.ndarray, z2: jn.ndarray, u: jn.ndarray) -> jn.ndarray:
        zsum = z1 + z2
        cont = jn.hstack([[1], u]).reshape(-1,1)
        newfeat = (cont*zsum).reshape(-1)
        return newfeat
    newfeat = jax.vmap(newfeatfunc, in_axes=(0,0,0))
    #newfeat = newfeatfunc

    def slo_loss_single(params: optax.Params, traj: jn.ndarray) -> jn.ndarray:
        # Variables
        xs = traj[:,:KBF.Nx]
        us = traj[:,KBF.Nx:]
        zs = KBF.encoder(xs, params)

        # Reconstruction
        xd = KBF.decoder(zs, params)
        Lr = jn.mean((xd-xs)**2)

        # Dynamics
        #fs = vfeat(params, xs, us) # Old Formulation
        fs = newfeat(zs[:-1,:], zs[1:,:], us[:-1,:])
        #fs = newfeat(zs[1:-1,:], zs[2:,:], us[1:-1,:])
        dz = jn.dot(W, jn.dot(fs, params['As'].T))
        _d = zs[-1]-zs[0]-dz
        Ld = jn.mean(_d**2) / ltraj
        #Ld = jn.mean(_d**2) / (ltraj-1)

        return jn.array([Lr, Ld])

    vslo = jax.vmap(slo_loss_single, in_axes=(None,0))
    def _slo_loss(params: optax.Params, batch: jn.ndarray) -> jn.ndarray:
        Ls = vslo(params, batch)
        Ls = jn.mean(Ls, axis=0)
        L  = jn.sum(Ls)
        return L, dict(L=L, Lr=Ls[0], Ld=Ls[1])

    slo_loss_vg = jax.value_and_grad(_slo_loss, has_aux=True)
    def slo_loss(params: optax.Params, batch: jn.ndarray) -> jn.ndarray:
        (_, losses), grads = slo_loss_vg(params, batch)
        return grads, losses

    # ---------------------------------------

    def _pre_lr(params: optax.Params, traj: jn.ndarray) -> jn.ndarray:
        xs = traj[:,:KBF.Nx]
        us = traj[:,KBF.Nx:]
        zs = KBF.encoder(xs, params)
        #fs = vfeat(params, xs, us) # Old Formulation
        fs = newfeat(zs[:-1,:], zs[1:,:], us[:-1,:])
        #fs = newfeat(zs[1:-1,:], zs[2:,:], us[1:-1,:])
        df = jn.dot(W, fs).squeeze()
        dz = zs[-1]-zs[0]
        return dz, df
    v_pre_lr = jax.vmap(_pre_lr, in_axes=(None,0))

    def param_reset(params: optax.Params, batch: jn.ndarray) -> jn.ndarray:
        trajs = jn.vstack(batch)
        dz, df = v_pre_lr(params, trajs)
        _A = jn.linalg.lstsq(df, dz)[0].T
        params.update(As = _A)
        if 'de' in params.keys():
            # Linear decoder
            xs = jn.vstack(trajs[:,:,:KBF.Nx])
            zs = KBF.encoder(xs, params)
            _C = jn.linalg.lstsq(zs, xs)[0]
            params.update(de = _C)
        return params

    return slo_loss, param_reset

def gen_data(solver, ts, x0s, horizon, batch):
    dat = []
    for _x0 in x0s:
        sol = solver(ts, _x0)
        dat.append(sol.reshape(len(sol)//horizon, horizon, -1))
    dat = np.concatenate(dat)
    Nd  = len(dat)
    dat = dat.reshape(Nd//batch, batch, horizon, -1)
    return dat

def plt_data_nd(dat):
    _, N = dat[0][0].shape
    f, ax = plt.subplots(nrows=N)
    for _t in dat:
        for _b in _t:
            for _i in range(N):
                ax[_i].plot(_b[:,_i])
    return f



