import argparse
import jax.numpy as jn
import jax
import matplotlib.pyplot as plt
import numpy as np
import optax
import copy

def dyn_dsc_inp(x, u, ml):
    MU, LM = ml
    return np.array([
        MU*x[0] + u[0]*x[0],
        LM*(x[1]-x[0]**2) + u[1]*x[0]**2])

def solIVP_di(ml, nt, x0, u=None):
    sol = np.zeros((nt+1,2))
    if u is None:
        inp = 0.5*(np.random.rand(nt+1,2) - 1)
    else:
        inp = u
    sol[0] = x0
    for _i in range(1, nt):
        sol[_i] = dyn_dsc_inp(sol[_i-1], inp[_i-1], ml)
    return np.hstack([sol[:-1], inp[:-1]])

def init_mat(shape, key):
    return jax.random.normal(shape=shape, key=jax.random.PRNGKey(key)) * jn.sqrt(2/shape[0])

class KBF_base:
    def init_params(self):
        """Initialize the parameter list."""
        raise NotImplementedError("This is the base class.")

    def encoder(self, x: jn.ndarray, params: jn.ndarray) -> jn.ndarray:
        """Map physical states to Koopman states."""
        raise NotImplementedError("This is the base class.")

    def decoder(self, z: jn.ndarray, params: jn.ndarray) -> jn.ndarray:
        """Map Koopman states to physical states."""
        raise NotImplementedError("This is the base class.")

    def dynamics(self, z: jn.ndarray, u: jn.ndarray, params: jn.ndarray) -> jn.ndarray:
        """Advance one step in Koopman space."""
        raise NotImplementedError("This is the base class.")

    def predict(self, x0: jn.ndarray, us: jn.ndarray, params: jn.ndarray) -> jn.ndarray:
        """KBF-based prediction, returning both Koopman and physical states."""
        raise NotImplementedError("This is the base class.")

    def adjoint(self, dl: jn.ndarray, us: jn.ndarray, params: jn.ndarray) -> jn.ndarray:
        """Sensitivity of the prediction process.  Needed for FLO."""
        raise NotImplementedError("This is the base class.")

    def constraint_part(self, us: jn.ndarray, zs: jn.ndarray, params: jn.ndarray) -> jn.ndarray:
        """Constraint values without initial conditions.  Diagnostic purpose."""
        raise NotImplementedError("This is the base class.")

    def constraint_full(self, us: jn.ndarray, zs: jn.ndarray, x0: jn.ndarray, params: jn.ndarray) -> jn.ndarray:
        """Constraint values with initial conditions."""
        raise NotImplementedError("This is the base class.")

    def constraint_mat_part(self, us: jn.ndarray, params: jn.ndarray) -> jn.ndarray:
        """Linear matrix in constraints without initial conditions.  Diagnostic purpose."""
        raise NotImplementedError("This is the base class.")

    def constraint_mat_full(self, us: jn.ndarray, params: jn.ndarray) -> jn.ndarray:
        """Linear matrix in constraints with initial conditions.  Diagnostic purpose."""
        raise NotImplementedError("This is the base class.")

class KBF_ENC(KBF_base):
    """Full autoencoder form for KBF."""
    def __init__(self, dims, nums, ifone, act):
        self.Nx, self.Nu, self.Nk = dims
        self.Nns   = np.atleast_1d(nums)
        self.Nl    = len(nums)+1
        self.ifone = ifone
        self.act   = act

        if ifone:
            self.encoder = self._encoder_one
        else:
            self.encoder = self._encoder_smpl

    def init_params(self):
        enc = np.hstack([[self.Nx], self.Nns, [self.Nk]])
        dec = np.array(enc[::-1])
        if self.ifone:
            enc[-1] -= 1

        _p = {}
        for _i in range(self.Nl):
            _p.update({f'en{_i}': init_mat(enc[_i:_i+2], _i)})
            _p.update({f'eb{_i}': init_mat((enc[_i+1],), _i+200)})
            _p.update({f'de{_i}': init_mat(dec[_i:_i+2], _i+100)})
            _p.update({f'db{_i}': init_mat((dec[_i+1],), _i+300)})
        _p.update(As = init_mat([self.Nk, self.Nk, self.Nu+1], 415411))

        return _p

    def _encoder_smpl(self, x: jn.ndarray, params: jn.ndarray) -> jn.ndarray:
        for _i in range(self.Nl-1):
            x = jn.dot(x, params[f'en{_i}']) + params[f'eb{_i}']
            x = self.act(x)
        x = jn.dot(x, params[f'en{self.Nl-1}']) + params[f'eb{self.Nl-1}']
        return x

    def _encoder_one(self, x: jn.ndarray, params: jn.ndarray) -> jn.ndarray:
        x = jn.atleast_2d(self._encoder_smpl(x, params))
        return jn.hstack([jn.ones((len(x),1)), x]).squeeze()

    def decoder(self, z: jn.ndarray, params: jn.ndarray) -> jn.ndarray:
        for _i in range(self.Nl-1):
            z = jn.dot(z, params[f'de{_i}']) + params[f'db{_i}']
            z = self.act(z)
        z = jn.dot(z, params[f'de{self.Nl-1}']) + params[f'db{self.Nl-1}']
        return z

    def dynamics(self, z: jn.ndarray, u: jn.ndarray, params: jn.ndarray) -> jn.ndarray:
        w = jn.hstack([[1], u])
        dz = jn.dot(jn.dot(params['As'], w), z)
        return dz

    def predict(self, x0: jn.ndarray, us: jn.ndarray, params: jn.ndarray) -> jn.ndarray:
        Nt = len(us)
        zp, xp = [self.encoder(x0, params)], [x0]
        for _i in range(1,Nt):
            zp.append( self.dynamics(zp[-1], us[_i-1], params) )
            xp.append( self.decoder(zp[-1], params) )
        return jn.vstack(zp), jn.vstack(xp)

    def adjoint(self, dl: jn.ndarray, us: jn.ndarray, params: jn.ndarray) -> jn.ndarray:
        Nt = len(us)
        mu = [-dl[-1]]
        for _i in range(Nt-2,-1,-1):
            Ai = params['As'].dot(jn.hstack([1, us[_i]]))
            mu.append( Ai.T.dot(mu[-1]) - dl[_i] )
        return jn.flipud(jn.array(mu))

    def constraint_part(self, us: jn.ndarray, zs: jn.ndarray, params: jn.ndarray) -> jn.ndarray:
        Nt = len(us)
        g = []
        for _i in range(Nt-1):
            Ai = params['As'].dot(jn.hstack([1, us[_i]]))
            g.append(Ai.dot(zs[_i])-zs[_i+1])
        return jn.hstack(g)

    def constraint_full(self, us: jn.ndarray, zs: jn.ndarray, x0: jn.ndarray, params: jn.ndarray) -> jn.ndarray:
        z0 = self.encoder(x0, params)
        g0 = z0-zs[0]
        gs = self.constraint_part(us, zs, params)
        return jn.hstack([g0, gs])

    def constraint_mat_part(self, us: jn.ndarray, params: jn.ndarray) -> jn.ndarray:
        Nt, Nz = len(us), self.Nk
        II = -jn.eye(Nz)
        M  = jn.zeros((Nz*(Nt-1), Nz*Nt))
        for _i in range(Nt-1):
            Ai = params['As'].dot(jn.hstack([1, us[_i]]))
            M  = M.at[Nz*_i:Nz*(_i+1),Nz*_i:Nz*(_i+1)].set(Ai)
            M  = M.at[Nz*_i:Nz*(_i+1),Nz*(_i+1):Nz*(_i+2)].set(II)
        return M

    def constraint_mat_full(self, us: jn.ndarray, params: jn.ndarray) -> jn.ndarray:
        N = jn.zeros((self.Nk, self.Nk*Nt))
        N = N.at[:, :self.Nk].set(-jn.eye(self.Nk))
        M = self.constraint_mat_part(us, params)
        return np.vstack([N, M])

class KBF_LND(KBF_ENC):
    """Linear decoder case."""
    def init_params(self):
        enc = np.hstack([[self.Nx], self.Nns, [self.Nk]])
        if self.ifone:
            enc[-1] -= 1

        _p = {}
        for _i in range(self.Nl):
            _p.update({f'en{_i}': init_mat(enc[_i:_i+2], _i)})
            _p.update({f'eb{_i}': init_mat((enc[_i+1],), _i+200)})
        _p.update(de = init_mat([self.Nk, self.Nx], 100))
        _p.update(As = init_mat([self.Nk, self.Nk, self.Nu+1], 415411))

        return _p

    def decoder(self, z: jn.ndarray, params: jn.ndarray) -> jn.ndarray:
        z = jn.dot(z, params['de'])
        return z

def make_slo_loss(KBF: KBF_base):

    def slo_loss_single(params: optax.Params, data: jn.ndarray) -> jn.ndarray:
        # Variables
        xs = data[:,:KBF.Nx]
        us = data[:,KBF.Nx:]
        zs = KBF.encoder(xs, params)

        # Reconstruction
        xd = KBF.decoder(zs, params)
        Lr = jn.mean((xd-xs)**2)

        # Dynamics
        zp, xp = KBF.predict(xs[0], us, params)

        # Losses
        Lz = jn.mean((zp-zs)**2)
        Lx = jn.mean((xp-xs)**2)

        return jn.array([Lr, Lz, Lx])

    def slo_loss_dyn(params: optax.Params, data: jn.ndarray) -> jn.ndarray:
        # Variables
        xs = data[:,:KBF.Nx]
        us = data[:,KBF.Nx:]
        zs = KBF.encoder(xs, params)
        Lc = jn.max(jn.abs(KBF.constraint_full(us, zs, xs[0], params)))
        return Lc

    vslo = jax.vmap(slo_loss_single, in_axes=(None,0))
    vsld = jax.vmap(slo_loss_dyn, in_axes=(None,0))
    def _slo_loss(params: optax.Params, batch: jn.ndarray) -> jn.ndarray:
        Ls = vslo(params, batch)
        Ls = jn.mean(Ls, axis=0)
        L  = jn.sum(Ls)
        Ld = vsld(params, batch)
        return L, dict(L=L, Lr=Ls[0], Lz=Ls[1], Lx=Ls[2], Lc=jn.mean(Ld))

    slo_loss_vg = jax.value_and_grad(_slo_loss, has_aux=True)
    def slo_loss(params: optax.Params, batch: jn.ndarray) -> jn.ndarray:
        (_, losses), grads = slo_loss_vg(params, batch)
        return grads, losses

    # ---------------------------------------

    def _feat(params: optax.Params, x: jn.ndarray, u: jn.ndarray) -> jn.ndarray:
        z = KBF.encoder(x, params)
        w = jn.hstack([[1], u]).reshape(-1,1)
        f = (w*z).reshape(-1)
        return f
    vfeat = jax.vmap(_feat, in_axes=(None,0,0))

    def _pre_lr(params: optax.Params, traj: jn.ndarray) -> jn.ndarray:
        xs = traj[:,:KBF.Nx]
        us = traj[:,KBF.Nx:]
        zs = KBF.encoder(xs, params)
        fs = vfeat(params, xs, us)
        return zs[1:], fs[:-1]
    v_pre_lr = jax.vmap(_pre_lr, in_axes=(None,0))

    def param_reset(params: optax.Params, batch: jn.ndarray) -> jn.ndarray:
        trajs = jn.vstack(batch)
        zs, fs = v_pre_lr(params, trajs)
        zs = jn.concatenate(zs)
        fs = jn.concatenate(fs)
        _A = jn.linalg.lstsq(fs, zs)[0].T
        params.update(As = _A.reshape(KBF.Nk, KBF.Nk, KBF.Nu+1))
        return params

    return slo_loss, param_reset

def make_flo_loss(KBF: KBF_base):

    def _outer_obj(params: optax.Params, zs: jn.ndarray, xs: jn.ndarray) -> jn.ndarray:
        # Reconstruction
        xd = KBF.decoder(KBF.encoder(xs, params), params)
        Lr = jn.mean((xd-xs)**2)
        # Encoder
        zp = KBF.encoder(xs, params)
        Le = jn.mean((zp-zs)**2)
        # Decoder
        xp = KBF.decoder(zs, params)
        Ld = jn.mean((xp-xs)**2)

        L  = Lr+Ld+Le
        return L, jn.array([Lr, Ld, Le])
    _outer_obj_dz = jax.value_and_grad(_outer_obj, argnums=1, has_aux=True)

    def _lagrange(
        params: optax.Params, xs: jn.ndarray, us: jn.ndarray, mu: jn.ndarray, zp: jn.ndarray) -> jn.ndarray:
        J, _ = _outer_obj(params, zp, xs)
        g1 = jn.dot(mu, KBF.constraint_full(us, zp, xs[0], params))
        return J + g1
    _lagrange_dp  = jax.grad(_lagrange, argnums=0)

    def flo_loss_single(params: optax.Params, data: jn.ndarray) -> jn.ndarray:
        xs = data[:,:KBF.Nx]
        us = data[:,KBF.Nx:]
        zs = KBF.encoder(xs, params)

        # Inner solve
        zp, xp = KBF.predict(xs[0], us, params)

        # Adjoint
        (L, Ls), dLdz = _outer_obj_dz(params, zp, xs)
        mu = KBF.adjoint(-dLdz, us, params).reshape(-1)

        # Sensitivity
        grad = _lagrange_dp(params, xs, us, mu, zp)

        # Losses for monitoring
        Lr, Ld, Le = Ls
        Lz = jn.mean((zp-zs)**2)
        Lx = jn.mean((xp-xs)**2)
        # N  = KBF.constraint_mat_full(us, params)
        # diff = N.T.dot(mu.reshape(-1)) + dLdz.reshape(-1)
        # Lc = jn.max(jn.abs(diff))
        # # Lc = jn.max(jn.abs(KBF.constraint_part(us, zp, params)))  # Always 0
        Lc = -1

        return grad, jn.array([L, Lr, Le, Ld, Lz, Lx, Lc])

    vflo = jax.vmap(flo_loss_single, in_axes=(None,0), out_axes=(0,0))
    def flo_loss(params: optax.Params, batch: jn.ndarray) -> jn.ndarray:
        gs, ls = vflo(params, batch)
        grad = {_k:jn.mean(gs[_k], axis=0) for _k in params.keys()}
        loss = jn.mean(ls, axis=0)
        return grad, dict(L=loss[0], Lr=loss[1], Le=loss[2], Ld=loss[3], Lz=loss[4], Lx=loss[5], Lc=loss[6])

    # ---------------------------------------

    def _feat(params: optax.Params, x: jn.ndarray, u: jn.ndarray) -> jn.ndarray:
        z = KBF.encoder(x, params)
        w = jn.hstack([[1], u]).reshape(-1,1)
        f = (w*z).reshape(-1)
        return f
    vfeat = jax.vmap(_feat, in_axes=(None,0,0))

    def _pre_lr(params: optax.Params, traj: jn.ndarray) -> jn.ndarray:
        xs = traj[:,:KBF.Nx]
        us = traj[:,KBF.Nx:]
        zs = KBF.encoder(xs, params)
        fs = vfeat(params, xs, us)
        return zs[1:], fs[:-1]
    v_pre_lr = jax.vmap(_pre_lr, in_axes=(None,0))

    def param_reset(params: optax.Params, batch: jn.ndarray) -> jn.ndarray:
        trajs = jn.vstack(batch)
        zs, fs = v_pre_lr(params, trajs)
        zs = jn.concatenate(zs)
        fs = jn.concatenate(fs)
        _A = jn.linalg.lstsq(fs, zs)[0].T
        params.update(As = _A.reshape(KBF.Nk, KBF.Nk, KBF.Nu+1))
        if 'de' in params.keys():
            # Linear decoder
            xs = jn.vstack(trajs[:,:,:KBF.Nx])
            zs = KBF.encoder(xs, params)
            _C = jn.linalg.lstsq(zs, xs)[0]
            params.update(de = _C)
        return params

    return flo_loss, param_reset

def fit_model(case, loss, data, numEpoch, maxIter, printInt, reset=None, init_only=True):
    def _fit(params: optax.Params, optimizer: optax.GradientTransformation) -> optax.Params:
        def step(params, opt_state, batch):
            grads, losses = loss(params, batch)
            updates, opt_state = optimizer.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            return params, opt_state, losses

        _, losses = loss(params, data[0])
        keys = losses.keys()

        if reset is not None:
            if init_only:
                params = reset(params, data)
        opt_state = optimizer.init(params)

        Ndata = len(data)
        best, opti = 1e100, None
        hist = {_k:[] for _k in keys}
        flag = 1
        for j in range(numEpoch):
            print(f"Epoch {j}")
            _D = np.concatenate(data)
            np.random.shuffle(_D)
            _D = _D.reshape(data.shape)

            if reset is not None:
                if not init_only:
                    params = reset(params, data)
                    opt_state = optimizer.init(params)

            tmp = {_k:[] for _k in keys}
            for i, batch in enumerate(_D):
                params, opt_state, losses = step(params, opt_state, batch)
                for _k in keys:
                    tmp[_k].append(losses[_k])
                if losses['L'] < best:
                    best = losses['L']
                    opti = copy.deepcopy(params)
                if i % printInt == 0:
                    ss = f"    step {i}"
                    for _k in keys:
                        ss += f", {_k}:{losses[_k]:5.4e}"
                    print(ss)

                if maxIter > 0:
                    if Ndata*j + i > maxIter:
                        flag = -1
                        break

            for _k in keys:
                hist[_k].append(jn.array(tmp[_k]))
            print(f"Saving {case} {j}")
            #np.save(open(f'{case}_mdl.npy', 'wb'), [opti], allow_pickle=True)
            #np.save(open(f'{case}_hst.npy', 'wb'), [hist])

            if flag < 0:
                break

        return opti, hist
    return _fit

def plt_hist(hist, keys, stys, avr=1, lbls=None, fig=None):
    if keys == 'all':
        keys = np.sort([_k for _k in hist.keys()])
    Nk = len(keys)
    if fig is None:
        fig = plt.figure()
    else:
        plt.figure(fig.number)
    for _k in range(Nk):
        if lbls is None:
            lbl = keys[_k]
        else:
            lbl = lbls[_k]
        fig = _pltSegments(fig, hist[keys[_k]], stys[_k], lbl, avr)
    plt.legend()
    return fig

def sample_unif_2d(xrng, Nx):
    x01 = np.linspace(xrng[0][0], xrng[0][1], Nx[0])
    x02 = np.linspace(xrng[1][0], xrng[1][1], Nx[1])
    X1, X2 = np.meshgrid(x01, x02)
    x0s = np.vstack([X1.reshape(-1), X2.reshape(-1)]).T
    return x0s

def gen_data_c(solver, ts, x0s, horizon, shift, batch):
    dat = []
    for _x0 in x0s:
        sol = solver(ts, _x0)
        N = (len(sol) - (horizon-shift)) // shift
        tmp = []
        for _i in range(N):
            tmp.append(sol[_i*shift:_i*shift+horizon])
        dat.append(np.array(tmp))
    dat = np.concatenate(dat)
    Nd  = len(dat)
    Nb  = Nd//batch
    dat = dat[:Nb*batch].reshape(Nb, batch, horizon, -1)
    return dat

def normalize(dat, Nx=2, mode='scl'):
    if mode == '-11':
        mx = np.max(dat, axis=(0,1,2))
        mn = np.min(dat, axis=(0,1,2))
        off = (mx[:Nx]+mn[:Nx])/2
        scl = (mx[:Nx]-mn[:Nx])/2
    elif mode == 'scl':
        mx = np.max(dat, axis=(0,1,2))
        mn = np.min(dat, axis=(0,1,2))
        scl = (mx[:Nx]-mn[:Nx])/2
        off = np.zeros_like(scl)
    elif mode == 'none':
        scl = np.ones(Nx,)
        off = np.zeros(Nx,)
    dat[:,:,:,:Nx] = (dat[:,:,:,:Nx] - off) / scl
    return dat, scl, off

def plt_data(dat):
    f = plt.figure()
    for _t in dat:
        for _b in _t:
            x = _b[:,:2]
            plt.plot(x[:,0], x[:,1])
    return f

def _pltSegments(fig, segs, sty, lbl, avr):
    _l = np.hstack(segs)
    _N = len(_l)
    _n = np.arange(_N)
    if avr > 1:
        _M = (_N//avr)*avr
        _l = _l[:_M].reshape(-1,avr).mean(axis=1)
        _n = _n[:_M:avr]
    plt.semilogy(_n, _l, sty, label=lbl)
    return fig



