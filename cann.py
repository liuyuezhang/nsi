from models.rnn import Zhang
import utils.functions as f

import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
from models.runner import test
from utils.visualize import calc_ratemap, plot_ratemap


dim = 1
B = 100
T = 1000
joystick = False
visualize = True
# env and agent
if dim == 1:
    from envs.env import Env
    env = Env(batch=B, dim=dim, d=2 * np.pi, periodic_boundary=True, random_init=False)
    from envs.agent import GaussMomentumAgent

    # agent = ConstAgent(batch=B, dim=dim, mu=1.)
    agent = GaussMomentumAgent(batch=B, dim=dim, sigma=1.2, momentum=0.8)
elif dim == 2:
    from envs.arena import ArenaEnv2D
    env = ArenaEnv2D(batch=B, random_init=False)
    if joystick is False:
        from envs.agent import RayleighAgent2D
        agent = RayleighAgent2D(batch=B, mu=0.2, sigma=3.)
    else:
        from cns.envs.agent import JoyStickAgent2D
        agent = JoyStickAgent2D(batch=B)

# hyperparamters
if dim == 1:
    norm = False
    alpha = 1.0  # dt / tau
    n = 100
    N = n**dim

    a = 0.02
    b = None

    D, Dx = f.get_1D(n, require_Dx=True)
    c = 0.1
    k = 0.06
    h = -0.03
    W = c * np.cos(k * D) + h
    Wx = -c * np.sin(k * D) * k * Dx
    Wv = a * np.stack((Wx,), axis=0)
elif dim == 2:
    norm = False
    alpha = 0.2
    n = 32
    N = n**dim

    A = 5.
    lbd = 13.
    abar = 1.05
    beta = 3. / lbd**2
    a = 1.0
    b = 1.0 * np.ones(N)

    D, Dx, Dy = f.get_2D(n, require_Dx=True)
    W = f.dog(D, A, abar * beta, A, beta)
    Wx = f.dog_grad(D, Dx, A, abar * beta, A, beta)
    Wy = f.dog_grad(D, Dy, A, abar * beta, A, beta)
    Wv = a * np.stack((Wx, Wy), axis=0)

# model
model = Zhang(in_size=dim, hid_size=N, nonlinear='relu-tanh', alpha=alpha, bias=True, norm=norm)
model.set_weights(W=W, b=b, W_in=Wv)
model.to_gpu()

# init: you have to control the initial conditions across batches
np.random.seed(0)
u = np.tile(np.random.normal(loc=0., scale=1. / np.sqrt(N), size=N), (B, 1)).astype(np.float32)
u = cp.asarray(u)

# test
rs, xs, vs, x0 = test(env=env, agent=agent, model=model, u=u, T=T, T_init=500, visualize=visualize, joystick=joystick)
print(rs.shape, xs.shape)
np.savez('./data/dim={}_{}_B={}_T={}_N={}_noise=0.1'.format(dim, norm, B, T, N), rs=rs, xs=xs, vs=vs, x0=x0)

# ratemap
if dim == 1:
    vxs = np.concatenate((vs, xs), axis=-1)
    activations = calc_ratemap(r=rs, x=vxs, coords=((-5, 5), (0, 2 * np.pi)), resolutions=(4, 10))
elif dim == 2:
    activations = calc_ratemap(r=rs[:, :, 0:32], x=xs, coords=((-1.1, 1.1), (-1.1, 1.1)), resolutions=(10, 10))
rm_fig = plot_ratemap(activations, n_plots=len(activations), width=10)
plt.savefig("{}.png".format(dim))
