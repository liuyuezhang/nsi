import numpy as np
import cupy as cp
import chainer
import cv2
from utils.graphics import data2img, render_env, render_single
import matplotlib.pyplot as plt
from tqdm import trange


def test(env, model, agent, u=None, T=1000, T_init=0, joystick=False, visualize=False):
    dim = env.dim
    rs, xs, vs = [], [], []
    x0 = env.reset()
    model.init(u if u is not None else cp.asarray(x0))
    # run model (init 800 steps)
    for t in trange(T_init + T):
        # action
        if joystick:
            action = agent.act(cv2.waitKey(0) & 0xFF)
            # cv2.waitKey(0)
        else:
            action = agent.act()
        if t < T_init:
            action = agent.no_act()

        # step
        x, v = env.step(action)
        with chainer.no_backprop_mode():
            # return r rather than self.u
            noise = cp.random.normal(0, 1, size=(env.batch, 100))
            r = model.forward(cp.asarray(v), noise=noise, noise_scale=0.1).array
            # r = model.forward(cp.asarray(v)).array

        # log
        rs.append(cp.asnumpy(r))
        xs.append(x)  # no init q
        vs.append(v)

        # plots
        if t % 10 == 1 and visualize:
            if dim == 1:
                # visualize
                neural_img = render_single(cp.asnumpy(r[0, :]), bottom=0., top=None)
                cv2.imshow("neural", neural_img)
                cv2.waitKey(0)

                # display env
                env_img = render_single(np.array(xs)[:, 0, :])
                cv2.imshow("env", env_img)
            elif dim == 2:
                # visualize
                ng = int(np.sqrt(r.shape[-1]))
                neural_img = data2img(cp.asnumpy(r[0, :]).reshape(ng, ng), size=256)
                cv2.imshow("neural", neural_img)
                cv2.waitKey(0)

                # display env
                fig, ax = env.plot()
                env_img = render_env(ax, np.array(xs)[:, 0, :], x_preds=None, rs=rs)
                plt.close(fig)
                cv2.imshow('env', env_img)

    rs = np.array(rs).swapaxes(0, 1)[:, T_init:,:]  # shape: (B, T, Ng)
    xs = np.array(xs).swapaxes(0, 1)[:, T_init:,:]  # shape: (B, T, d)
    vs = np.array(vs).swapaxes(0, 1)[:, T_init:,:]  # shape: (B, T, d)
    return rs, xs, vs, x0
