import numpy as np
import cupy as cp
import chainer
import chainer.functions as F
import chainer.links as L
import utils.functions as f
from chainer import Variable, Parameter


class ElmanId(chainer.Chain):
    def __init__(self, N, in_size, idx, feature_size, nonlinear='relu-tanh', alpha=1., norm=False, bias=True,
                 noise_scale=0.0, lr=1e-4, reg='l2', coef_neuron=0., coef_rnn=0.):
        super(ElmanId, self).__init__()
        self.N = N
        self.in_size = in_size
        self.idx_size = len(idx)
        self.idx = idx
        self.feature_size = feature_size

        self.nonlinear = nonlinear
        self.alpha = alpha
        self.norm = norm
        self.noise_scale = noise_scale
        if self.nonlinear == 'relu':
            self.sigma = F.relu
        elif self.nonlinear == 'tanh':
            self.sigma = F.tanh
        elif self.nonlinear == 'relu-tanh':
            self.sigma = lambda x: F.relu(F.tanh(x))

        self.loss_fn = F.mean_squared_error
        self.coef_neuron = coef_neuron
        self.coef_rnn = coef_rnn
        if reg == 'l2':
            self.reg_fn = lambda x: F.mean(x**2)
        elif reg == 'l1':
            self.reg_fn = lambda x: F.mean(F.absolute(x))

        self.u = None  # (B, N)
        glorot_uniform = chainer.initializers.GlorotUniform()
        with self.init_scope():
            self.W = Parameter(shape=(self.N, self.N), initializer=glorot_uniform)
            self.b = Parameter(shape=self.N, initializer=0., requires_grad=bias)
            self.W_in = L.Linear(in_size=self.feature_size, out_size=self.in_size, initialW=glorot_uniform)

        # optimizers
        self.optimizer = chainer.optimizers.Adam(alpha=lr)
        self.optimizer.setup(self)

        # loss
        self.loss = 0.
        self.neuron = 0.
        self.loss_rnn = 0.
        self.T = 0

    def clear_loss(self):
        self.loss = 0.
        self.neuron = 0.
        self.loss_rnn = 0.
        self.T = 0

    def init(self, u):
        self.u = F.concat((u, Variable(cp.zeros((u.shape[0], self.N - u.shape[1]), dtype=np.float32))), axis=1)

    def detach(self):
        self.u = Variable(self.u.array)

    def input(self, x):
        return self.W_in(x)

    def dynamics(self, h):
        h = F.concat((Variable(cp.zeros((h.shape[0], self.N - h.shape[1]), dtype=np.float32)), h), axis=1)
        return self.sigma(self.u @ self.W + self.b + h)

    def forward(self, x, noise=None):
        if noise is not None:
            norm_noise = f.l2_norm(noise, axis=-1)
            self.u += self.noise_scale / norm_noise * noise
        h = self.input(x)
        r = self.dynamics(h)
        self.u = (1. - self.alpha) * self.u + self.alpha * r
        if self.norm:
            self.u = F.normalize(self.u, axis=-1)
        return r

    def step(self, v, x, train=True):
        # step
        with chainer.configuration.using_config("enable_backprop", train):
            # forward
            r = self.forward(v)

            # loss
            self.loss += self.loss_fn(r[:, self.idx], x)
            self.neuron += self.reg_fn(r)
            self.loss_rnn += self.reg_fn(self.W)
            self.T += 1
        return r

    def update(self):
        # main loss
        loss = self.loss / self.T

        # neuron and weight loss
        loss_neuron = self.neuron / self.T
        loss_rnn = self.loss_rnn / self.T

        # loss
        loss_total = loss + self.coef_neuron * loss.item() * f.normalize_variable(loss_neuron) \
                     + self.coef_rnn * loss.item() * f.normalize_variable(loss_rnn)

        # backward and update
        self.cleargrads()
        loss_total.backward()
        self.optimizer.update()

        # detach
        self.detach()
        self.clear_loss()
        return loss_total.item(), loss.item(), loss_neuron.item(), loss_rnn.item()
