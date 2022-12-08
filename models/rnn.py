import numpy as np
import chainer
import chainer.functions as F
from chainer import Variable, Parameter
from utils.functions import l2_norm


class Elman(chainer.Chain):
    def __init__(self, in_size, hid_size, nonlinear='relu-tanh', alpha=1., norm=False, bias=True):
        super(Elman, self).__init__()
        self.in_size = in_size
        self.hid_size = hid_size
        self.nonlinear = nonlinear
        self.alpha = alpha
        self.norm = norm
        if self.nonlinear == 'relu':
            self.sigma = F.relu
        elif self.nonlinear == 'tanh':
            self.sigma = F.tanh
        elif self.nonlinear == 'relu-tanh':
            self.sigma = lambda x: F.relu(F.tanh(x))

        self.u = None  # (B, N)
        glorot_uniform = chainer.initializers.GlorotUniform()
        with self.init_scope():
            self.W = Parameter(shape=(self.hid_size, self.hid_size), initializer=glorot_uniform)
            self.b = Parameter(shape=self.hid_size, initializer=0., requires_grad=bias)
            self.W_in = Parameter(shape=(self.in_size, self.hid_size), initializer=glorot_uniform)

    def init(self, u):
        self.u = u

    def detach(self):
        self.u = Variable(self.u.array)

    def input(self, x):
        return x @ self.W_in

    def dynamics(self, h):
        return self.sigma(self.u @ self.W + self.b + h)

    def forward(self, x, noise=None, noise_scale=0.0):
        if noise is not None:
            norm_noise = l2_norm(noise, axis=-1)
            self.u += noise_scale / norm_noise * noise
        h = self.input(x)
        r = self.dynamics(h)
        self.u = (1. - self.alpha) * self.u + self.alpha * r
        if self.norm:
            self.u = F.normalize(self.u, axis=-1)
        return r

    def set_weights(self, W=None, b=None, W_in=None):
        if W is not None:
            assert (self.W.array.shape == W.shape)
            self.W.array = W.astype(np.float32)
        if b is not None:
            assert (self.b.array.shape == b.shape)
            self.b.array = b.astype(np.float32)
        if W_in is not None:
            assert (self.W_in.array.shape == W_in.shape)
            self.W_in.array = W_in.astype(np.float32)


class Zhang(Elman):
    def __init__(self, in_size, hid_size, nonlinear='relu-tanh', alpha=1., bias=True, norm=False):
        super(Zhang, self).__init__(in_size=in_size, hid_size=hid_size, nonlinear=nonlinear, alpha=alpha, bias=bias, norm=norm)
        glorot_uniform = chainer.initializers.GlorotUniform()
        with self.init_scope():
            self.W_in = Parameter(shape=(self.in_size, self.hid_size, self.hid_size), initializer=glorot_uniform)

    def input(self, x):
        return F.einsum('bd,dmn,bn->bm', x, self.W_in, self.u)


class LowRankElman(Elman):
    def __init__(self, in_size, hid_size, p, nonlinear='relu', alpha=1., bias=True, norm=False):
        super(LowRankElman, self).__init__(in_size=in_size, hid_size=hid_size, nonlinear=nonlinear, alpha=alpha, bias=bias, norm=norm)
        self.p = p
        glorot_uniform = chainer.initializers.GlorotUniform()
        with self.init_scope():
            self.W1 = Parameter(shape=(self.hid_size, self.p), initializer=glorot_uniform)
            self.W2 = Parameter(shape=(self.hid_size, self.p), initializer=glorot_uniform)

    def dynamics(self, h):
        return self.sigma(self.u @ self.W1 @ self.W2.T + self.b + h)



