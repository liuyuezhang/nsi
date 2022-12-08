import chainer
import numpy as np
from ..utils import functions as f
from abc import abstractmethod
from skimage.draw import line


class ObservationWrapper:
    def __init__(self, env, wrap_reset=True, wrap_step=True):
        self.env = env
        self.wrap_reset = wrap_reset
        self.wrap_step = wrap_step
        self.reset_space = env.reset_space
        self.ob_space = env.ob_space

    def reset(self, s0=None):
        return self.ob(self.env.reset(s0)) if self.wrap_reset else self.env.reset(s0)

    def step(self, a=None):
        return self.ob(self.env.step(a)) if self.wrap_step else self.env.step(a)

    def get_state(self):
        return self.env.get_state()

    @abstractmethod
    def ob(self, s):
        return NotImplementedError


class CosSinWrapper(ObservationWrapper):
    def __init__(self, env, wrap_reset, wrap_step):
        super(CosSinWrapper, self).__init__(env, wrap_reset, wrap_step)
        if wrap_reset:
            self.reset_space = 2
        if wrap_step:
            self.ob_space = 2

    def ob(self, s):
        return np.concatenate((np.cos(s), np.sin(s)), axis=-1).astype(np.float32)


class LinearWrapper(ObservationWrapper):
    def __init__(self, env, wrap_reset, wrap_step, in_size=2, out_size=100, seed=0):
        super(LinearWrapper).__init__(env, wrap_reset, wrap_step)
        np.random.seed(seed)
        self.W = np.random.rand(in_size, out_size).astype(np.float32)
        if wrap_reset:
            self.reset_space = out_size
        if wrap_step:
            self.ob_space = out_size

    def ob(self, s):
        return s @ self.W


class PopulationCoding:
    def __init__(self, dim=2, d=1.0, n=20, s=0.1, periodic_boundary=False):
        self.dim = dim
        self.d = 1.0
        self.n = n  # number of neurons on each dimension
        self.s = s

        # receptive field
        self.rf = lambda x: f.gaussian(x, a=1., s=s)
        self.distance = lambda x, y: f.l2_distance(x, y, d=d if periodic_boundary else None)
        # uniform tile
        u = np.meshgrid(*[np.linspace(0, d, n) for _ in range(dim)])
        self.u = np.stack([_.reshape(-1).astype(np.float32) for _ in u], axis=-1)

    def forward(self, s):
        d = self.distance(s[:, None, :], self.u[None, :, :])
        return self.rf(d)


class PopulationCodingWrapper(ObservationWrapper):
    def __init__(self, env, wrap_reset, wrap_step, n=20, s=0.1):
        super().__init__(env, wrap_reset, wrap_step)
        self.pc = PopulationCoding(dim=env.dim, d=env.d, n=n, s=s, periodic_boundary=env.periodic_boundary)
        if wrap_reset:
            self.reset_space = n
        if wrap_step:
            self.ob_space = n

    def ob(self, s):
        return self.pc.foward(s)


class SingleOrientationWrapper(ObservationWrapper):
    def __init__(self, env, wrap_reset, wrap_step, l=10, l1=10, noise_scale=0.0):
        super().__init__(env, wrap_reset, wrap_step)
        self.l = l
        self.size = 2 * l + 1
        self.l1 = l1
        self.noise_scale = noise_scale
        self.idx = np.arange(0, self.l1 + 1)
        if wrap_reset:
            self.reset_space = self.size * self.size
        if wrap_step:
            self.ob_space = self.size * self.size

    def draw(self, theta, x0=0, y0=0, length=1):
        rr, cc = line(int(self.l + x0 - length * np.cos(theta)), int(self.l + y0 - length * np.sin(theta)), 
                      int(self.l + x0 + length * np.cos(theta)), int(self.l + y0 + length * np.sin(theta)))
        return rr, cc


    def ob(self, s):
        canvas = np.zeros((self.env.batch, self.size, self.size))
        for i in range(self.env.batch):
            # line 1
            rr, cc = self.draw(s[i, 0], x0=0, y0=0, length=self.l1)
            canvas[i, rr, cc] = 0.5
        canvas += np.random.uniform(low=-self.noise_scale, high=self.noise_scale, size=(self.env.batch, self.size, self.size))
        return np.clip(canvas.copy(), 0, 1).astype(np.float32)


class DoubleOrientationWrapper(ObservationWrapper):
    def __init__(self, env, wrap_reset, wrap_step, l=10, l1=10, l2=5):
        super().__init__(env, wrap_reset, wrap_step)
        self.l = l
        self.size = 2 * l + 1
        self.l1 = l1
        self.l2 = l2
        self.idx = np.arange(0, self.l1 + 1)
        if wrap_reset:
            self.reset_space = self.size * self.size
        if wrap_step:
            self.ob_space = self.size * self.size

    def draw(self, theta, x0=0, y0=0, length=1):
        rr, cc = line(int(self.l + x0 - length * np.cos(theta)), int(self.l + y0 - length * np.sin(theta)), 
                      int(self.l + x0 + length * np.cos(theta)), int(self.l + y0 + length * np.sin(theta)))
        return rr, cc


    def ob(self, s):
        canvas = np.zeros((self.env.batch, self.size, self.size))
        for i in range(self.env.batch):
            # line 1
            rr, cc = self.draw(s[i, 0], x0=0, y0=0, length=self.l1)
            canvas[i, rr, cc] = 0.5
            # line 2
            rr, cc = self.draw(s[i, 1], x0=0, y0=0, length=self.l2)
            canvas[i, rr, cc] = 0.5
        return np.clip(canvas.copy(), 0, 1).astype(np.float32)


class Gabor(ObservationWrapper):
    pass
