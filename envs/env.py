import numpy as np


class Env:
    def __init__(self, batch=1, dim=1, d=2*np.pi, periodic_boundary=False, random_init=False, dt=0.025):
        self.batch = batch
        self.dim = dim
        self.reset_space = dim
        self.ob_space = dim
        self.d = d
        self.periodic_boundary = periodic_boundary
        self.random_init = random_init

        self.dt = dt
        self.x = np.random.rand(self.batch, self.dim).astype(np.float32) * self.d if self.random_init \
            else np.ones((self.batch, self.dim)).astype(np.float32) * self.d / 2

    def reset(self, x=None):
        if x is None:
            self.x = np.random.rand(self.batch, self.dim).astype(np.float32) * self.d if self.random_init \
                else np.ones((self.batch, self.dim)).astype(np.float32) * self.d / 2
        else:
            self.x = x.astype(np.float32)
        if self.periodic_boundary:
            self.x = self.x % self.d
        return self.x

    def step(self, v=None):
        if v is None:
            v = np.zeros((self.batch, self.dim), dtype=np.float32)
        self.x += v * self.dt
        self.x = self.x % self.d if self.periodic_boundary else np.clip(self.x, 0, self.d)
        # todo: in some env, the planned and exerted action are not the same (obstacle)
        return self.x, v

    def get_state(self):
        return self.x.copy()
