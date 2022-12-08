import numpy as np


class Agent:
    def __init__(self, batch=1, dim=1):
        self.batch = batch
        self.dim = dim
        self.act_space = dim

    def act(self):
        pass

    def no_act(self):
        return np.zeros((self.batch, self.dim), dtype=np.float32)

class ConstAgent(Agent):
    def __init__(self, batch=1, dim=1, mu=0.):
        super(ConstAgent, self).__init__(batch=batch, dim=dim)
        self.mu = mu

    def act(self):
        return self.mu * np.ones((self.batch, self.dim), dtype=np.float32)


class JoyStickAgent(Agent):
    def __init__(self, batch=1, dim=1, mu=0.):
        super(JoyStickAgent, self).__init__(batch=batch, dim=dim)
        self.mu = mu

    def act(self, cmd):
        v = np.zeros((self.batch, self.dim), dtype=np.float32)
        if cmd == ord('a'):
            v = -self.mu * np.ones((self.batch, self.dim), dtype=np.float32)
        elif cmd == ord('d'):
            v = self.mu * np.ones((self.batch, self.dim), dtype=np.float32)
        else:
            pass
        return v


class GaussMomentumAgent(Agent):
    def __init__(self, batch=1, dim=1, sigma=0.03, momentum=0.8):
        super(GaussMomentumAgent, self).__init__(batch=batch, dim=dim)
        self.sigma = sigma
        self.momentum = momentum
        self.v = np.zeros((self.batch, self.dim), dtype=np.float32)

    def act(self):
        x = np.random.normal(size=(self.batch, self.dim), loc=0, scale=1.).astype(np.float32)
        self.v = self.sigma * x + self.momentum * self.v
        return self.v.copy()


class RayleighAgent2D(Agent):
    def __init__(self, batch, mu=0.13*2*np.pi, sigma=5.76*2):
        super(RayleighAgent2D, self).__init__(batch=batch, dim=2)
        self.mu = mu  # forward velocity rayleigh dist scale (m/sec)
        self.sigma = sigma  # std rotation velocity (rads/sec)

    def act(self):
        v = np.random.rayleigh(self.mu, self.batch)
        omega = np.random.normal(0., self.sigma, self.batch)
        return np.stack((v, omega), axis=1).astype(np.float32)


class JoyStickAgent2D(Agent):
    def __init__(self, batch=1, mu=0.13*2*np.pi, sigma=5.76 * 2):
        super(JoyStickAgent2D, self).__init__(batch=batch, dim=2)
        self.mu = mu  # forward velocity rayleigh dist scale (m/sec)
        self.sigma = sigma  # std rotation velocity (rads/sec)

    def act(self, cmd):
        v = np.zeros(self.batch, dtype=np.float32)
        omega = np.zeros(self.batch, dtype=np.float32)
        if cmd == ord('w'):
            v = self.mu * np.ones(self.batch, dtype=np.float32)
        elif cmd == ord('s'):
            v = -self.mu * np.ones(self.batch, dtype=np.float32)
        elif cmd == ord('a'):
            omega = -self.sigma * np.ones(self.batch, dtype=np.float32)
        elif cmd == ord('d'):
            omega = self.sigma * np.ones(self.batch, dtype=np.float32)
        else:
            pass
        return np.stack((v, omega), axis=1).astype(np.float32)
