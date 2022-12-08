import numpy as np
import cupy as cp


class Bunch(object):
    def __init__(self, adict):
        self.__dict__.update(adict)


def boolean(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'


def relu(x):
    return x * (x > 0)


def normalize_variable(x, eps=1e-8):
    return x / (x.item() + eps)


def l2_norm(x, axis=-1):
    return cp.sqrt(cp.sum(x ** 2, axis=axis, keepdims=True))


def distance(x, y, d=None):
    """
    Calculate distances on each dimensions
    """
    xp = cp.get_array_module(x)
    abs_val = xp.abs(x - y)
    if d is None:
        res = abs_val
    else:
        abs_val = abs_val % d
        res = xp.minimum(abs_val, d - abs_val)
    return res.astype(xp.float32)


def l2_distance(x, y, d=None):
    """
    :param d: if None--line, equiv to perimeter is np.inf
    :return: l2 distance based on distances on each dimensions
    """
    # d
    xp = cp.get_array_module(x)
    s = distance(x, y, d=d)
    res = xp.sqrt(xp.sum(s**2, -1))
    return res.astype(xp.float32)


def subtract(x, y, d=None):
    """
    subtraction on a ring is adding its inverse element (group theory)
    it has nothing to do with the absolute value, since everything is ``relative'' quantity in convolution
    """
    xp = cp.get_array_module(x)
    return x - y if d is None else xp.mod(x + (d/2 - y), d) - d/2


def get_1D(n=64, a=None, b=None, d=None, require_Dx=False):
    """
    Define a ring on [a, b] of length d with n points
    Calculate the relative x coordinates matrix Dx and the relative distance matrix D
    """
    # Burak's setting
    if a is None and b is None and d is None:
        a, b, d = -n / 2, n / 2 - 1, n
    x = np.linspace(a, b, n)
    x = np.expand_dims(x, -1)
    Dx = subtract(x, x.T, d=d)
    D = distance(x, x.T, d=d)
    return D if not require_Dx else (D, Dx)


def get_2D(n=64, a=None, b=None, d=None, require_Dx=False):
    """
    Define a ring on (-ng/2, ..., 0, ...,  ng/2-1) with length ng
    Calculate the relative x, y coordinates matrix Dx, Dy, and the relative distance matrix D
    """
    # Burak's setting
    if a is None and b is None and d is None:
        a, b, d = -n / 2, n / 2 - 1, n
    x = np.linspace(a, b, n)
    y = np.linspace(a, b, n)
    x, y = np.meshgrid(x, y, indexing='xy')  # or 'ij'?
    x = x.reshape(n * n, -1)
    y = y.reshape(n * n, -1)
    Dx = subtract(x, x.T, d=d)
    Dy = subtract(y, y.T, d=d)

    dx = distance(x, x.T, d=d)
    dy = distance(y, y.T, d=d)
    D = np.sqrt(dx ** 2 + dy ** 2)
    return D if not require_Dx else (D, Dx, Dy)


def gaussian(d, a, s):
    xp = cp.get_array_module(d)
    return a * xp.exp(-d**2 / (2*s**2)).astype(np.float32)


# dog in the Burak & Fiete style
def dog(d, a1=1., s1=1.05 * 3. / (13.**2), a2=1., s2=1. * 3. / (13.**2)):
    xp = cp.get_array_module(d)
    return a1 * xp.exp(-s1 * d**2) - a2 * xp.exp(-s2 * d**2)


def dog_grad(d, x, a1=1., s1=1.05 * 3. / (13.**2), a2=1., s2=1. * 3. / (13.**2)):
    xp = cp.get_array_module(d)
    return (a1 * s1 * xp.exp(-s1 * d**2) - a2 * s2 * xp.exp(-s2 * d**2)) * (-2 * x)


def square(d, c, a, b):
    res = np.zeros(d.shape, dtype=np.float32)
    res[(d >= a) & (d <= b)] = c
    return res


# def square_grad(d, c, a, b):
#     res = np.zeros(d.shape, dtype=np.float32)
#     d >= a


def dos(d, c1=-0.05, a1=25., b1=50., c2=0.1, a2=0., b2=5.):
    return square(d, c=c1, a=a1, b=b1) + square(d, c=c2, a=a2, b=b2)


# def dos_grad(d, x, c1=-0.05, a1=25., b1=50., c2=0.1, a2=0., b2=5.):



# def dog(d, a1=1., s1=1.05 * 3. / (13.**2), a2=1., s2=1. * 3. / (13.**2)):
#     return gaussian(d, a1, s1) - gaussian(d, a2, s2)
#
#
# def dog_grad(d, x, a1=1., s1=1.05 * 3. / (13.**2), a2=1., s2=1. * 3. / (13.**2)):
#     xp = cp.get_array_module(d)
#     return (a1 * s1 * xp.exp(-s1 * d**2) - a2 * s2 * xp.exp(-s2 * d**2)) * (-2 * x)
#
#
def tri2rad(x):
    cos, sin = x[..., 0], x[..., 1]
    res = np.arctan2(sin, cos)
    return res % (2 * np.pi)
