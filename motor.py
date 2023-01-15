"""
Motor and kinematic data processing + containers
"""

import numpy as np
from typechecking import *
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d
from uniformly_sampled import UniformlySampled

# ----------------------------------


class KinData(UniformlySampled):
    """ uniformly sampled kinematic data """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

# ----------------------------------


def _cart2polar(v: NpNx2[float]) -> Tuple[NpVec[float], NpVec[float]]:
    v = v - np.mean(v)
    rho = np.linalg.norm(v, axis=1)
    unit_vec = v / rho
    return unit_vec, rho


def derivative(X, t, n=1):
    """
    n-th order derivatives of X wrt t
    Args:
        X: 2d array, NxD, N = points, D = dim
        t: 1d array, parameter to differentiate by.
        n: order of derivative
    Returns:
        [dX/dt, d^2X/dt, .., d^nX/dt]
    """
    """
    n-th order derivatives of X wrt t
    :param X: 2d array, NxD, N = points, D = dim
    :param t: 1d array, parameter to differentiate by.
    :param n: order of derivative
    :return: drvs: list of length n, drvs[i] contains the (i+1)-th order derivative of X
    """
    X = np.expand_dims(X, axis=1) if X.ndim == 1 else X
    drvs = []
    for k in range(n):
        X = np.copy(X)
        for j in range(X.shape[1]):
            X[:, j] = np.gradient(X[:, j], t, edge_order=1)
        drvs.append(X.squeeze())
    return drvs


# ----------------------------------


def calc_kinematics(X: NpNx2[float], t: NpVec[float], smooth_sig: float = .1) -> KinData:
    assert X.ndim == 2
    assert X.shape[1] == 2

    _dbg = False

    fs = (len(t) - 1) / (t[-1] - t[0])
    t_ = np.ceil(t[0] * fs) / fs + np.arange(len(t) - 1) / fs
    X = interp1d(t, X, kind="cubic", axis=0)(t_)
    t = t_
    del t_

    if _dbg:
        Xo = np.copy(X)

    X = gaussian_filter1d(X, sigma=smooth_sig * fs, axis=0)
    vel, acc, jrk = derivative(X, t, n=3)
    kin = {'X': X,
           'velx': vel[:, 0], 'vely': vel[:, 1],
           'accx': acc[:, 0], 'accy': acc[:, 1],
           'spd2': np.linalg.norm(vel, axis=1),
           'acc2': np.linalg.norm(acc, axis=1)}

    return KinData(fs, t[0], kin)


