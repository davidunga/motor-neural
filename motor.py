"""
Motor and kinematic data processing + containers
"""

import numpy as np
from typechecking import *
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d
from uniformly_sampled import UniformlySampled, UniformTime
from abc import ABC, abstractmethod

# ----------------------------------


class KinData(UniformlySampled):
    """ uniformly sampled kinematic data """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


# ----------------------------------


def cart2polar(v: NpNx2[float], c: Pair[float] = None) -> Tuple[NpVec[float], NpVec[float]]:
    if c is None:
        c = np.mean(v, axis=0)
    v -= c
    rho = np.linalg.norm(v, axis=1)
    uvec = v / rho[: None]
    return uvec, rho


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


def preprocess_traj(X: NpNx2[float], t: NpVec[float], smooth_sig: float = .1) -> Tuple[NpNx2[float], UniformTime]:
    assert X.ndim == 2
    assert X.shape[1] == 2
    ut = UniformTime.from_time(t)
    X = interp1d(t, X, kind="cubic", axis=0)(ut.t)
    if smooth_sig > 0:
        X = gaussian_filter1d(X, sigma=smooth_sig * ut.fs, axis=0)
    return X, ut

# ----------------------------------


class KinFnc(ABC):

    def __init__(self):
        pass

    @abstractmethod
    def __call__(self, X: NpNx2[float], t: NpVec[float]) -> KinData:
        pass


class DefaultKinFnc(KinFnc):

    def __init__(self, smooth_sig: float = .1):
        super().__init__()
        self.smooth_sig = smooth_sig

    def __call__(self, X: NpNx2[float], t: NpVec[float]) -> KinData:
        X, ut = preprocess_traj(X, t, self.smooth_sig)
        vel, acc, jrk = derivative(X, ut.t, n=3)
        kin = {'X': X,
               'velx': vel[:, 0], 'vely': vel[:, 1],
               'accx': acc[:, 0], 'accy': acc[:, 1],
               'spd2': np.linalg.norm(vel, axis=1),
               'acc2': np.linalg.norm(acc, axis=1)}
        return KinData(ut.fs, ut.t0, kin)


