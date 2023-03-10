"""
Motor and kinematic data processing + containers
"""

import numpy as np
from motorneural.typechecking import *
from scipy.interpolate import interp1d
from motorneural.uniformly_sampled import UniformlySampled

# ----------------------------------


class KinData(UniformlySampled):
    """ uniformly sampled kinematic data """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


# ----------------------------------


def numdrv(X: np.ndarray, t: np.ndarray, n=1):
    """
    Simple numeric derivative(s) of X wrt t.
    Args:
        X: 2d np array, points x dims
        t: 1d np array, same length as X.
        n: order of derivative.
    Returns:
        a list of np arrays, same size as X: [d/dt X, (d/dt)^2 X, .., (d/dt)^n X],
    """

    shape = X.shape
    if X.ndim == 1:
        X = X[:, None]

    drvs = []
    for _ in range(n):
        X = np.copy(X)
        for j in range(X.shape[1]):
            X[:, j] = np.gradient(X[:, j], t, edge_order=1)
        drvs.append(X.reshape(shape))
    return drvs


def basic_kinematics(X: NpNx2[float], t: NpVec[float], dst_t: NpVec[float], dx: float = .5):

    vel, acc, jrk = numdrv(X, t, n=3, ret_all=True)

    kin = {'X': X,
           'velx': vel[:, 0],
           'vely': vel[:, 1],
           'accx': acc[:, 0],
           'accy': acc[:, 1],
           'spd2': np.linalg.norm(vel, axis=1),
           'acc2': np.linalg.norm(acc, axis=1),
           }

    fs = (len(dst_t) - 1) / (dst_t[-1] - dst_t[0])
    t0 = dst_t[0]
    deviation_from_uniform = np.max(np.abs(dst_t - (t0 + np.arange(len(dst_t)) / fs)))
    max_deviation = .01  # in dt units
    assert deviation_from_uniform * fs < max_deviation

    kin = {k: interp1d(t, v, axis=0, kind="cubic")(dst_t) for k, v in kin.items()}
    return KinData(fs, t0, kin)


