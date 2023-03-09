"""
Motor and kinematic data processing + containers
"""

import numpy as np
from typechecking import *
from scipy.interpolate import interp1d
from uniformly_sampled import UniformlySampled
from geometrik.geometrik.spcurve_factory import make_ndspline
from geometrik.geometrik.invariants import geometric_invariants

# ----------------------------------


class KinData(UniformlySampled):
    """ uniformly sampled kinematic data """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


# ----------------------------------


def numdrv(X: np.ndarray, t: np.ndarray, n=1, ret_all=False):
    """
    Simple numeric derivative(s) of X wrt t.
    Args:
        X: 2d np array, points x dims
        t: 1d np array, same length as X.
        n: order of derivative.
        ret_all: return all derivatives up to n; 1,2,..n.
    Returns:
        if ret_all is False: n-th order derivative as np array, same size as X.
        if ret_all is True: a list of np arrays, same size as X: [d/dt X, (d/dt)^2 X, .., (d/dt)^n X],
    """

    shape = X.shape
    if X.ndim == 1:
        X = X[:, None]

    all_drvs = []
    for _ in range(n):
        X = np.copy(X)
        for j in range(X.shape[1]):
            X[:, j] = np.gradient(X[:, j], t, edge_order=1)
        if ret_all:
            all_drvs.append(X)

    if ret_all:
        return [X.reshape(shape) for X in all_drvs]
    return X.reshape(shape)


def calc_kinematics(X: NpNx2[float], t: NpVec[float], dst_t: NpVec[float], dx: float = .5):

    def _softpositive(x):
        d = -x
        d[x > 0] = 0
        d = np.interp(np.arange(len(d)), np.nonzero(d)[0], d[d != 0])
        return x + d

    spl = make_ndspline(X=X, t=t, dx=dx, default_t=None, stol=.1)
    geom_invars = geometric_invariants(spl)
    vel = spl(der=1)
    acc = spl(der=2)

    kin = {'X': spl(),
           'velx': vel[:, 0],
           'vely': vel[:, 1],
           'accx': acc[:, 0],
           'accy': acc[:, 1],
           'spd2': np.linalg.norm(vel, axis=1),
           'acc2': np.linalg.norm(acc, axis=1),
           'spd1': numdrv(geom_invars['s1'], spl.t),
           'spd0': numdrv(geom_invars['s0'], spl.t),
           'crv2': geom_invars['k2']
           }

    fs = (len(dst_t) - 1) / (dst_t[-1] - dst_t[0])
    t0 = dst_t[0]
    deviation_from_uniform = np.max(np.abs(dst_t - (t0 + np.arange(len(dst_t)) / fs)))
    max_deviation = .01  # in dt units
    assert deviation_from_uniform * fs < max_deviation

    s = kin['spd2'].copy()
    assert np.all(kin['spd2'] >= 0)

    positives = ['spd0', 'spd1', 'spd2', 'acc2']

    for k, v in kin.items():
        vi = interp1d(spl.t, v, axis=0, kind="cubic")(dst_t)
        if k in positives:
            assert np.mean(vi < 0) < .2
            vi[vi < 0] = 0
        kin[k] = vi

    return KinData(fs, t0, kin)


