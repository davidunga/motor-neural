"""
Motor and kinematic data processing + containers
"""

import numpy as np
from dataclasses import dataclass
from typechecking import *
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d


@dataclass
class KinData:

    _t: NpVec[float]
    _d: dict[str, NpVec[float]]
    _dt = None

    def __post_init__(self):
        self._dt = np.diff(self.t)
        assert np.all(self._dt > 0)
        assert np.min(self._dt) > .999 * np.max(self._dt), "Time samples not sufficiently uniform"
        assert all([len(v) == len(self.t) for v in self._d.values()])

    @property
    def t(self) -> NpVec[float]:
        return self._t

    def index(self, tms: Vec[float]) -> NpVec[int]:
        edges = np.linspace(self.t[0] - .5 * self._dt, self.t[-1] + .5 * self._dt, self.num_samples + 1)
        return np.searchsorted(edges, tms)

    @property
    def num_samples(self):
        return len(self.t)

    def TimeSlice(self, tlims: Pair[float]) -> Self:
        ifm, ito = np.searchsorted(self.t, tlims)
        ifm -= 1
        return KinData(self.t[ifm: ito], {k: v[ifm: ito] for k, v in self._d.items()})

    def resample_(self, t: NpVec[float]) -> None:
        assert is_sorted(t), "time parameter must be monotonically increasing"
        assert t[0] >= self.t[0] and t[-1] <= self.t[-1],\
            f"New time range ({t[0]} - {t[-1]}) exceeds current range ({self.t[0]} - {self.t[-1]})"
        assert len(t) <= len(self.t), "up-sampling is not supported"
        for k, v in self._d.items():
            self._d[k] = interp1d(self.t, v, kind="cubic", axis=0)(t)
        self._t = t

    def __getitem__(self, item):
        return self._d[item]

    def __getattr__(self, item):
        return self._d[item]


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
    t_ = t[0] + np.arange(len(t)) / fs
    X = interp1d(t, X, kind="cubic", axis=0)(t_)
    t = t_
    del t_

    if _dbg:
        Xo = np.copy(X)

    X = gaussian_filter1d(X, sigma=smooth_sig * fs, axis=0)
    vel, acc, jrk = derivative(X, t, n=3)
    return KinData(t, {
                'X': X,
                'velx': vel[:, 0],
                'vely': vel[:, 1],
                'accx': acc[:, 0],
                'accy': acc[:, 1],
                'spd2': np.linalg.norm(vel, axis=1),
                'acc2': np.linalg.norm(acc, axis=1)
    })


