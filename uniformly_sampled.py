from dataclasses import dataclass
from typechecking import *
from scipy.interpolate import interp1d
import numpy as np

# ----------------------------------


def snap_to_uniform_sampling(fs: float, tm: Vec[float], tol=1e-6) -> NpVec[float]:
    """ Adjust time values [tm] to sampling rate [fs], don't allow adjustments greater than [tol] / [fs] """
    tm = np.array(tm)
    dt = 1 / fs
    result = np.round(tm / dt) * dt
    assert np.max(np.abs(tm - result)) < dt * tol
    return result


def make_time_bins(fs: float, tlims: Pair[float], tol: float = 1e-6) -> NpVec[float]:
    """ Make time-bin edges with sampling rate [fs] and time limits [tlims] """
    tlims = snap_to_uniform_sampling(fs, tlims, tol)
    nbins = int(.5 + np.diff(tlims) * fs)
    t0 = tlims[0]
    dt = 1 / fs
    bin_edges = t0 + np.arange(nbins + 1) * dt
    bin_edges[0] -= .5 * tol * dt
    bin_edges[-1] += .5 * tol * dt
    return bin_edges


# ----------------------------------


@dataclass
class UniformlySampled:
    """
    Container for uniformly sampled data
    """

    def __init__(self, fs: float, t0: float, *args, **kwargs):
        """
        UniformlySampled(fs: float, t0: float, d: [str, NpVec[float]])
        UniformlySampled(fs: float, t0: float, keys: NpVec[str], vals: NpMat[float])
        Args:
            fs: Sampling rate [Hz]
            t0: Time offset

            d: dict of variables (sampled @ fs)

            vals: 2d numpy array such that vals[i, :] is the i-th variable's values
            keys: keys[i] is the name of the i-th variable
        """

        if len(args) == 0:
            assert set(kwargs.keys()) == {"keys", "vals"}
            keys = kwargs['keys']
            vals = kwargs['vals']
        elif len(args) == 1:
            assert len(kwargs) == 0
            vals = np.concatenate([v[:, None] if v.ndim == 1 else v for v in args[0].values()], axis=1).T
            keys = np.concatenate([np.tile(k, 1 if v.ndim == 1 else v.shape[1]) for k, v in args[0].items()])
        else:
            raise ValueError

        assert len(keys) == vals.shape[0]
        t0 = snap_to_uniform_sampling(fs, t0)
        self._fs = fs
        self._t = t0 + np.arange(vals.shape[1]) / self._fs
        self._keys = np.array(keys)
        self._vals = vals

    def Resample(self, fs: float, tlims: Pair[float]) -> Self:
        """
        Creates a new object by cropping and resampling
        Args:
            fs: new sampling rate
            tlims: new time limits
        """
        if self.fs < fs:
            raise ValueError("Only downsampling is currently supported")
        ifm, ito = self.index(tlims)
        t_curr = self.t[ifm: ito]
        t = np.arange(np.ceil(t_curr[0] * fs), np.floor(t_curr[-1] * fs) + 1) / fs
        vals = interp1d(t_curr, self._vals[:, ifm: ito], kind="cubic", axis=1)(t)
        return UniformlySampled(fs=fs, t0=t[0], keys=self._keys, vals=vals)

    def index(self, tms: NpVec[float]) -> NpVec[int]:
        """ Index of time values """
        return np.floor((tms - self.t[0]) * self.fs).astype(int)

    def _as_array(self) -> NpMat[float]:
        """ Returns values array arr, such that arr[i,:] is the i-th variable """
        return self._vals

    def _set_array(self, arr: NpMat[float]):
        """ Sets values array, with array of the same size """
        assert arr.shape == self._vals.shape
        self._vals = arr

    @property
    def t(self) -> NpVec[float]:
        return self._t

    @property
    def fs(self) -> float:
        return self._fs

    @property
    def num_samples(self) -> int:
        return len(self.t)

    def __getitem__(self, item: str) -> NDArray:
        return self._vals[self._keys == item, :].T.squeeze()

    def __getattr__(self, item):
        return self._vals[self._keys == item, :].T.squeeze()

