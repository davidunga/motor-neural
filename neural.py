"""
Handles processing and storing neural information in spike times and spike count formats
"""

from typechecking import *


class NeuronSpikeTimes(dict[str, NpVec]):
    """ Stores neuron spike times as a dictionary <neuron_name>: <spike times array> """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class PopulationSpikeTimes:
    """ Stores neuron spike times in a flat time-sorted array
        Optimized for population-level analysis
    """

    def __init__(self, tms: Vec[float], ixs: Vec[int], names: Sequence[str], tlims: Pair[float]):
        """
        Args:
            tms: times of spikes
            ixs: index of spiking neuron: ixs[i] is the neuron that spiked at time tms[i]
            names: neurons names: names[ix] is the name of the neuron with index ix
            tlims: [min, max] time limits of the data
        """

        self.tms = np.array(tms)
        self.ixs = np.array(ixs)
        self.names = np.array(names)
        self.num_neurons = len(self.names)
        self._tlims = tlims

        assert max(self.ixs) < self.num_neurons
        assert self.tms.ndim == self.ixs.ndim == 1
        assert len(self.tms) == len(self.ixs)
        assert is_sorted(self.tms)
        assert self.tms[0] >= self._tlims[0] and self.tms[-1] <= self._tlims[-1]

    @classmethod
    def fromNeuronSpikeTimes(cls, neuron_spktimes: NeuronSpikeTimes, tlims: Pair[float] = None) -> Self:
        neuron_ixs = np.concatenate([np.tile(ix, len(tms)) for ix, tms in enumerate(neuron_spktimes.values())])
        spktimes = np.concatenate(list(neuron_spktimes.values()))
        si = np.argsort(spktimes)
        if tlims is None:
            tlims = [spktimes[si[0]] - np.finfo(float).eps, spktimes[si[-1]] + np.finfo(float).eps]
        return cls(tms=spktimes[si], ixs=neuron_ixs[si], names=list(neuron_spktimes.keys()), tlims=tlims)

    @property
    def tlims(self) -> Pair[float]:
        return self._tlims

    def TimeSlice(self, tlims: Pair[float]) -> Self:
        tms, ixs = self.timeslice(tlims)
        return PopulationSpikeTimes(tms, ixs, self.names, tlims)

    def timeslice(self, tlims: Pair[float]) -> Tuple[Vec[float], Vec[int]]:
        ifm, ito = np.searchsorted(self.tms, tlims)
        return self.tms[ifm: ito], self.ixs[ifm: ito]

    def neuron_spktimes(self, tlims: Pair[float] = None) -> NeuronSpikeTimes:
        if tlims is None:
            ifm, ito = 0, len(self.ixs)
        else:
            ifm, ito = np.searchsorted(self.tms, tlims)
        result = NeuronSpikeTimes({name: np.array([], float) for name in self.names})
        for ix, tm in zip(self.ixs[ifm: ito], self.tms[ifm: ito]):
            result[self.names[ix]] = np.append(result[self.names[ix]], tm)
        return result


SpikeTimes = PopulationSpikeTimes | NeuronSpikeTimes


def bin_spikes(spktimes: SpikeTimes, bins: float | Vec[float]) -> NpMat[UInt]:
    """
    Build spike-counts matrix from spike times
    Args:
        spktimes: spike times data
        bins: either: either bin edges, or bin size
    """

    if isinstance(spktimes, NeuronSpikeTimes):
        spktimes = PopulationSpikeTimes.fromNeuronSpikeTimes(spktimes)

    if isinstance(bins, float):
        bin_start = np.ceil(spktimes.tlims[0] / bins)
        bin_stop = np.floor(spktimes.tlims[-1] / bins)
        t = np.arange(bin_start, bin_stop) * bins
        assert t[0] >= spktimes.tlims[0]
        assert t[-1] <= spktimes.tlims[-1]
        bin_edges = np.zeros(len(t) + 1, float)
        bin_edges[:-1] = t - .5 * bins
        bin_edges[-1] = t[-1] + .5 * bins
    else:
        bin_edges = bins

    spike_tms, neuron_ixs = spktimes.timeslice([bin_edges[0], bin_edges[-1]])

    bin_ixs = np.digitize(spike_tms, bin_edges) - 1
    result = np.zeros((spktimes.num_neurons, max([0, len(bin_edges) - 1])), UInt)
    for bin_ix, neurons_ix in zip(bin_ixs, neuron_ixs):
        result[neurons_ix, bin_ix] += 1

    return result, bin_edges


class NeuralData:
    """ Container for neural data """

    def __init__(self, spktimes: SpikeTimes, neuron_sites: dict[str, str]):
        self.neuron_sites = neuron_sites
        if isinstance(spktimes, PopulationSpikeTimes):
            self.population_spktimes = spktimes
        else:
            self.population_spktimes = PopulationSpikeTimes.fromNeuronSpikeTimes(spktimes)
        assert set(self.neuron_sites.keys()) == set(self.population_spktimes.names)

        self._bin_sz = None
        self._t = None
        self._spktimes = None
        self._spkcounts = None

    def _check_spkcounts(self):
        if self._spkcounts is None:
            raise AttributeError("Spike counts data required.")

    def TimeSlice(self, tlims: Pair[float]) -> Self:
        """ Get a time-sliced copy of this object """
        result = NeuralData(self.population_spktimes.TimeSlice(tlims), self.neuron_sites)
        result._spkcounts = self._spkcounts
        if self._t is not None:
            ifm, ito = self.index(tlims)
            result._t = self._t[ifm: ito]
            result._spkcounts = self._spkcounts[:, ifm: ito]
        return result

    def make_spkcounts_(self, bin_sz: float) -> None:
        self._spkcounts, bin_edges = bin_spikes(self.population_spktimes, bin_sz)
        assert np.allclose(bin_sz, np.diff(bin_edges))
        self._bin_sz = bin_sz
        self._t = .5 * (bin_edges[1:] + bin_edges[:-1])
        assert np.allclose(np.diff(self._t), bin_sz)

    @property
    def t(self) -> NpVec[float]:
        self._check_spkcounts()
        return self._t

    def index_slice(self, tlims: Pair[float]):
        """ Convert time range to bin index slice object """
        self._check_spkcounts()
        bin_start = self.index(tlims[0])
        if tlims[-1] == self.t[-1]:
            bin_stop = self.num_samples
        else:
            dur = np.diff(tlims)
            bin_span = int(np.ceil((dur / self._bin_sz)))
            if np.mod(dur, self._bin_sz) < 1e-6:
                # avoid being too aggressive with the ceil()..
                bin_span -= 1
            bin_stop = bin_start + bin_span
        return slice(bin_start, bin_stop)

    def index(self, tms: Vec[float]) -> NpVec[int]:
        """ Convert time to bin index """
        self._check_spkcounts()
        if np.min(tms) < self._t[0] or np.max(tms) > self._t[-1]:
            ValueError("Time value out of range")
        return np.digitize(tms, self._t) - 1

    @property
    def num_neurons(self) -> int:
        return self.population_spktimes.num_neurons

    @property
    def num_samples(self) -> int:
        if self.t is None:
            return 0
        return len(self.t)

    @property
    def sites(self) -> Set[str]:
        """ Set of all brain sites in data """
        return set(self.neuron_sites.values())

    @property
    def spktimes(self) -> NeuronSpikeTimes:
        if self._spktimes is None:
            self._spktimes = self.population_spktimes.neuron_spktimes()
        return self._spktimes

    def get_spkcounts(self, tlims: Pair[float] = None) -> NpMat[float]:
        self._check_spkcounts()
        if tlims is None:
            tlims = [self.t[0], self.t[-1]]
        bin_slice = self.index_slice(tlims)
        return self._spkcounts[:, bin_slice], self.t[bin_slice], bin_slice

    def set_spkcounts(self, spkcounts: NpMat[float]):
        assert self._spkcounts.shape == spkcounts.shape
        self._spkcounts = spkcounts
