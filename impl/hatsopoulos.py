"""
Interface to Hatsopoulos's 2007 data
Motor-neural (M1/PMd) and kinematic data of 2 macaque monkeys performing
Random Target Pursuit & Center Out tasks.
More details: Hatsopoulos 2007, monkeys RS & RJ.

Overview:

    monkey1 = RockStar = RS:
        Performed TP & 2*COs, recordings only from M1
    monkey2 = Raju = RJ:
        Performed only TP, recordings are from M1 & PMd

    i.e. available data:
        monkey=RS task=TP region=M1     (100 neurons)
        monkey=RS task=CO region=M1     (141 neurons)
        monkey=RS task=CO region=M1     (68 neurons)
        monkey=RJ task=TP region=M1     (54 neurons)
        monkey=RJ task=TP region=PMd    (50 neurons)

    kinematics were sampled at 500 Hz


Structure:
- Fields of raw CenterOut Data:
'ans', 'cpl_0deg', 'cpl_0deg_endmv', 'cpl_0deg_go', 'cpl_0deg_instr',
'cpl_0deg_stmv', 'cpl_135deg', 'cpl_135deg_endmv', 'cpl_135deg_go',
'cpl_135deg_instr', 'cpl_135deg_stmv', 'cpl_180deg', 'cpl_180deg_endmv',
'cpl_180deg_go', 'cpl_180deg_instr', 'cpl_180deg_stmv', 'cpl_225deg',
'cpl_225deg_endmv', 'cpl_225deg_go', 'cpl_225deg_instr', 'cpl_225deg_stmv',
'cpl_270deg', 'cpl_270deg_endmv', 'cpl_270deg_go', 'cpl_270deg_instr',
'cpl_270deg_stmv', 'cpl_315deg', 'cpl_315deg_endmv', 'cpl_315deg_go',
'cpl_315deg_instr', 'cpl_315deg_stmv', 'cpl_45deg', 'cpl_45deg_endmv',
'cpl_45deg_go', 'cpl_45deg_instr', 'cpl_45deg_stmv', 'cpl_90deg',
'cpl_90deg_endmv', 'cpl_90deg_go', 'cpl_90deg_instr',
'cpl_90deg_stmv', 'cpl_st_trial', 'endmv', 'go_cue', 'instruction', 'reward',
'st_trial', 'stmv', 'spikes', 'chans', 'MIchans', 'instr_cell', 'go_cell',
'stmv_cell', 'endmv_cell', 'x', 'y', 'MIchan2rc'

- Fields of raw RTP Data:
'PositionX', 'PositionY', 'endmv', 'reward', 'st_trial', 'target_hit',
'Digital', 'y', 'x', 'spikes', 'chans', 'MIchans', 'PMdchans',
'cpl_st_trial_rew', 'PMdchan2rc', 'MIchan2rc', 'trial', 'monkey',
'force_x', 'force_y', 'shoulder', 'elbow', 't_sh', 't_elb', 'hit_target'

"""

# -----------------------

import numpy as np
from data import Trial, Data, DatasetMeta
from neural import NeuralData, NeuronSpikeTimes
from motor import calc_kinematics
from scipy.io import loadmat
import os
import re


# -----------------------

_DATASETS = {
    "TP_RS": {"file": "rs1050211_clean_spikes_SNRgt4.mat", "task": "TP", "monkey": "RS"},
    "TP_RJ": {"file": "r1031206_PMd_MI_modified_clean_spikesSNRgt4.mat", "task": "TP", "monkey": "RJ"},
    "CO_01": {"file": "rs1050225_clean_SNRgt4.mat", "task": "CO", "monkey": "RS"},
    "CO_02": {"file": "rs1051013_clean_SNRgt4.mat", "task": "CO", "monkey": "RS"}
}

# -----------------------


class HatsoData(Data):

    def __init__(self, trials, meta):
        super().__init__(trials, meta)

    @classmethod
    def Make(cls, data_dir: str, dataset: str, lag: float, bin_sz: float):
        return _load_data(data_dir, dataset, lag, bin_sz)

# -----------------------


def _load_data(data_dir: str, dataset: str, lag: float, bin_sz: float) -> HatsoData:

    assert np.abs(lag) <= 1, f"Extreme lag value: {lag}. Make sure its in seconds."
    assert 0 < bin_sz <= 1, f"Extreme bin size value: {bin_sz}. Make sure its in seconds."

    # ----------
    # helper functions:

    def _get_TP_events(raw):
        """ Get Target Pursuit trial events and properties """
        st = np.real(raw['cpl_st_trial_rew'])[:, 0]
        end = np.real(raw['cpl_st_trial_rew'])[:, 1]
        mv_end = raw['endmv'].flatten()
        mv_end = mv_end[np.searchsorted(mv_end, st[0]):]
        assert len(mv_end) == len(st)
        events = [{"st": st[ix], "end": end[ix], "mv_end": mv_end[ix]} for ix in range(len(st))]
        return events, [{} for _ in range(len(st))]

    def _get_CO_events(raw):
        """ Get Center Out trial events and properties """
        st = np.concatenate([raw[f'cpl_{ang}deg'].flatten() for ang in range(0, 360, 45)])
        si = np.argsort(st)
        st = st[si]
        angs = np.concatenate([np.tile(ang, raw[f'cpl_{ang}deg'].size) for ang in range(0, 360, 45)])[si]
        go = np.concatenate([raw[f'cpl_{ang}deg_go'].flatten() for ang in range(0, 360, 45)])[si]
        instr = np.concatenate([raw[f'cpl_{ang}deg_instr'].flatten() for ang in range(0, 360, 45)])[si]
        mv_st = np.concatenate([raw[f'cpl_{ang}deg_stmv'].flatten() for ang in range(0, 360, 45)])[si]
        mv_end = np.concatenate([raw[f'cpl_{ang}deg_endmv'].flatten() for ang in range(0, 360, 45)])[si]
        end = raw['reward'].flatten()
        events = [{"st": st[ix], "end": end[ix], "instr": instr[ix], "go": go[ix],
                   "mv_st": mv_st[ix], "mv_end": mv_end[ix]} for ix in range(len(st))]
        properties = [{"ang": int(angs[ix])} for ix in range(len(angs))]
        return events, properties

    def _make_neural_data(raw):
        """ Make neural data """
        chans = list(raw['MIchans'].squeeze())
        if 'PMdchans' in raw:
            chans += list(raw['PMdchans'].squeeze())
        neuron_spktimes = NeuronSpikeTimes()
        neuron_sites = {}
        for k in sorted(list(k for k in raw.keys() if k.startswith('Chan'))):
            assert re.match("Chan[0-9]{3}[a-z]{1}", k) is not None
            neuron_name = k[4:]
            neuron_chan = int(k[4:-1])
            neuron_spktimes[neuron_name] = np.real(raw[k].flatten())
            neuron_sites[neuron_name] = 'm1' if neuron_chan in raw['MIchans'] else 'pmd'
        assert len(neuron_spktimes) > 0
        return NeuralData(neuron_spktimes, neuron_sites)

    # ----------
    # core:

    raw_ = loadmat(os.path.expanduser(data_dir) + "/" + _DATASETS[dataset]["file"])

    # full kinematics:
    X = np.stack([raw_['x'][:, 1], raw_['y'][:, 1]], axis=1)
    t = .5 * (raw_['x'][:, 0] + raw_['y'][:, 0])
    kin = calc_kinematics(X, t, smooth_sig=bin_sz)

    # full neural:
    neural = _make_neural_data(raw_)

    # make trials:
    events, properties = _get_CO_events(raw_) if 'cpl_0deg' in raw_ else _get_TP_events(raw_)
    trials = []
    for ix in range(len(events)):

        # trial skeleton:
        tr = Trial(dataset=dataset, ix=ix, lag=lag, bin_sz=bin_sz, _events=events[ix], _properties=properties[ix])

        # add neural data:
        tr.neural = neural.TimeSlice([tr.st, tr.end - lag])
        tr.neural.make_spkcounts_(bin_sz)

        # add kinematics:
        tr.kin = kin.TimeSlice([tr.st + lag - .1, tr.end])
        tr.kin.resample_(tr.neural.t + lag)

        # additional events:
        tr.add_events({
            "max_spd": tr.kin.t[np.argmax(tr.kin["spd2"])],
        })

        trials.append(tr)
        assert tr.kin.num_samples == tr.neural.num_samples
        assert np.max(np.abs((tr.kin.t - tr.neural.t) - lag)) < 1e-6

    # normalize spike counts per neuron:
    all_spkcounts = np.concatenate([tr.neural.get_spkcounts()[0] for tr in trials], axis=1)
    mu = np.mean(all_spkcounts, axis=1)
    sd = np.maximum(np.std(all_spkcounts, axis=1), np.finfo(float).eps)
    for tr in trials:
        tr.neural.set_spkcounts((tr.neural.get_spkcounts()[0] - mu[:, None]) / sd[:, None])

    # ---
    # sanity-
    # properties that are suppose to be the same for all trials:
    assert all([tr.neural.sites == trials[0].neural.sites for tr in trials])
    assert all([tr.dataset == trials[0].dataset for tr in trials])
    assert all([tr.lag == trials[0].lag for tr in trials])
    assert all([tr.bin_sz == trials[0].bin_sz for tr in trials])
    # verify indices:
    assert [tr.ix for tr in trials] == list(range(len(trials)))
    # verify equal number of neural and kinematic samples, in each trial:
    assert all([tr.neural.num_samples == tr.kin.num_samples for tr in trials])
    # ---

    meta = DatasetMeta(**{**{"name": dataset, "sites": trials[0].neural.sites}, **_DATASETS[dataset]})

    return HatsoData(trials, meta)
