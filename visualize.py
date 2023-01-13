import numpy as np
import matplotlib.pyplot as plt
from data import Data
from typechecking import Pair, NpMat, NpVec, Tuple


def calc_event_triggered_response(data: Data, event="go", tradius=Pair[float], per_ang=False,
                                  tlims_mode="filter", shuff=False) -> Tuple[NpMat[float], NpVec]:
    """
    Calc event-triggered neural response
    Args:
        data:
        event: event name
        tradius: time radius [before, after] event
        per_ang: compute response per angle (for center out data)
        tr_mode: how time radius parameter show be used:
            "filter" - include only trials which fit within time radius
            "clip"  - include all trials, clip time radius to accommodate all trials
            "align" - include all trials as-is, align to event
        shuff: shuffle spike counts
    Returns:
        spkcounts - spike counts matrix
        tm_from_event - vector with len = spkcounts.shape[1], of time to event at each bin
    """

    # --
    # prep:

    t_before, t_after = tradius
    before_tms = np.array([tr[event] for tr in data]) - data.lag - t_before
    after_tms = np.array([tr[event] for tr in data]) - data.lag + t_after
    trial_starts = np.array([tr.neural.t[0] for tr in data])
    trial_ends = np.array([tr.neural.t[-1] for tr in data])

    print("Getting event triggered response")

    match tlims_mode:
        case "filter":
            valid_trials = (before_tms > trial_starts) & (after_tms < trial_ends)
            data.set_trials(valid_trials)
            print(f"Filter: Using {sum(valid_trials)}/{len(valid_trials)} ({np.mean(valid_trials):2.2f}) of trials")
        case "clip":
            t_before = min(np.max([np.min(before_tms - trial_starts), 0]), t_before)
            t_after = min(np.max([np.min(trial_ends - after_tms), 0]), t_after)
        case "align":
            pass
        case _:
            raise ValueError("Unknown time limits mode")

    print(f"Event: {event}, time before/after: {t_before}, {t_after}")

    bin_span = int(np.ceil((t_before + t_after) / data.bin_sz))
    event_bin = int(.5 + t_before / (t_before + t_after) * bin_span)
    tm_from_event = np.linspace(-t_before, t_after, bin_span)

    if per_ang:
        trial_angs = [tr['ang'] for tr in data]
    else:
        trial_angs = [0 for _ in data]

    angs = list(set(trial_angs))
    visitcounts = np.zeros((data.num_neurons, bin_span, len(angs)), int)
    spkcounts = np.zeros((data.num_neurons, bin_span, len(angs)), float)
    for ix, tr in enumerate(data):
        ang_ix = angs.index(trial_angs[ix])
        tlims = np.array([tr[event] - t_before, tr[event] + t_after]) - tr.lag
        if tlims_mode == "align":
            tlims = [np.max([tr.neural.t[0], tlims[0]])+1e-8, np.min([tr.neural.t[-1], tlims[-1]])-1e-8]
        s, tms, _ = tr.neural.get_spkcounts(tlims)
        if shuff:
            s = s[:, np.random.permutation(s.shape[1])]
        curr_event_bin = np.digitize(tr[event] - tr.lag, tms) - 1
        bin_st = event_bin - curr_event_bin
        bin_sp = bin_st + s.shape[1]
        spkcounts[:, bin_st: bin_sp, ang_ix] += s
        visitcounts[:, bin_st: bin_sp, ang_ix] += 1

    spkcounts /= np.maximum(visitcounts, 1)
    return spkcounts, tm_from_event


def plot_event_triggered_response(data: Data, event="go", tradius=Pair[float], per_ang=False,
                                  tlims_mode="filter", shuff=False):

    spkcounts, tms = calc_event_triggered_response(data=data, event=event, tradius=tradius, per_ang=per_ang,
                                                   tlims_mode=tlims_mode, shuff=shuff)
    assert len(tms) == spkcounts.shape[1]

    tm_ticks_lbls = np.sort(list(np.linspace(tms[0], tms[-1], np.min([len(tms), 9]))) + [0])
    tm_tick_ixs = np.searchsorted(tms, tm_ticks_lbls)
    tm_ticks_lbls = [f"{lbl:2.2f}" for lbl in tm_ticks_lbls]
    for k in range(spkcounts.shape[2]):
        plt.figure()
        img = spkcounts[:, :, k].T
        plt.imshow(img, cmap="hot")
        plt.yticks(tm_tick_ixs, tm_ticks_lbls)
        t0 = np.searchsorted(tms, -data.lag)
        plt.plot([0, img.shape[1] - 1], [t0, t0], "w")
        plt.title(str(data) + (f"[{k}]" if spkcounts.shape[2] > 1 else ""))


def plot_center_out_trajectories(data):
    angs = list(set([tr.ang for tr in data]))
    colors = plt.get_cmap('rainbow', len(angs))
    plt.figure()
    for tr in data:
        X = tr.kin.X
        plt.plot(X[:, 0], X[:, 1], color=colors(angs.index(tr.ang)))


