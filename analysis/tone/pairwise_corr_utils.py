import numpy as np
import pandas as pd
import scipy.stats as stats
import sys
import matplotlib.pyplot as plt 
from tqdm import tqdm
import matplotlib.patches as mpatches
sys.path.extend(['/projectnb/sramirezlab/amonast/Engram_2P/utilities',
                 '/projectnb/sramirezlab/amonast/Amy_2P/analysis/correlation'])
from Correlation import correlation, corr_matrix

#  Epoch / mask helpers
def mask_from_windows(t, windows):
    """
    Build a boolean mask over t for one or more (start, stop) time windows.

    Parameters
    ----------
    t : array-like of float
        1-D time array (seconds).
    windows : tuple or list of tuples
        A single (start, stop) tuple or a list of (start, stop) tuples (seconds).

    Returns
    -------
    mask : ndarray of bool, shape (len(t),)
        True where t falls within any of the specified windows.
    """
    t = np.asarray(t).squeeze()
    if isinstance(windows, tuple) and np.ndim(windows[0]) == 0:
        windows = [windows]
    mask = np.zeros(len(t), dtype=bool)
    for start, stop in windows:
        mask |= (t >= start) & (t <= stop)
    return mask

def get_run_epochs(t, run_bool, min_dur=1.0, merge_gap=0.5):
    """
    Extract running-bout (start, stop) times from a boolean running array.

    Parameters
    ----------
    t : array-like of float
        1-D time array (seconds), aligned to run_bool.
    run_bool : array-like of bool
        Boolean array where True indicates the animal is running.
    min_dur : float, optional
        Minimum bout duration in seconds to retain. Default 1.0.
    merge_gap : float, optional
        Merge consecutive bouts separated by fewer than this many seconds.
        Default 0.5.

    Returns
    -------
    epochs : list of tuple of float
        List of (start, stop) times in seconds for each running bout.
    """
    t = np.asarray(t).squeeze()
    rb = np.asarray(run_bool, dtype=bool)

    onsets  = np.where(np.diff(rb.astype(int)) ==  1)[0] + 1
    offsets = np.where(np.diff(rb.astype(int)) == -1)[0] + 1
    if rb[0]:
        onsets = np.insert(onsets, 0, 0)
    if rb[-1]:
        offsets = np.append(offsets, len(t))

    bouts = [[t[on], t[off - 1]] for on, off in zip(onsets, offsets)]

    merged = []
    for start, stop in bouts:
        if merged and (start - merged[-1][1]) < merge_gap:
            merged[-1][1] = stop
        else:
            merged.append([start, stop])

    return [(s, e) for s, e in merged if (e - s) >= min_dur]

def get_rest_epochs(t, run_bool, min_dur=1.0):
    """
    Extract rest-bout (start, stop) times as the complement of running.

    Parameters
    ----------
    t : array-like of float
        1-D time array (seconds).
    run_bool : array-like of bool
        Boolean array where True indicates the animal is running.
    min_dur : float, optional
        Minimum bout duration in seconds to retain. Default 1.0.

    Returns
    -------
    epochs : list of tuple of float
        List of (start, stop) times in seconds for each rest bout.
    """
    return get_run_epochs(t, ~np.asarray(run_bool, dtype=bool),
                          min_dur=min_dur, merge_gap=0.0)

def get_peri_event_windows(event_times, pre=5.0, post=5.0):
    """
    Build (start, stop) windows centered around event onset times.

    Parameters
    ----------
    event_times : array-like of float
        Event onset times in seconds.
    pre : float, optional
        Seconds before each event to include (positive value). Default 5.0.
    post : float, optional
        Seconds after each event to include. Default 5.0.

    Returns
    -------
    windows : list of tuple of float
        List of (start, stop) tuples in seconds.

    Examples
    --------
    5 s before each run start, 10 s after:
    windows = get_peri_event_windows(run_starts, pre=5, post=10)
    """
    return [(float(ev) - pre, float(ev) + post) for ev in np.asarray(event_times)]

def filter_windows_by_mask(t, windows, condition_mask):
    """
    Restrict epoch windows to timepoints where condition_mask is True.

    Intersects a set of time windows with a boolean condition array, returning
    a single mask. Useful for extracting e.g. rest-only frames within trial
    epochs to compare "rest during CS1" vs "matched baseline rest".

    Parameters
    ----------
    t : array-like of float
        1-D time array (seconds).
    windows : list of tuple of float
        List of (start, stop) tuples in seconds.
    condition_mask : array-like of bool
        Boolean array aligned to t (e.g. ``~run_bool``).

    Returns
    -------
    mask : ndarray of bool, shape (len(t),)
        True where t is inside any window AND condition_mask is True.

    Examples
    --------
    cs1_rest      = filter_windows_by_mask(t, zip(cs1_start, cs1_stop), ~run_bool)
    baseline_rest = filter_windows_by_mask(t, [(0, cs1_start[0]-20)], ~run_bool)
    bl_matched    = match_epoch_time(t, cs1_rest, baseline_rest)
    """
    t = np.asarray(t).squeeze()
    epoch_mask = mask_from_windows(t, list(windows))
    return epoch_mask & np.asarray(condition_mask, dtype=bool)

def _get_contiguous_segments(mask):
    """
    Return index pairs for every contiguous run of True values in mask.

    Parameters
    ----------
    mask : array-like of bool
        Boolean array.

    Returns
    -------
    segments : list of tuple of int
        List of (start_idx, stop_idx) half-open index pairs in temporal order.
    """
    mask = np.asarray(mask, dtype=bool)
    padded  = np.concatenate([[False], mask, [False]])
    onsets  = np.where(np.diff(padded.astype(int)) ==  1)[0]
    offsets = np.where(np.diff(padded.astype(int)) == -1)[0]
    return list(zip(onsets.tolist(), offsets.tolist()))

def sample_equal_frames(source_mask, n_frames):
    """
    Take exactly n_frames from source_mask as contiguous time segments.

    Always starts from the first available segment and accumulates forward
    in time. The final segment is truncated at its end if needed to reach
    exactly n_frames.

    Parameters
    ----------
    source_mask : ndarray of bool
        Boolean array of candidate frames.
    n_frames : int
        Total number of frames to collect.

    Returns
    -------
    out : ndarray of bool, shape (len(source_mask),)
        Boolean mask with exactly n_frames True, drawn from contiguous
        segments in temporal order.

    Raises
    ------
    ValueError
        If source_mask contains fewer True frames than n_frames.
    """
    segments = _get_contiguous_segments(source_mask)
    total_available = sum(e - s for s, e in segments)

    if total_available < n_frames:
        raise ValueError(
            f"source_mask has only {total_available} frames across all segments, "
            f"but {n_frames} were requested."
        )

    out = np.zeros(len(source_mask), dtype=bool)
    collected = 0
    for start, stop in segments:
        if collected >= n_frames:
            break
        needed  = n_frames - collected
        seg_len = stop - start
        if seg_len <= needed:
            out[start:stop] = True
            collected += seg_len
        else:
            out[start:start + needed] = True
            collected += needed

    return out

def match_epoch_time(t, target, source_mask):
    """
    Sample from source_mask a frame count matching the total frames in target.

    Determines how many frames are covered by target, then calls
    `sample_equal_frames` to take that many frames from source_mask starting
    from the earliest available segment. Primary helper for equal-time
    comparisons between epoch types.

    Parameters
    ----------
    t : array-like of float
        1-D time array (seconds).
    target : ndarray of bool or list of tuple of float
        Either a boolean mask whose True-frame count sets the target, or a
        list of (start, stop) windows from which the frame count is derived.
    source_mask : ndarray of bool
        Boolean mask of candidate frames to draw from.

    Returns
    -------
    out : ndarray of bool, shape (len(source_mask),)
        Boolean mask with frame count equal to target, drawn from contiguous
        segments of source_mask in temporal order.

    Examples
    --------
    Baseline rest matched in duration to total rest during CS1 trials:

    >>> cs1_rest   = filter_windows_by_mask(t, zip(cs1_start, cs1_stop), ~run_bool)
    >>> bl_rest    = filter_windows_by_mask(t, [(0, cs1_start[0]-20)], ~run_bool)
    >>> bl_matched = match_epoch_time(t, cs1_rest, bl_rest)
    """
    t = np.asarray(t).squeeze()
    if isinstance(target, np.ndarray) and target.dtype == bool:
        n_frames = int(target.sum())
    else:
        n_frames = int(mask_from_windows(t, list(target)).sum())
    return sample_equal_frames(source_mask, n_frames)
    
# Core correlation computation (single epoch slice) 
def _corr_pairs_df(d_tag, d_non, period_name, trial_num, corr_method):
    """
    Compute all three pairwise correlation groups for one epoch slice.

    Internal helper used by `pairwise_corr_epochs`.

    Parameters
    ----------
    d_tag : ndarray, shape (N_tag, T)
        Tagged-cell activity for this epoch.
    d_non : ndarray, shape (N_non, T)
        Non-tagged-cell activity for this epoch.
    period_name : str
        Label written to the ``Period`` column.
    trial_num : int
        Trial number written to the ``Trial`` column.
    corr_method : {'spearman', 'kendall'}
        Correlation method.

    Returns
    -------
    df : DataFrame
        Tidy DataFrame with columns: ``corr``, ``pvals``, ``pair_type``,
        ``period``, ``trial``,.
    """
    C_tag = correlation(d_tag)
    C_tag.get_corr_matrix(method=corr_method)
    r_tag = C_tag.get_lower()
    p_tag = C_tag.get_lower_pvals()

    C_non = correlation(d_non)
    C_non.get_corr_matrix(method=corr_method)
    r_non = C_non.get_lower()
    p_non = C_non.get_lower_pvals()

    C_mix, p_mix = corr_matrix(d_tag, d_non, method=corr_method)
    r_mix = C_mix.ravel()
    p_mix = p_mix.ravel()

    n_tag, n_non, n_mix = r_tag.shape[0], r_non.shape[0], r_mix.shape[0]
    n_total = n_tag + n_non + n_mix

    #Store indices of cell pairs in the corr matrix
    N_tag = d_tag.shape[0]
    N_non = d_non.shape[0]
    i_tag,j_tag = np.tril_indices(N_tag,k=-1)
    i_non,j_non = np.tril_indices(N_non,k=-1)
    i_mix,j_mix = np.indices((N_tag,N_non))
    i_mix,j_mix = i_mix.ravel(),j_mix.ravel()
    
    df = pd.DataFrame({
        'corr':       np.concatenate([r_tag, r_non, r_mix]),
        'pvals':      np.concatenate([p_tag, p_non, p_mix]),
        'pair_type': (['EE'] * n_tag +
                       ['NN'] * n_non +
                       ['EN'] * n_mix),
        'period':     [period_name] * n_total,
        'trial':      [trial_num] * n_total, 
        'cell_i':      np.concatenate([i_tag,i_non,i_mix]),
        'cell_j':       np.concatenate([j_tag,j_non,j_mix])})
    return df

#  Main per-animal/FOV correlation function 
def pairwise_corr_epochs(d_tag, d_non, t, named_epochs,
                         ani, group, fov,
                         trial_table=None, 
                         corr_method='spearman'):
    """
    Compute pairwise correlations for a flexible set of named time epochs.

    Called once per animal / FOV after loading data. Accepts pre-built boolean
    masks (e.g. from `filter_windows_by_mask` or `match_epoch_time`) as well
    as raw (start, stop) window specifications.

    Parameters
    ----------
    d_tag : ndarray, shape (N_tag, T)
        Tagged-cell activity for the full session.
    d_non : ndarray, shape (N_non, T)
        Non-tagged-cell activity for the full session.
    t : array-like of float
        1-D time vector (seconds), length T.
    named_epochs : dict
        Maps period name (str) to an epoch specification. Each value can be:

        (a) ``(start, stop)`` tuple â€” single window, trial=0.
        (b) Boolean ndarray of length T â€” used directly as a mask, trial=0.
        (c) ``[(start, stop), ...]`` â€” one row per window; trial numbers
            looked up from trial_table by list index.
        (d) ``[((start, stop), trial_num), ...]`` â€” explicit trial numbers.

        Examples::

            named_epochs = {
                'baseline' : (0, 580),
                'cs1'      : list(zip(cs1_start, cs1_stop)),
                'trace1'   : list(zip(cs1_stop, cs1_stop + 25)),
                'run_bouts': get_run_epochs(t, run_bool),
                'pre_run'  : get_peri_event_windows(run_starts, pre=5, post=0),
                'cs1_rest' : filter_windows_by_mask(
                                 t, zip(cs1_start, cs1_stop), ~run_bool),
                'bl_matched': match_epoch_time(t, cs1_rest_mask, bl_rest_mask),}

    ani : str
        Animal identifier, added to every row.
    group : str
        Experimental group label, added to every row.
    fov : str
        FOV identifier, added to every row.
    trial_table : DataFrame, optional
        Must contain columns ``trial`` and ``type_trial``. Used to look up
        trial numbers by list index for case (c) epoch specs.
    corr_method : {'spearman', 'kendall'}, optional
        Correlation method. Default ``'spearman'``.

    Returns
    -------
    DF : DataFrame
        Tidy DataFrame with all epochs stacked, ready to concatenate across
        animals. Columns: ``corr``, ``pvals``, ``pair_type``, ``period``,
        ``trial``, ``animal``, ``group``, ``fov``.
    """
    t = np.asarray(t).squeeze()
    DF = pd.DataFrame()

    for period_name, spec in named_epochs.items():

        # normalise spec â†’ list of (boolean_mask, trial_num)
        entries = []

        if isinstance(spec, np.ndarray) and spec.dtype == bool:
            # case (b): pre-built boolean mask
            entries = [(spec, 0)]

        elif isinstance(spec, tuple) and np.ndim(spec[0]) == 0:
            # case (a): bare (start, stop)
            entries = [(mask_from_windows(t, spec), 0)]

        else:
            # case (c) or (d): list
            for ii, item in enumerate(spec):
                if (isinstance(item, (list, tuple)) and len(item) == 2
                        and isinstance(item[0], tuple)
                        and isinstance(item[1], (int, np.integer))):
                    # case (d): ((start, stop), trial_num)
                    window, t_num = item
                    entries.append((mask_from_windows(t, window), int(t_num)))
                else:
                    # case (c): bare (start, stop), look up trial from table
                    t_num = 0
                    if trial_table is not None:
                        row = trial_table.loc[trial_table['type_trial'] == ii]
                        if len(row):
                            t_num = int(row['trial'].values[0])
                    entries.append((mask_from_windows(t, item), t_num))

        for mask, trial_num in entries:
            if mask.sum() == 0:
                continue
            df = _corr_pairs_df(d_tag[:, mask], d_non[:, mask],
                                period_name, trial_num, corr_method)
            df['animal'] = ani
            df['group']  = group
            df['fov']    = fov
            DF = pd.concat([DF, df], ignore_index=True)

    return DF

# Epoch selection diagnostic plot
def plot_epoch_selection(t, vt, named_epochs,
                         cs_starts=None, cs_stops=None,
                         epoch_colors=None,
                         epoch_alpha=0.25,
                         vline_color='k',
                         ax=None,
                         title=None):
    """
    Plot velocity trace with CS onset/offset lines and shaded epoch selections.

    Parameters
    ----------
    t : array-like of float
        1-D time vector (seconds), length T.
    vt : array-like of float
        Velocity (or Î”F/F proxy) trace, length T.
    named_epochs : dict
        Maps period name (str) to a boolean mask (ndarray of bool, length T).
        Masks produced by ``filter_windows_by_mask``, ``match_epoch_time``,
        ``mask_from_windows``, etc. are all accepted directly.
    cs_starts : array-like of float, optional
        CS onset times (seconds). Drawn as solid vertical lines.
    cs_stops : array-like of float, optional
        CS offset times (seconds). Drawn as dashed vertical lines.
    epoch_colors : dict, optional
        Maps period name to a matplotlib colour string.  Missing names get
        auto-colours from the default colour cycle.
    epoch_alpha : float, optional
        Alpha for epoch shading.  Default 0.25.
    vline_color : str, optional
        Colour for the CS onset/offset lines.  Default ``'k'``.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on.  Created if None.
    title : str, optional
        Axes title.

    Returns
    -------
    ax : matplotlib.axes.Axes
    """
    t   = np.asarray(t).squeeze()
    vt  = np.asarray(vt).squeeze()

    if ax is None:
        fig, ax = plt.subplots(figsize=(14, 3))

    ax.plot(t, vt, color='0.3', lw=0.8, zorder=2)

    # default colour cycle
    prop_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    auto_colors = iter(prop_cycle)
    if epoch_colors is None:
        epoch_colors = {}

    legend_handles = []
    for name, mask in named_epochs.items():
        mask = np.asarray(mask, dtype=bool)
        color = epoch_colors.get(name) or next(auto_colors)
        epoch_colors[name] = color  # store so shading and legend match

        # shade each contiguous run of True frames
        segs = _get_contiguous_segments(mask)
        for i, (s, e) in enumerate(segs):
            ax.axvspan(t[s], t[e - 1],
                       color=color, alpha=epoch_alpha, lw=0, zorder=1,
                       label=name if i == 0 else None)

        n_frames = int(mask.sum())
        patch = mpatches.Patch(color=color, alpha=min(epoch_alpha * 2, 1.0),
                               label=f'{name}  (n={n_frames})')
        legend_handles.append(patch)

    # CS onset / offset lines
    if cs_starts is not None:
        for x in np.asarray(cs_starts):
            ax.axvline(x, color=vline_color, lw=1.0, ls='-', zorder=3)
    if cs_stops is not None:
        for x in np.asarray(cs_stops):
            ax.axvline(x, color=vline_color, lw=1.0, ls='--', zorder=3)

    # compact legend entries for CS lines
    if cs_starts is not None:
        legend_handles.append(
            plt.Line2D([0], [0], color=vline_color, lw=1.0, ls='-',  label='CS onset'))
    if cs_stops is not None:
        legend_handles.append(
            plt.Line2D([0], [0], color=vline_color, lw=1.0, ls='--', label='CS offset'))

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Velocity')
    ax.legend(handles=legend_handles, loc='upper right', fontsize=7,
              framealpha=0.7)
    if title:
        ax.set_title(title)

    return ax