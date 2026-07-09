#%%
import numpy as np
import os, time, sys, warnings
import pickle as pickle
#import scipy.io as sio
import FastLZeroSpikeInference.fast as fast
from L0_analysis import L0_analysis


#%%
def fit_lambda(dffs):
    """
    Fit L0 event-detection lambda values for a set of dF/F traces.

    Parameters
    ----------
    dffs : ndarray, shape (N, T)
        dF/F traces for N cells over T frames.

    Returns
    -------
    l0a : L0_analysis
        Fitted L0_analysis object, with per-cell lambda values computed via
        bisection and positive-only event constraints applied.
    """
    l0a = L0_analysis(dffs, event_min_size=5,use_cache=False)
    l0a.dff_traces
    l0a.L0_constrain=True # only zero or positive events
    l0a.get_events(use_bisection=True)
    return l0a
#%%
def run_l0events(l0,dffs):
    """
    Run L0 spike estimation for each trace using previously fit lambda values.

    Parameters
    ----------
    l0 : L0_analysis
        Fitted L0_analysis object (e.g. from `fit_lambda`), providing
        per-cell `lambdas` and `gamma`.
    dffs : ndarray, shape (N, T)
        dF/F traces for N cells over T frames.

    Returns
    -------
    dec : dict
        Deconvolution results with keys:

        - ``'dff'`` : the input dF/F traces.
        - ``'est'`` : ndarray, estimated (smoothed) calcium trace per cell.
        - ``'times'`` : list, per-cell arrays of event frame indices.
        - ``'events'`` : list, per-cell arrays of nonzero event magnitudes.
        - ``'gamma'`` : calcium decay constant used for estimation.
        - ``'lambda'`` : list of per-cell lambda values used.
    """
    est_all=[]
    times_all=[]
    events_all=[]
    for ii,dff in enumerate(dffs):
        lam = l0.lambdas[ii]
        gamma = l0.gamma
        y = dffs[ii]
        fit = fast.estimate_spikes(y, gamma, lam, False, True)

        times =  fit['spikes'] #event frames
        events = fit['pos_spike_mag'] #this still returns zt=0 as possible event size.

        if 0 in events:
            zero_ind = np.argwhere(events == 0).flatten()
            times = np.delete(times,zero_ind)
        events = events[events!= 0]

        est_all.append(fit['estimated_calcium'])
        times_all.append(times)
        events_all.append(events)

    dec={
        'dff': dffs,
        'est': np.asarray(est_all),
        'times': times_all,
        'events': events_all,
        'gamma': l0.gamma,
        'lambda': l0.lambdas,
    }
    return dec
#%%
def deconvolve(file,path):
    """
    Run L0 deconvolution on a saved dF/F file and save the results.

    Parameters
    ----------
    file : str
        Path to an `.npz` file containing a 'dff' array of dF/F traces.
    path : str
        Directory to save the output pickle files to.

    Returns
    -------
    None

    Notes
    -----
    Saves two pickle files to `path`, named `{file basename}_l0analysis.pkl`
    (the fitted `L0_analysis` object) and `{file basename}_l0deconv.pkl`
    (the deconvolution results dict from `run_l0events`).
    """
    fname = os.path.split(file)[-1].split('.')[0]
    dffs = np.load(file,allow_pickle=True)['dff']

    l0 = fit_lambda(dffs)
    dec = run_l0events(l0,dffs)

    with open(os.path.join(path,fname+'_l0analysis.pkl'),'wb') as file:
        pickle.dump(l0,file,protocol=pickle.HIGHEST_PROTOCOL)
    with open(os.path.join(path,fname+'_l0deconv.pkl'),'wb') as file2:
            pickle.dump(dec,file2,protocol=pickle.HIGHEST_PROTOCOL)
#%%
def main():
    """
    Run L0 deconvolution on a single dF/F file given as a command-line argument.
    Reads the input file path from `sys.argv[1]`.
    """
    path='/projectnb/sramirezlab/amonast/Tone2P/deconvolution/deconv_results/deconv_results_min5'
    os.makedirs(path,exist_ok=True)
    deconvolve(sys.argv[1],path)

if __name__ == '__main__':
    main()


