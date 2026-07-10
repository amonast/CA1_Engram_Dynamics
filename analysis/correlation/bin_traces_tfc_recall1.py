#%%
import os
import pickle
import numpy as np
import warnings
warnings.filterwarnings("ignore", message=r"Passing", category=FutureWarning)
import sys
sys.path.extend(['/Users/amonast/Documents/GitHub/Engram_2P/Engram_2P/'])
#from traces import get_traces
from bin_traces import save_bin_traces
import pandas as pd 
from utilities import animal

#%%
#ani,fov='1912L','FOV1'
def main():
    """
    Bin and save event traces (tagged vs. non-tagged) for the Recall1 session across a list of TFC animals.

    For each combination of bin size and overlap setting, splits each
    animal's Recall1 event traces into tagged/non-tagged cell groups, bins
    them (and the frame timestamps), and saves the results to an `.npz`
    file.

    Parameters
    ----------
    None

    Returns
    -------
    None

    Notes
    -----
    `animals`, `bin_sizes`, `file_key`, and `base_dir` are hardcoded within
    the function body (`file_key`/`base_dir` are set per-animal via the
    `animal.animal` constructor). Saves to
    `{base_dir}/Traces/binned_traces/binned_traces_{[overlap]bin_size}/{ani}_{fov}_{bin_size}_binned_traces{_split|_split_all}.npz`.
    """

    animals = ['997B','639N','M1N','M2L','M5L','194L',
               'M8BL2', 'M9BR2', 'F5L', 'F7N','939L']
    bin_sizes=[4,8,16,25,50,80,100,125]

    #%% save traces split by tagged/untagged -- all cells across days
    print('saving binned split traces')
    for b in bin_sizes:
        for overlapping in [True,False]:
            print('bin size: '+str(b))
            for a,ani in enumerate(animals):
                fov='FOV1'
                reg=False
                mouse = animal.animal(ani,fov,file_key='/Users/amonast/Desktop/Tone2P/Data_info_TFC.csv',base_dir='/Users/amonast/Desktop/Tone2P')
                timestamps = mouse.load_traces(sessions=['Recall1'])['timestamps'].to_numpy()
                binned_path =os.path.join(mouse.base_dir,'Traces','binned_traces')
                split_dict = split_allcells(ani, fov, mouse.file_key, mouse.base_dir,sessions=['Recall1'],reg=reg)
                append='_split.npz' if reg else '_split_all.npz'
                os.makedirs(os.path.join(binned_path,'binned_traces_overlap' + str(b)),exist_ok=True)
                os.makedirs(os.path.join(binned_path,'binned_traces_' + str(b)),exist_ok=True)
                if overlapping == False:
                    d_post_tag = bin_traces(split_dict['Recall1_tag'].to_numpy().T,bin_size=b)
                    d_post_non = bin_traces(split_dict['Recall1_nontag'].to_numpy().T,bin_size=b)
                    np.savez(os.path.join(binned_path,'binned_traces_' + str(b),ani + '_' + fov + '_' + str(b) + f'_binned_traces{append}'),
                            t = bin_traces_time(timestamps,bin_size=b),
                            d_post_tag=d_post_tag, d_post_non=d_post_non)
                elif overlapping == True:
                    d_post_tag = bin_traces_overlap(split_dict['Recall1_tag'].to_numpy().T,bin_size=b)
                    d_post_non = bin_traces_overlap(split_dict['Recall1_nontag'].to_numpy().T,bin_size=b)
                    np.savez(os.path.join(binned_path,'binned_traces_overlap' + str(b), ani + '_' + fov + '_' + str(b) + f'_binned_traces{append}'),
                                t = bin_overlap_time(timestamps,bin_size=b),
                                d_post_tag=d_post_tag, d_post_non=d_post_non)
                    
#%%
def get_bin_indices(T,bin_size,overlap=True):
    """
    Compute frame-index bin boundaries, optionally overlapping.

    Parameters
    ----------
    T : int
        Number of samples (frames) in time.
    bin_size : int
        Size of each bin, in frames.
    overlap : bool, optional
        If True, use 50%-overlapping bins (step size = bin_size / 2); if
        False, use non-overlapping bins (default True).

    Returns
    -------
    bin_dict : dict
        Dictionary mapping bin number to (bin_start, bin_end) frame index
        tuples.
    """
    n_samples = np.arange(0,T,1)
    bin_dict={}
    if overlap:
        bin_starts = np.arange(0,T,bin_size/2,dtype=int)
        for i,b in enumerate(bin_starts):
            bin_end = b+bin_size
            bin_dict[i]=(b,bin_end)
    else:
        bin_starts = np.arange(0,T,bin_size,dtype=int)
        for i,b in enumerate(bin_starts):
            bin_end = b+bin_size
            bin_dict[i]=(b,bin_end)
    return bin_dict

def bin_traces_overlap(D,bin_size=4):
    """
    Bin traces into overlapping bins (50% overlap) using the sum.

    Parameters
    ----------
    D : ndarray, shape (N, T)
        Deconvolved calcium activity for N cells over T frames.
    bin_size : int, optional
        Length of each bin, in frames (default 4).

    Returns
    -------
    d_bin : ndarray, shape (N, B)
        Binned activity array, summing values within each overlapping bin.
        Only full bins (windows entirely within [0, T)) are included, so B
        may differ from the `T / bin_size * 2` used by the other binning
        scripts' `bin_traces_overlap`.
    """
    T = D.shape[1]
    step = max(1, int(round(bin_size * 0.5)))
    bin_starts = np.arange(0, T - bin_size + 1, step, dtype=int)
    d_bin = np.empty([D.shape[0], len(bin_starts)])
    for n in np.arange(0,D.shape[0]):
        d = D[n,:]
        for i,b in enumerate(bin_starts):
            d_bin[n,i] = np.sum(d[b:b+bin_size])
    return d_bin

def bin_traces(D, bin_size=4):
    """
    Bin traces into non-overlapping bins using the sum.

    Parameters
    ----------
    D : ndarray, shape (N, T)
        Deconvolved calcium activity for N cells over T frames.
    bin_size : int, optional
        Length of each bin, in frames (default 4).

    Returns
    -------
    d_bin : ndarray, shape (N, T // bin_size)
        Binned activity array, summing values within each bin.
    """
    T = D.shape[1]
    d_bin = np.empty([D.shape[0],int(T / bin_size)])
    t = np.arange(0, T, bin_size)
    for n in np.arange(0,D.shape[0]):
        d=D[n,:]
        for i in np.arange(0, t.shape[0]):
            try:
                d_bin[n,i] = np.sum(d[t[i]:t[i] + bin_size])
            except IndexError:
                pass
    return d_bin

def bin_traces_time(D, bin_size=4):
    """
    Bin a time-series array into non-overlapping bins using the mean.

    Suitable for downsampling time arrays.

    Parameters
    ----------
    D : ndarray, shape (M, T)
        Input time array, where M is trials (or cells) and T is
        timepoints. 1D input is treated as a single trial.
    bin_size : int, optional
        Size of each bin, in frames (default 4).

    Returns
    -------
    d_bin : ndarray, shape (M, T // bin_size)
        Binned activity array, using the mean within each bin. Any
        trailing frames that don't fill a full bin are dropped.
    """
    if D.ndim == 1:
        D = D[np.newaxis, :]  # Convert to 2D if 1D input

    M, T = D.shape
    num_bins = T // bin_size
    D = D[:, :num_bins * bin_size]  # Truncate to fit full bins only

    D_reshaped = D.reshape(M, num_bins, bin_size)
    d_bin = D_reshaped.mean(axis=2)

    return d_bin

def bin_overlap_time(D, bin_size=4, overlap=0.5, agg="mean"):
    """
    Bin a time-series array into overlapping bins.

    Parameters
    ----------
    D : ndarray, shape (M, T) or (T,)
        Input time array, where M is trials and T is timepoints; a 1D
        array of shape (T,) is treated as a single trial.
    bin_size : int, optional
        Size of each bin, in frames (default 4).
    overlap : float, optional
        Fractional overlap between consecutive bins, in [0, 1). An
        overlap of 0.5 means 50% overlap (step size = bin_size / 2)
        (default 0.5).
    agg : {"mean", "sum"}, optional
        Aggregation function applied within each bin (default "mean").

    Returns
    -------
    d_bin : ndarray, shape (M, B)
        Binned array, where B is the number of overlapping bins. Only
        full bins (windows entirely within [0, T)) are included.

    Raises
    ------
    ValueError
        If `overlap` is not in [0, 1), if `bin_size` is not positive, or
        if `agg` is not "mean" or "sum".
    """
    if D.ndim == 1:
        D = D[np.newaxis, :]

    if not (0 <= overlap < 1):
        raise ValueError("overlap must be in [0, 1).")
    if bin_size <= 0:
        raise ValueError("bin_size must be a positive integer.")

    M, T = D.shape
    step = max(1, int(round(bin_size * (1 - overlap))))

    # Start indices such that [start, start+bin_size) is fully within the array
    starts = np.arange(0, T - bin_size + 1, step, dtype=int)
    B = starts.size
    d_bin = np.empty((M, B), dtype=float)

    for i, s in enumerate(starts):
        window = D[:, s:s + bin_size]
        if agg == "mean":
            d_bin[:, i] = window.mean(axis=1)
        elif agg == "sum":
            d_bin[:, i] = window.sum(axis=1)
        else:
            raise ValueError('agg must be "mean" or "sum".')

    return d_bin

def split_allcells(ani,fov,file_key,base_dir,sessions=['Recall1'],signal='events',reg=False):
    """
    Split each session's event traces into tagged and non-tagged cell columns.

    Parameters
    ----------
    ani : str
        Animal identifier.
    fov : str
        FOV identifier.
    file_key : str
        Path to the metadata CSV for the experiment.
    base_dir : str
        Base directory for the experiment.
    sessions : list of str, optional
        Sessions to load and split (default ['Recall1']).
    signal : str, optional
        Unused; `load_traces` is always called with `signal='events'`
        regardless of this argument (default 'events').
    reg : bool, optional
        If True, restrict to cells registered (present, index != -1) across
        all of `sessions` simultaneously; if False, restrict per-session to
        cells present in that session alone (default False).

    Returns
    -------
    split_dict : dict
        Dictionary keyed by `f"{session}_tag"` and `f"{session}_nontag"` for
        each session in `sessions`, mapping to a dataframe of that session's
        timestamps plus the tagged (or non-tagged) cells' event traces.
    """
    mouse = animal.animal(ani,fov,file_key,base_dir)

    ind_df = mouse.load_cellreg()
    traces_dict = mouse.load_traces(sessions=sessions,signal='events')
    # Create a list of boolean expressions for each session
    if len(sessions)==1:
        traces_dict={sessions[0]:traces_dict}
    # Example usage: Evaluate the first boolean expression
    if reg:
        # get the registered cell idxs where ind_df is not -1 for all sessions
        # store in dict of {session: idxs}
        idx_dict_unsort = {session:ind_df[session].loc[(ind_df[sessions] != -1).all(axis=1)].to_numpy() for session in sessions}
        # store if tagged or not in dict of {session: idxs}
        engram_bool = {session:ind_df['Tagged'].loc[(ind_df[sessions] != -1).all(axis=1)].to_numpy() for session in sessions}
    else:
        # get cell idxs where ind_df is not -1 for each session
        # store in dict of {session: idxs}
        idx_dict_unsort = {session:ind_df[session].loc[(ind_df[session]!=-1)].to_numpy() for session in sessions}
        # store if tagged or not in dict of {session: idxs}
        engram_bool={session:ind_df['Tagged'].loc[(ind_df[session]!=-1)].to_numpy() for session in sessions}
    
    split_dict={}
    for k in traces_dict.keys():
        cols = []
        for column in traces_dict[k].columns[1:]:
            tup = eval(column) #get the tuple of (EngramTrue/False, idx) from traces dataframe
            idx=tup[1] 
            engram=tup[0]
            
            if idx in idx_dict_unsort[k]:
                cols.append(column)
        df = pd.concat([traces_dict[k]['timestamps'],traces_dict[k][cols]],axis=1)
        split_dict[k+'_tag']=df.iloc[:,df.columns.str.contains('True')]
        split_dict[k+'_nontag']=df.iloc[:,df.columns.str.contains('False')]
        
    return split_dict 

if __name__=='__main__':
    main()
# %%
