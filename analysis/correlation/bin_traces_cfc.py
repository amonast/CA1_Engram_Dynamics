#%%
import os
import pickle
import tqdm
import numpy as np
import sys
sys.path.extend(['/Users/amonast/Documents/GitHub/Engram_2P/Engram_2P'])
import pandas as pd 
from tqdm import tqdm
from utilities import animal
#%%
#ani,fov='1912L','FOV1'
def main():
    """
    Bin and save Baseline/Post event traces (tagged vs. non-tagged, all cells and registered-only) across a list of CFC animals/FOVs.

    For each combination of bin size, overlap setting, and registration
    filter (`reg`), splits each animal/FOV's Baseline and Post event traces
    into tagged/non-tagged cell groups, bins them, and saves the results to
    an `.npz` file.

    Parameters
    ----------
    None

    Returns
    -------
    None

    Notes
    -----
    `fovs_list`, `animals`, `bin_sizes`, `file_key`, and `base_dir` are
    hardcoded within the function body. Saves to
    `{base_dir}/Traces/{ani}/binned_traces_{[overlap]bin_size}/{ani}_{fov}_{bin_size}_binned_traces{_split|_split_all}.npz`.
    Unlike the Recall1/Recall2 scripts, this version does not bin or save
    timestamps.
    """
    fovs_list= [['FOV1','FOV2'],
                ['FOV1','FOV2'],
                ['FOV2'],
                ['FOV2'],
                ['FOV1','FOV2'],
                ['FOV1','FOV2'],
                ['FOV1'],
                ['FOV1','FOV2'],
                ['FOV1','FOV2'],
                ['FOV1'],
                ['FOV1','FOV2'],
                ['FOV2'],
                ['FOV1','FOV2'],
                ['FOV1'],
                ['FOV1','FOV2'],
                ['FOV1','FOV2']]
                # 
    
    animals =  ['589L','989N','992N','992L','994R','9972R',
                '217R','217N','218L','034R','149L',
                 '146R','160R','493R','492N','1912L']
    
    bin_sizes=[4,8,10,16,20,25,32,40, 50, 64, 80, 100, 125]

    #%% save traces split by tagged/untagged -- all cells across days
    print('saving binned split traces')
    for b in bin_sizes:
        for overlapping in [True,False]:
            for reg in [True,False]:
                print('bin size: '+str(b))
                for a,ani in enumerate(animals):
                    for fov in fovs_list[a]:
                        
                        mouse = animal.animal(ani,fov,file_key='/Volumes/AM_SSD1/Spont2P/Data_info.csv',base_dir='/Volumes/AM_SSD1/Spont2P')

                        binned_path =os.path.join(mouse.base_dir,'Traces',ani)

                        split_dict = split_allcells(ani, fov, mouse.file_key, mouse.base_dir,reg=reg)
                                        
                        append='_split.npz' if reg else '_split_all.npz'
                            
                        if overlapping == False:
                            d_pre_tag = bin_traces(split_dict['Baseline_tag'].to_numpy().T,bin_size=b)
                            d_post_tag = bin_traces(split_dict['Post_tag'].to_numpy().T,bin_size=b)
                            d_pre_non = bin_traces(split_dict['Baseline_nontag'].to_numpy().T,bin_size=b)
                            d_post_non = bin_traces(split_dict['Post_nontag'].to_numpy().T,bin_size=b)
                            np.savez(os.path.join(binned_path,'binned_traces_' + str(b),ani + '_' + fov + '_' + str(b) + f'_binned_traces{append}'),
                                    d_pre_tag=d_pre_tag, d_post_tag=d_post_tag, d_pre_non=d_pre_non, d_post_non=d_post_non)
                        elif overlapping == True:
                            d_pre_tag = bin_traces_overlap(split_dict['Baseline_tag'].to_numpy().T,bin_size=b)
                            d_post_tag = bin_traces_overlap(split_dict['Post_tag'].to_numpy().T,bin_size=b)
                            d_pre_non = bin_traces_overlap(split_dict['Baseline_nontag'].to_numpy().T,bin_size=b)
                            d_post_non = bin_traces_overlap(split_dict['Post_nontag'].to_numpy().T,bin_size=b)
                            np.savez(os.path.join(binned_path,'binned_traces_overlap' + str(b), ani + '_' + fov + '_' + str(b) + f'_binned_traces{append}'),
                                        d_pre_tag=d_pre_tag, d_post_tag=d_post_tag, d_pre_non=d_pre_non, d_post_non=d_post_non)
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
        Binned activity array, where B = T / bin_size * 2, summing values
        within each overlapping bin. Bins that run past the end of the
        trace are silently skipped (via `IndexError`), leaving those
        entries uninitialized.
    """
    T = D.shape[1]
    d_bin = np.empty([D.shape[0],int(T / bin_size * 2)])
    bin_starts = np.arange(0,T,bin_size/2,dtype=int)
    for n in np.arange(0,D.shape[0]):
        d = D[n,:]
        for i,b in enumerate(bin_starts):
            bin_end = b+bin_size
            try:
                d_bin[n,i] = np.sum(d[b:bin_end])
            except IndexError:
                pass
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

def split_allcells(ani,fov,file_key,base_dir,signal='events',reg=False):
    """
    Split every session's event traces into tagged and non-tagged cell columns.

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
    signal : str, optional
        Unused; `load_traces` is always called with `signal='events'`
        regardless of this argument (default 'events').
    reg : bool, optional
        If True, restrict to cells registered (present, index != -1) across
        all of the animal's sessions simultaneously; if False, restrict
        per-session to cells present in that session alone (default False).

    Returns
    -------
    split_dict : dict
        Dictionary keyed by `f"{session}_tag"` and `f"{session}_nontag"` for
        every session recorded for this animal/FOV (`mouse.sessions`),
        mapping to a dataframe of that session's timestamps plus the tagged
        (or non-tagged) cells' event traces.

    Notes
    -----
    Sessions are taken from `mouse.sessions` (every session for this
    animal/FOV), not a caller-supplied argument. Calls
    `mouse.load_cellreg(filter=True)`, but `animal.load_cellreg`'s
    parameter is actually named `drop_bad_cells`, not `filter` — this will
    still raise a `TypeError` as written.
    """
    mouse = animal.animal(ani,fov,file_key,base_dir)
    sessions=mouse.sessions

    ind_df = mouse.load_cellreg(filter=True)
    traces_dict = mouse.load_traces(signal='events')
    # Create a list of boolean expressions for each session

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