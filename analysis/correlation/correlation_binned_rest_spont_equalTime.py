#%%
import pandas as pd
import numpy as np
import os
import scipy.stats as stats
from scipy.ndimage import label
import os
from scipy import stats
import sys
sys.path.extend(['/projectnb/sramirezlab/amonast/Amy_2P',
                 '/projectnb/sramirezlab/amonast/Engram_2P/Engram_2P',
                 '/Users/amonast/Documents/GitHub/Engram_2P/Amy_2P',
                 '/Users/amonast/Documents/GitHub/Engram_2P/Engram_2P'])

from behavior.running import get_rest_array,load_position,calc_velocity,thr_velocity
from analysis.correlation.Correlation import correlation,corr_matrix
#%%
def main():
    animals=['989N',
            '994R',
            '9972R',
            '217R',
            '217N',
            '218L',
            '034R',
            '149L',
            '146R',
            '160R',
            '493R',
            '492N',
            '1912L']

    fov_lists=[
            ['FOV1','FOV2'],
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

    #bin_sizes = [4,8,10,16,20,25,32,40, 50, 64, 80, 100, 125]
    base_dir='/Volumes/AM_SSD1/Spont2P'
    path = '/Volumes/AM_SSD1/Spont2P/Analysis/correlation/pairwise_dataframes_rest_equalTime'
    bin_path = '/Volumes/AM_SSD1/Spont2P/Traces'
    file_key = '/Volumes/AM_SSD1/Spont2P/Data_info.csv'
    metadata = pd.read_csv(file_key)
    overlap_bins=True 
    session='Baseline'
    
    #for b in bin_sizes:
    #### Save binned dataframes 
    b=4#int(sys.argv[1])
    if overlap_bins==False:
        savepath=os.path.join(path,'0_lag',str(b)+'_bin')
        if not os.path.exists(savepath):
            os.makedirs(savepath)
    elif overlap_bins==True:
        savepath=os.path.join(path,'0_lag',str(b)+'_bin_overlap')
        if not os.path.exists(savepath):
            os.makedirs(savepath)
            
    DF_pre = pairwise_corr_df(animals, fov_lists, 'Baseline', file_key,base_dir,
                                 binned_path=bin_path, binned=b,overlap_bins=overlap_bins,epoch='rest')
    DF_pre.to_csv(os.path.join(savepath,'pairwise_corr_df_pre.csv'))
    print('saved:'+ os.path.join(savepath,'pairwise_corr_df_pre.csv'))
    
    DF_post = pairwise_corr_df(animals, fov_lists, 'Post', file_key,base_dir,
                                 binned_path=bin_path, binned=b,overlap_bins=overlap_bins,epoch='rest')
    DF_post.to_csv(os.path.join(savepath,'pairwise_corr_df_post.csv'))
    print('saved:'+ os.path.join(savepath,'pairwise_corr_df_post.csv'))

def bin_traces_time(D, bin_size=4):
    """
    Bin a time-series array into non-overlapping windows by averaging.

    Parameters
    ----------
    D : ndarray, shape (N, T)
        Time array to bin, where N is the number of traces (e.g. cells) and
        T is the number of time frames.
    bin_size : int, optional
        Number of frames per bin (default 4).

    Returns
    -------
    d_bin : ndarray, shape (N, T // bin_size)
        Binned activity array, with each entry the mean of the corresponding
        time bin.
    """
    T = D.shape[1]
    d_bin = np.empty([D.shape[0],int(T / bin_size)])
    t = np.arange(0, T, bin_size)
    for n in np.arange(0,D.shape[0]):
        d=D[n,:]
        for i in np.arange(0, t.shape[0]):
            try:
                d_bin[n,i] = np.mean(d[t[i]:t[i] + bin_size])
            except IndexError:
                pass
    return d_bin

def pairwise_corr_df(animals, fov_lists, session, file_key,base_dir,
                     binned_path,binned=0,overlap_bins=True,ind=None,corr_method='spearman',reg='reg',epoch=None):
    """
    Compute pairwise correlations between tagged and non-tagged cells across animals/FOVs.

    For each animal and FOV, loads pre-binned activity traces (tagged vs.
    non-tagged cells, pre/post session), optionally restricts to a behavioral
    epoch (rest or run) based on velocity, and computes pairwise correlations
    (tagged-tagged, non-tagged-non-tagged, and tagged-non-tagged) for the
    specified session. Results from all animals/FOVs are concatenated into a
    single long-format DataFrame.

    Parameters
    ----------
    animals : list of str
        List of animal identifiers.
    fov_lists : list of list of str
        List of FOV identifiers per animal; must be the same length as
        ``animals``, with ``fov_lists[i]`` giving the FOVs for ``animals[i]``.
    session : {'Baseline', 'Post'}
        Session identifier specifying which session's correlations to compute.
    file_key : str
        Path to the metadata CSV containing experiment info and animal groups.
    base_dir : str
        Base directory for processed data, used when loading position data for
        epoch filtering.
    binned_path : str
        Path to the directory containing binned traces. Must contain
        subfolders named ``binned_traces_{binned}`` (non-overlapping) or
        ``binned_traces_overlap{binned}`` (overlapping), each holding files
        named ``{animal}_{fov}_{binned}_binned_traces_split.npz`` (or
        ``..._split_all.npz`` when ``reg='all'``).
    binned : int, optional
        Number of frames per bin; must match an existing binned-traces folder
        (default 0).
    overlap_bins : bool, optional
        If True, use overlapping bins; if False, use non-overlapping bins
        (default True).
    ind : array-like of int, optional
        Time indices of traces to use when ``binned == 0`` (default None,
        meaning all time points are used).
    corr_method : {'spearman', 'kendall'}, optional
        Correlation method: Spearman's rho or Kendall's tau-b (default
        'spearman').
    reg : {'reg', 'all'}, optional
        Which binned-traces file variant to load: ``'reg'`` for the registered
        cells file, ``'all'`` for the ``_split_all`` file (default 'reg').
    epoch : {'rest', 'run'}, optional
        If given, restricts correlations to time bins matching the specified
        behavioral epoch, determined from velocity-thresholded position data.
        If None, all time bins are used (default None).

    Returns
    -------
    DF : pandas.DataFrame
        Long-format dataframe of pairwise correlations for the given session,
        with columns:

        - ``Spearmans R`` : correlation coefficient for each cell pair
        - ``pvals`` : p-value for each correlation
        - ``Pair Group`` : one of 'Tagged vs Tagged', 'Non-tagged vs Non-tagged',
        'Tagged vs Non-tagged'
        - ``Session`` : session label ('Baseline' or 'Post')
        - ``Animal`` : animal identifier
        - ``Group`` : experimental group from metadata
        - ``ZScored Spearmans R`` : z-scored correlation coefficient
    """
    
    metadata = pd.read_csv(file_key)
    DF = pd.DataFrame()
    shapes = []
    for a, ani in enumerate(animals):
        print(ani)
        for fov in fov_lists[a]:
            print(fov)
            

            if reg=='reg':
                if (overlap_bins==False) or (binned==0):
                    data = np.load(os.path.join(binned_path, ani,"binned_traces_" + str(binned),
                                            ani + "_" + fov + "_" + str(binned) + "_binned_traces_split.npz"))
                elif overlap_bins==True:
                    data = np.load(os.path.join(binned_path,ani, "binned_traces_overlap" + str(binned),
                                            ani + "_" + fov + "_" + str(binned) + "_binned_traces_split.npz"))
            elif reg=='all':
                if (overlap_bins==False) or (binned==0):
                    data = np.load(os.path.join(binned_path, ani,"binned_traces_" + str(binned),
                                            ani + "_" + fov + "_" + str(binned) + "_binned_traces_split_all.npz"))
                elif overlap_bins==True:
                    data = np.load(os.path.join(binned_path, ani,"binned_traces_overlap" + str(binned),
                                            ani + "_" + fov + "_" + str(binned) + "_binned_traces_split_all.npz"))
            
            
            if ind is None:
                d_pre_tag = data['d_pre_tag']
                d_pre_non = data['d_pre_non']
                d_post_tag = data['d_post_tag']
                d_post_non = data['d_post_non']
            else:
                if binned==0:
                    d_pre_tag = data['d_pre_tag'][:,ind]
                    d_pre_non = data['d_pre_non'][:,ind]
                    d_post_tag = data['d_post_tag'][:,ind]
                    d_post_non = data['d_post_non'][:,ind]
            
            # Initialize container for indices
            ind_ds_baseline = None
            ind_ds_post = None

            # Precompute velocity and epoch-specific binary vector if epoch filtering is needed
            if epoch is not None:
                # Load position data
                df = load_position(ani, fov, session, file_key, base_dir)
                min_bout = 15  # Minimum bout duration in frames
                # Compute velocity and threshold it
                vt0 = calc_velocity(df.position.values, df.frame_times.values,
                                    frame_period=df.frame_times.values[1] - df.frame_times.values[0],
                                    window_size=min_bout)
                vt = thr_velocity(vt0)
                # Get binary rest/run arrays
                rest, run = get_rest_array(vt)
                # Choose appropriate binary state
                binary_state = rest if epoch == 'rest' else run
                # Bin the binary state array
                state_ds = np.round(bin_traces_time(np.expand_dims(binary_state, axis=0), bin_size=binned))
                # Store the indices where epoch == 1
                ind_ds_epoch = np.where(state_ds == 1)[1]
            else:
                # If no epoch filtering, default to using all bins
                total_bins = data['d_pre_tag'].shape[1]
                ind_ds_epoch = np.arange(total_bins)
            # Get number of valid bins for each session
            if session == 'Baseline':
                ind_ds_baseline = ind_ds_epoch
            elif session == 'Post':
                ind_ds_post = ind_ds_epoch
            # After running for both sessions, compute the minimum length
            if ind_ds_baseline is not None and ind_ds_post is not None:
                min_len = min(len(ind_ds_baseline), len(ind_ds_post))
                ind_ds_baseline = ind_ds_baseline[:min_len]
                ind_ds_post = ind_ds_post[:min_len]

                # Apply synchronized indexing
                d_pre_tag = data['d_pre_tag'][:, ind_ds_baseline]
                d_pre_non = data['d_pre_non'][:, ind_ds_baseline]
                d_post_tag = data['d_post_tag'][:, ind_ds_post]
                d_post_non = data['d_post_non'][:, ind_ds_post]

                print("d_pre_tag shape:", d_pre_tag.shape)
                print("d_post_tag shape:", d_post_tag.shape)  
                
            if session == 'Baseline':
                C_pre_mch = correlation(d_pre_tag)
                C_pre_mch.get_corr_matrix(method=corr_method)[0]
                pre_mch = C_pre_mch.get_lower()
                p_pre_mch = C_pre_mch.get_lower_pvals()
                pre_mch_group = ['Tagged vs Tagged'] * pre_mch.shape[0]
                pre_mch_session = ['Baseline'] * pre_mch.shape[0]

                C_pre_non = correlation(d_pre_non)
                C_pre_non.get_corr_matrix(method=corr_method)[0]
                pre_non = C_pre_non.get_lower()
                p_pre_non = C_pre_non.get_lower_pvals()
                pre_non_group = ['Non-tagged vs Non-tagged'] * pre_non.shape[0]
                pre_non_session = ['Baseline'] * pre_non.shape[0]

                C_pre_mix,p_pre_mix = corr_matrix(d_pre_tag, d_pre_non,method=corr_method)
                pre_mix = np.reshape(C_pre_mix, C_pre_mix.size)
                pre_mix_group = ['Tagged vs Non-tagged'] * pre_mix.shape[0]
                pre_mix_session = ['Baseline'] * pre_mix.shape[0]

                df = pd.DataFrame()
                df['Spearmans R'] = pd.concat([pd.Series(pre_mch), pd.Series(pre_non), pd.Series(pre_mix)],
                                              ignore_index=True)
                df['pvals']= pd.concat([pd.Series(p_pre_mch), pd.Series(p_pre_non), pd.Series(p_pre_mix.flatten())],
                                              ignore_index=True)
                df['Pair Group'] = pre_mch_group + pre_non_group + pre_mix_group
                df['Session'] = pre_mch_session + pre_non_session + pre_mix_session
                df['Animal'] = ani
                df['Group'] = metadata['Group'].loc[metadata['Animal'] == ani].values[0]
                df['ZScored Spearmans R'] = stats.zscore(df['Spearmans R'].values)

                #shapes.append(df.shape[0])
                DF = pd.concat([DF, df], ignore_index=True)

            elif session == 'Post':
                C_post_mch = correlation(d_post_tag)
                C_post_mch.get_corr_matrix(method=corr_method)[0]
                post_mch = C_post_mch.get_lower()
                p_post_mch = C_post_mch.get_lower_pvals()
                post_mch_group = ['Tagged vs Tagged'] * post_mch.shape[0]
                post_mch_session = ['Post'] * post_mch.shape[0]

                C_post_non = correlation(d_post_non)
                C_post_non.get_corr_matrix(method=corr_method)[0]
                post_non = C_post_non.get_lower()
                p_post_non = C_post_non.get_lower_pvals()

                post_non_group = ['Non-tagged vs Non-tagged'] * post_non.shape[0]
                post_non_session = ['Post'] * post_non.shape[0]

                C_post_mix,p_post_mix = corr_matrix(d_post_tag, d_post_non,method=corr_method)
                post_mix = np.reshape(C_post_mix, C_post_mix.size)
                post_mix_group = ['Tagged vs Non-tagged'] * post_mix.shape[0]
                post_mix_session = ['Post'] * post_mix.shape[0]

                df = pd.DataFrame()
                df['Spearmans R'] = pd.concat([pd.Series(post_mch), pd.Series(post_non), pd.Series(post_mix)],
                                              ignore_index=True)
                df['pvals']= pd.concat([pd.Series(p_post_mch), pd.Series(p_post_non), pd.Series(p_post_mix.flatten())],
                                              ignore_index=True)
                df['Pair Group'] = post_mch_group + post_non_group + post_mix_group
                df['Session'] = post_mch_session + post_non_session + post_mix_session
                df['Animal'] = ani
                df['Group'] = metadata['Group'].loc[metadata['Animal'] == ani].values[0]
                df['ZScored Spearmans R'] = stats.zscore(df['Spearmans R'].values)
                DF = pd.concat([DF, df], ignore_index=True)

    return DF


if __name__=='__main__':
    main()