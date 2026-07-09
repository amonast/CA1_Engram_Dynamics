import numpy as np
import pandas as pd
import glob
import os 
try:
    import caiman as cm 
    from caiman.source_extraction.cnmf import cnmf as cnmf
    from caiman.utils.visualization import get_contours
except ImportError:
    import warnings
    warnings.warn("Caiman package not found in env, cannot use any dependent functions")
import h5py


#%%

__all__=['get_h5','load_cnmf','load_rois','remove_bad_cells','get_fov_data','save_summary_images_batch','save_summary_images','load_avg_img']
#%%
def get_h5(base_dir,animals):
    '''
    base_dir: base directory
    animals: list
    '''
    
    h5_all = []
    for ani in animals:
        cnmf_path = os.path.join(base_dir,'GCaMP',ani,'caiman_output','cnmf')+os.path.sep
        h5files= glob.glob(os.path.join(cnmf_path,'*.hdf5'))
        [h5_all.append(h) for h in h5files]
    return h5_all

def load_mch_ind(animal,fov,file_key,base_dir,registered=False,snr_thr=4.0):
    '''
    updated function
    animal: animal, str
    fov: fov, str
    file_key: metadata csv
    base_dir: base directory, contains Tagging/*_indices_split.csv
    returns:
            mch_pre: indices of tagged cells in baseline session
            mch_post: indices of tagged cells in post session
            non_pre: indices of nontagged cells in baseline session
            non_post: indices of nontagged cells in post session
    '''
    data = remove_bad_cells(animal,fov,file_key,base_dir,snr_thr=4.0)
    if registered:
        mch_pre = data.Baseline.loc[(data.Tagged==1)&(data.Baseline!=-1)&(data.Post!=-1)].values
        mch_post = data.Post.loc[(data.Tagged==1)&(data.Baseline!=-1)&(data.Post!=-1)].values
        non_pre = data.Baseline.loc[(data.Tagged==0)&(data.Baseline!=-1)&(data.Post!=-1)].values
        non_post = data.Post.loc[(data.Tagged==0)&(data.Baseline!=-1)&(data.Post!=-1)].values
    else:
        mch_pre = data.Baseline.loc[(data.Tagged==1)&(data.Baseline!=-1)].values
        mch_post = data.Post.loc[(data.Tagged==1)&(data.Post!=-1)].values
        non_pre = data.Baseline.loc[(data.Tagged==0)&(data.Baseline!=-1)].values
        non_post = data.Post.loc[(data.Tagged==0)&(data.Post!=-1)].values

    return mch_pre, mch_post, non_pre, non_post


def remove_bad_cells(ani,fov,file_key,base_dir,snr_thr:float,filter_mchleak=True,return_leaky=False,suppress_prints=False):
    """
    Filter low-SNR and mCherry-leaky cells from a tagged cell index table. Only used for first CFC mouse cohorts. Fig 1-4.

    Loads the split cell-index CSV for a given animal/FOV, flags cells whose
    SNR falls below threshold in any session as bad (Score == 3), removes
    them, and optionally un-tags cells with abnormally high pre-tagging
    mCherry signal (>2.5 SD above the mean) as likely leaky/false positives.

    Parameters
    ----------
    ani : str
        Animal identifier used to locate the indices CSV and filter session info.
    fov : str
        Field-of-view identifier used to locate the indices CSV.
    file_key : str
        Path to the CSV file containing session/animal metadata.
    base_dir : str
        Root directory containing the ``Tagging`` folder with indices CSVs.
    snr_thr : float
        Minimum acceptable SNR; cells below this in any session are dropped.
    filter_mchleak : bool, optional
        If True, also un-tag cells with pre-tagging mCherry signal more than
        2.5 standard deviations above the mean (default True).
    return_leaky : bool, optional
        If True, also return a DataFrame of the cells that were un-tagged
        for mCherry leakiness (default False).
    suppress_prints : bool, optional
        If True, suppress informational print statements (default False).

    Returns
    -------
    filtered_df : pandas.DataFrame
        Cell index table with low-SNR cells removed and, if
        ``filter_mchleak`` is True, leaky cells un-tagged.
    leaky : pandas.DataFrame, optional
        Rows corresponding to cells un-tagged for mCherry leakiness.
        Only returned if ``return_leaky`` is True.
    """
    csvfile =  glob.glob(f"{base_dir}{os.path.sep}Tagging{os.path.sep}{ani}_{fov}_indices_split.csv")[0]
    if suppress_prints==False:
        print(csvfile)
    
    ind_df = pd.read_csv(csvfile,index_col=0)
    info=pd.read_csv(file_key) 
    
    sessions = info.Session.loc[info.Animal==ani].unique()
    if 'Score' not in ind_df.columns:
        ind_df['Score'] = [999] * ind_df.shape[0]

    cnms = get_fov_data(ani, fov, file_key, base_dir)[3]
    thr = snr_thr

    # Loop over ind_df to remove SNR low cells
    for i, row in ind_df.iterrows():
        for session_idx,session_name in enumerate(sessions):
            snr_array = cnms[session_idx].estimates.SNR_comp
            
            # Check if session exists in the row
            if row[session_name] != -1:
                cell_idx = int(row[session_name])
                if snr_array[cell_idx] < thr:
                    ind_df.at[i, 'Score'] = 3
                    break  # Stop checking further sessions if threshold is met

    filtered_df = ind_df.loc[ind_df.Score != 3]

    ### drop high mcherry cells from D0 that are >2.5 std above D0 distribution.
    if return_leaky:
        leaky=pd.DataFrame()
    if filter_mchleak:
        try:
            mch_pre = [float(filtered_df['mch_pre'].values[i][1:-1]) for i in range(len(filtered_df))]
            mch_post =  [float(filtered_df['mch_post'].values[i][1:-1]) for i in range(len(filtered_df))]

        except:
            mch_pre = [float(filtered_df['mch_pre'].values[i]) for i in range(len(filtered_df))]
            mch_post =  [float(filtered_df['mch_post'].values[i]) for i in range(len(filtered_df))]
        df2 = pd.DataFrame()
        df2['Tagged']=filtered_df['Tagged'].copy()
        df2['mch_pre']=mch_pre
        df2['mch_post']=mch_post
        df2['fold change']=np.array(mch_post,dtype=object)/np.array(mch_pre,dtype=object)
        for i,row in filtered_df.iterrows():
            #if (float(row['mch_pre'][1:-1])>=(np.mean(df2.mch_pre)+(2.5*np.std(df2.mch_pre))))&(row['Tagged']==1):
            if (float(row['mch_pre'])>=(np.mean(df2.mch_pre)+(2.5*np.std(df2.mch_pre))))&(row['Tagged']==1):
                filtered_df.at[i,'Tagged']=0
                if suppress_prints==False:
                    print('dropped leaky cell')
                if return_leaky:
                    leaky = pd.concat([leaky,row.to_frame().T],ignore_index=True)
                

    print(str(ind_df.shape[0]) + ' cells filtered to '+str(filtered_df.shape[0])+'cells')
   
    if return_leaky:
        return filtered_df,leaky
    else:
        return filtered_df


def load_cnmf(TSeries):
    """
    Load a saved CNMF object for a given TSeries session.

    Parameters
    ----------
    TSeries : str
        Path to the TSeries directory; used to locate the corresponding
        animal's ``caiman_output/cnmf`` folder and matching HDF5 file.

    Returns
    -------
    cnm : caiman.source_extraction.cnmf.cnmf.CNMF
        CaImAn CNMF object loaded from the matching HDF5 file.
    """
    animal_path = os.path.split(TSeries)[0]
    T = os.path.split(TSeries)[-1] # name of TSeries 
    cnmf_path = os.path.join(animal_path, 'caiman_output', 'cnmf')
    cnmf_file = [os.path.join(cnmf_path,f) for f in os.listdir(cnmf_path) if (T in f)&('.hdf5' in f)]
    cnm = cnmf.load_CNMF(cnmf_file[0])
    #print(cnmf_file[0])
    return cnm

def get_fov_data(animal: str, FOV: str, file_key, base_path,suppress_print=True):
    """
    Load registered cell indices, summary images, and CNMF objects for a FOV. Only used for first CFC mouse cohorts. Fig 1-4.

    Retrieves the cell-to-index registration map produced by CellReg (converted
    to zero-based Python indexing), the per-session mean/std summary images, and
    the CaImAn CNMF objects for every session recorded for a given animal and
    field of view.

    Parameters
    ----------
    animal : str
        Animal name.
    FOV : str
        Field-of-view name.
    file_key : str
        Path to the metadata CSV for the experiment. Session names should be
        alphabetized in temporal order (e.g. ``session1, session2`` or
        ``baseline1, baseline2, post1, post2``) so that multi-session outputs
        are aligned in the correct order.
    base_path : str
        Base directory containing the processed data.
    suppress_print : bool, optional
        If True, suppress informational print statements (default True).

    Returns
    -------
    reg_ind : ndarray of int, shape (N, M)
        Cell registration indices (Python equivalent of ``cell_to_index_map``
        from CellReg output). N is the number of unique cells across all
        sessions, M is the number of sessions. A value of -1 indicates the
        cell was absent in that session.
    mean_list : list of ndarray, length M
        Mean image for each session.
    std_list : list of ndarray, length M
        Standard deviation image for each session.
    cnm_list : list, length M
        CaImAn CNMF objects for each session.
    """
    info = pd.read_csv(file_key)

    # First get cell registration indices & convert to pythonic indexing
    # each column is a session, each row is a cell. each entry is that cell's index in that session. if cell was absent its entry is -1
    reg_path = os.path.join(base_path, 'CellReg' + os.path.sep + animal + '_' + FOV + os.path.sep)
    reg_file = [os.path.join(reg_path, f) for f in os.listdir(reg_path) if ('cellRegistered' in f)&('._' not in f)]
    file = h5py.File(reg_file[-1], 'r') #get the LAST cell reg output file
    if suppress_print==False:
        print('Using cell reg output file: '+ reg_file[-1])
    group = file.get('cell_registered_struct')
    dset = group.get('cell_to_index_map')
    reg_ind = dset[()] - 1 # convert to pythonic indexing
    reg_ind = reg_ind.T
    reg_ind = reg_ind.astype('int')
    if suppress_print==False:
        print(str(reg_ind.shape[0]) + ' Cells detected in registration')

    # Get summary images and store in a list.
    # Each session has a .npz file, they should be alphabetized in temporal order i.e. session1,session2 or, baseline1,baseline2,post1,post2
    im_files = [f for f in os.listdir(reg_path) if (('_summary_images.npz' in f)&('._' not in f))]
    im_files.sort()
    mean_list = []
    std_list = []
    for im_file in im_files:
        mean_list.append(np.load(os.path.join(reg_path, im_file),allow_pickle=True)['mean_im'])
        std_list.append(np.load(os.path.join(reg_path, im_file),allow_pickle=True)['std_im'])
    
    ## Get CNMF objects and store in a list.
    TSeries_df = info['TSeries_g'].loc[(info['FOV'] == FOV) & (info['Animal'] == animal)].values
    TSeries_list = [TSeries for TSeries in TSeries_df]
    cnm_list = [load_cnmf(os.path.join(base_path,'GCaMP',animal,TSeries)) for TSeries in TSeries_list]

    for ii, cnm in enumerate(cnm_list):
        N = cnm.estimates.A.shape[1]
        if suppress_print==False:
            print(str(N) + ' Cells detected in session ' + str(ii + 1) + ' of ' + str(len(cnm_list)))

    return reg_ind, mean_list, std_list, cnm_list
