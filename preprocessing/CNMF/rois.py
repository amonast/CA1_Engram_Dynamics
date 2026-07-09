import numpy as np
import holoviews as hv
import scipy.io as sio
import pickle
import matplotlib.pyplot as plt
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
from tkinter import filedialog
import fnmatch

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

def load_rois(ani,fov,roi_path=None,reg=False):
    if roi_path is None:
        roi_path = filedialog.askdirectory()
    if reg:
        fname = os.path.join(roi_path,ani + '_' + fov + '_rois_final_reg.pkl')
    else:
        fname = os.path.join(roi_path,ani + '_' + fov + '_rois_final.pkl')
    with open(fname,'rb') as file:
        data = pickle.load(file)
    return data


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


def remove_bad_cells(ani,fov,file_key,base_dir,snr_thr:float,filter_mchleak=True,return_leaky=False,suppress_prints=False,sessions=None):
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
    '''
    Loads saved cnmf object for a session
    :param TSeries: TSeries path
    :return: cnm, caiman's cnmf object
    '''
    animal_path = os.path.split(TSeries)[0]
    T = os.path.split(TSeries)[-1] # name of TSeries 
    cnmf_path = os.path.join(animal_path, 'caiman_output', 'cnmf')
    cnmf_file = [os.path.join(cnmf_path,f) for f in os.listdir(cnmf_path) if (T in f)&('.hdf5' in f)]
    cnm = cnmf.load_CNMF(cnmf_file[0])
    #print(cnmf_file[0])
    return cnm

def get_fov_data(animal: str, FOV: str, file_key, base_path,suppress_print=True):
    '''

    get FOVs cnmf objects, registered indices from CellReg and summary images from all sessions
    :param animal: animal name, str
    :param FOV: fov name,str
    :param file_key: metadata csv for experiment
                    #Note: Session names should be alphabetized in temporal order i.e. session1,session2 or,
                     baseline1,baseline2,post1,post2. This ensures getting data for multiple sessions are in the right order.
    :param base_path: base directory for processed data
    :return: reg_ind: array, python equivalent of cell_to_index_map from CellReg output data.
                    N x M: N unique cells for all sessions, M sessions
             mean_list: (M,) list of mean images from each session
             std_list: (M,) list of std dev images from each session
             cnm_list:(M,) list of cnmf objects from each session

    '''
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

def mch_int(rois, image_arr, idx=None):
    '''
    get mcherry intensity for  rois

    rois: 3D array of footprints
        OR cnmf object from caiman
    image array: ndarray of mcherry image
    idx=None default, pass array or list of cell indices
    returns
        mch_f: N sized array of mcherry mean intensities for cells
    '''
    if type(rois) == cm.source_extraction.cnmf.cnmf.CNMF:
        if idx is None:
            masks = get_mask_rois(rois)
        else:
            masks = get_mask_rois(rois, idx)
    else:
        masks = rois.copy()
        if idx is not None:
            masks = rois[idx, :, :]

    mch_f = []
    for i in np.arange(masks.shape[0]):
        mask = masks[i, :, :]
        mch = np.mean(image_arr[mask > 0])
        mch_f.append(mch)
    return mch_f

def summary_images(TSeries,downsample=False,ds_factor=4):
    '''
        # Getting mean + std deviation images from cnmf output data

    :param TSeries: TSeries identifier, str
    :returns:
        mean_im: array, mean image
        std_im: array, std dev image
    '''
    if 'dview' in locals():
        cm.stop_server(dview=dview)
    c, dview, n_processes = cm.cluster.setup_cluster(backend='local', n_processes=None, single_thread=False)
    mmaps = [os.path.join(TSeries,f) for f in os.listdir(TSeries) if f.endswith('.mmap')]
    mmap_file = fnmatch.filter(mmaps,"*memmap__d1_*_d2_*_d3_1_order_C_frames_*.mmap")[0]
    Yr, dims, T = cm.load_memmap(mmap_file)
    images = np.reshape(Yr.T, [T] + list(dims), order='F') 
    if downsample:
        mean_im = np.mean(images,axis=0)
        std_im = np.std(images,axis=0)
    else:
        mean_im = np.mean(images[::ds_factor,:,:],axis=0)
        std_im = np.std(images[::ds_factor,:,:],axis=0)

    cm.stop_server(dview=dview)
    return mean_im, std_im

def save_summary_images(TSeries,mean_im,std_im,file_key,base_path):
    '''
    saves mean and standard deviation images as arrays in npz files
    :param TSeries: TSeries identifier, str
    :param mean_im: array, mean image
    :param std_im: array, std dev image
    :param file_key: metadata csv for experiment
    :param base_path: base directory for processed data.
                        must have 'CellReg/animal_fov subfolders.
    :return:
    '''
    info = pd.read_csv(file_key)
    T = os.path.split(TSeries)[-1]
    FOV = info['FOV'].loc[info['TSeries_g'] == T].values[0]
    ani = info['Animal'].loc[info['TSeries_g'] == T].values[0]
    subfolder = f"{base_path}{os.path.sep}CellReg{os.path.sep}{ani}_{FOV}"
    session =  info['Session'].loc[info['TSeries_g'] == T].values[0]
    np.savez(f"{subfolder}{os.path.sep}{session}_summary_images.npz", mean_im=mean_im,std_im=std_im)

def load_avg_img(TSeries,file_key,base_path):
    info = pd.read_csv(file_key)
    T = os.path.split(TSeries)[-1]
    FOV = info['FOV'].loc[info['TSeries_g'] == T].values[0]
    ani = info['Animal'].loc[info['TSeries_g'] == T].values[0]
    subfolder = f"{base_path}{os.path.sep}CellReg{os.path.sep}{ani}_{FOV}"
    session = info['Session'].loc[info['TSeries_g'] == T].values[0]

    images = np.load(f"{subfolder}{os.path.sep}{session}_summary_images.npz")
    return images['mean_im'],images['std_im']

def save_summary_images_batch(TSeries_all,file_key,base_path,downsample=False):
    '''
    saves for all TSeries
    :param TSeries_all: list of all TSeries to save
    :return:
    '''
    for TSeries in TSeries_all:
        print('calculating images for'+os.path.split(TSeries)[-1])
        mean_im,std_im=summary_images(TSeries,downsample=downsample)
        print('saving')
        save_summary_images(TSeries,mean_im,std_im,file_key,base_path)

