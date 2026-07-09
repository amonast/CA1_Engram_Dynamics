import warnings
warnings.filterwarnings("ignore", message=r"Passing", category=FutureWarning)
import caiman
import pandas as pd
import numpy as np
import os
import sys
sys.path.extend(['/projectnb/sramirezlab/amonast/Engram_2P/Engram_2P'])

try:
    from spatial import get_fov_data,remove_bad_cells
    from visualization.plotting import get_mask_rois
except ImportError:
    from .rois import get_fov_data,remove_bad_cells
    from .plotting import get_mask_rois  
from tqdm import tqdm
import tkinter.filedialog as fd
import tifffile

def load_mch_image(animal,fov,session,file_key,base_dir,channel='red'):
    """
    Load the registered mCherry (or GCaMP) mean-projection image for a session.

    Parameters
    ----------
    animal : str
        Animal identifier.
    fov : str
        FOV identifier.
    session : str
        Session identifier, matching a session name in `file_key`.
    file_key : str
        Path to the metadata CSV for the experiment.
    base_dir : str
        Base directory for the experiment, containing 'mCherry/{animal}'
        subfolders.
    channel : {'red', 'green'}, optional
        Which channel's mean projection to return: 'red' (mCherry) or
        'green' (GCaMP registered to mCherry) (default 'red').

    Returns
    -------
    ndarray
        Mean-projection image for the requested channel, loaded from the
        saved registration `.npz` file. If no `.npz` file is found, prompts
        the user to select a `.tif` file instead and returns that image
        (with NaNs replaced by 0).
    """
    metadata = pd.read_csv(file_key)
    TSeries_mch = metadata['TSeries_mch'].loc[(metadata['Animal']==animal)&(metadata['FOV']==fov)&(metadata['Session']==session)].values[0]
    TSeries_g = metadata['TSeries_g'].loc[(metadata['Animal']==animal)&(metadata['FOV']==fov)&(metadata['Session']==session)].values[0]
    mch_path = os.path.join(base_dir,'mCherry',animal,TSeries_mch)
    file = [os.path.join(mch_path,z) for z in os.listdir(mch_path) if (('movies.npz' in z)&('._' not in z))]
    try:
        with np.load(file[0]) as data:
            temp_red = data['temp_red']
            temp_green = data['temp_green']
            
        if channel=='red':
            return temp_red
        elif channel=='green':
            return temp_green
    except:
        print(f"No .npz file found, please choose mCherry tif image for {animal} {fov} {session}")
        file = fd.askopenfilename(initialdir=os.path.join(base_dir,'mCherry'))
        im = tifffile.imread(file)
        im = np.nan_to_num(im,0)
        return im

def mch_int(rois,image_arr,idx=None):
    """
    Compute mean mCherry (or other channel) intensity within each cell's ROI mask.

    Parameters
    ----------
    rois : ndarray or caiman.source_extraction.cnmf.cnmf.CNMF
        Either a 3D array of cell footprint masks, or a CaImAn CNMF object
        (in which case masks are computed via `get_mask_rois`).
    image_arr : ndarray
        Image (e.g. mCherry mean projection) to sample intensities from.
    idx : array-like of int, optional
        Subset of cell indices to compute intensities for; if None, uses all
        cells (default None).

    Returns
    -------
    mch_f : list of float
        Mean intensity within each cell's mask, in the order given by
        `masks`/`idx`.
    """
    if type(rois)==caiman.source_extraction.cnmf.cnmf.CNMF:
        if idx is None:
            masks = get_mask_rois(rois)
        else:
            masks = get_mask_rois(rois,idx)
    else:
        masks=rois.copy()
        if idx is not None:
            masks=rois[idx,:,:]

    mch_f=[]
    for i in np.arange(masks.shape[0]):
        mask = masks[i,:,:]
        mch = np.mean(image_arr[mask>0])
        mch_f.append(mch)
    return mch_f

### some deprecated functions
def mch_int_across_sessions(animal,fov,file_key,base_dir):
    """
    Compute raw mCherry intensity for all and registered cells across Baseline and Post sessions.

    Loads registered cell indices (filtered for quality), the red-channel
    mCherry mean projections for Baseline and Post, and computes per-cell
    mCherry intensity for all detected cells as well as for the subset of
    cells registered (matched) across both sessions.

    Parameters
    ----------
    animal : str
        Animal identifier.
    fov : str
        FOV identifier.
    file_key : str
        Path to the metadata CSV for the experiment.
    base_dir : str
        Base directory for the experiment.

    Returns
    -------
    mch_int_all : pandas.DataFrame
        Per-cell mCherry intensity for all detected cells (silent and
        registered) in each session, with columns 'mCherry Intensity (AU)',
        'Session', 'FOV', 'Animal', 'Group'.
    mch_int_reg : pandas.DataFrame
        Per-cell mCherry intensity for cells registered across both
        sessions, with columns 'Baseline', 'Post', 'Tagged',
        'mCherry Intensity (AU)', 'Session', 'ROI', 'FOV', 'Animal', 'Group'.
    """
    #get_indices
    _,_,_,cnm_list = get_fov_data(animal,fov,file_key,base_dir)
    ind_df = remove_bad_cells(animal,fov,snr_thr=4.0,file_key=file_key,base_dir=base_dir) #filter out the bad cells
    silent_pre = ind_df['Baseline'].loc[ind_df['Post']==-1].values
    silent_post = ind_df['Post'].loc[ind_df['Baseline']==-1].values
    reg_pre = ind_df['Baseline'].loc[(ind_df['Baseline']!=-1)&(ind_df['Post']!=-1)]
    reg_post = ind_df['Post'].loc[(ind_df['Baseline']!=-1)&(ind_df['Post']!=-1)]
    pre_all=np.append(silent_pre,reg_pre)
    post_all = np.append(silent_post,reg_post)
    
    #get mcherry red channel registered mean projections
    red_pre = load_mch_image(animal,fov,'Baseline',file_key,base_dir,channel='red')
    red_post = load_mch_image(animal,fov,'Post',file_key,base_dir,channel='red')
    
    #registered cells only - get tagged label
    r_df = ind_df.loc[(ind_df['Baseline']!=-1)&(ind_df['Post']!=-1)]
    #get tagged or not
    tagged = r_df['Tagged']
    
    #get mcherry intensities 
    mch_f_pre = mch_int(cnm_list[0],red_pre,idx=pre_all)
    mch_f_post = mch_int(cnm_list[1],red_post,idx=post_all)
    mch_f_pre_reg = mch_int(cnm_list[0],red_pre,idx=reg_pre)
    mch_f_post_reg = mch_int(cnm_list[1],red_post,idx=reg_post)

    df1=pd.DataFrame()
    df1['mCherry Intensity (AU)']= mch_f_pre
    df1['Session']='Baseline'
    df2=pd.DataFrame()
    df2['mCherry Intensity (AU)']=mch_f_post 
    df2['Session']='Post'
    mch_int_all = pd.concat([df1,df2],ignore_index=True)
    mch_int_all['FOV']=fov
    
    df3=pd.DataFrame()
    df3['Baseline']=reg_pre
    df3['Post']=reg_post
    df3['Tagged']=tagged
    df3['mCherry Intensity (AU)']= mch_f_pre_reg
    df3['Session']='Baseline'
    df3['ROI']=reg_pre
    df4=pd.DataFrame()
    df4['Baseline']=reg_pre
    df4['Post']=reg_post
    df4['Tagged']=tagged
    df4['mCherry Intensity (AU)']=mch_f_post_reg
    df4['Session']='Post'
    df4['ROI']=reg_post
    mch_int_reg = pd.concat([df3,df4],ignore_index=True)
    mch_int_reg['FOV']=fov
    
    metadata = pd.read_csv(file_key)
    group = metadata['Group'].loc[metadata['Animal']==animal].values[0]
    mch_int_all['Animal']=[animal]*mch_int_all.shape[0]
    mch_int_all['Group']=[group] * mch_int_all.shape[0]
    
    mch_int_reg['Animal']=[animal]*mch_int_reg.shape[0]
    mch_int_reg['Group']=[group] * mch_int_reg.shape[0]
    return mch_int_all, mch_int_reg

def mch_int_across_sessions_C1(animal,fov,file_key,base_dir):
    """
    Deprecated: compute raw mCherry intensity for cells across sessions in the first cohort of mice.

    Uses an older, cohort-1-specific file layout (manual silent-cell CSVs
    and Fiji ROI dataframes) rather than the standard registration/tagging
    pipeline used by `mch_int_across_sessions`.

    Parameters
    ----------
    animal : str
        Animal identifier.
    fov : str
        FOV identifier.
    file_key : str
        Path to the metadata CSV for the experiment.
    base_dir : str
        Base directory for the experiment.

    Returns
    -------
    mch_int_all : pandas.DataFrame
        Per-cell mCherry intensity for all detected cells (silent and
        registered) in each session.
    mch_int_reg : pandas.DataFrame
        Per-cell mCherry intensity for cells registered across both
        sessions.

    Notes
    -----
    Deprecated in favor of `mch_int_across_sessions`.
    """
    #get registered cells PRE
    _,_,_,cnm_list = get_fov_data(animal,fov,file_key,base_dir)

    file =[f for f in os.listdir(os.path.join(base_dir,'ROIs','Manual assess - silent cells')) if ((animal in f) & (fov in f)) & ('lone_ind.csv' in f)]
    df = pd.read_csv(os.path.join(base_dir,'ROIs','Manual assess - silent cells',file[0]))
    silent_pre = df['Baseline'].loc[(df['Scores']==2)&(df['Post']==-1)].values
    
    #get registered cells PRE 
    reg_ind_path = os.path.join(base_dir,'Tagging','indices')
    file = [os.path.join(reg_ind_path,f) for f in os.listdir(reg_ind_path) if (animal in f)&(fov in f)&('reg_split.csv' in f)][0]
    r_df = pd.read_csv(file)
    reg_pre = r_df['Baseline'].values
    pre_all = np.append(silent_pre,reg_pre)

    #get silent cells POST & reg cells post 
    df_post_all = pd.read_csv(os.path.join(base_dir,'Tagging','Fiji','Roi_dataframes',animal+'_'+fov+"_Post_manual_split.csv"))
    post_all = df_post_all['roi'].values

    #get reg cells POST 
    reg_post = r_df['Post'].values

    #get mcherry red channel registered mean projections
    red_pre = load_mch_image(animal,fov,'Baseline',file_key,base_dir,channel='red')
    red_post = load_mch_image(animal,fov,'Post',file_key,base_dir,channel='red')
    #get tagged or not
    tagged = r_df['tagged']
    
    #get mcherry intensities 
    mch_f_pre = mch_int(cnm_list[0],red_pre,idx=pre_all)
    mch_f_post = mch_int(cnm_list[1],red_post,idx=post_all)
    mch_f_pre_reg = mch_int(cnm_list[0],red_pre,idx=reg_pre)
    mch_f_post_reg = mch_int(cnm_list[1],red_post,idx=reg_post)

    df1=pd.DataFrame()
    df1['mCherry Intensity (AU)']= mch_f_pre
    df1['Session']='Baseline'
    df2=pd.DataFrame()
    df2['mCherry Intensity (AU)']=mch_f_post 
    df2['Session']='Post'
    mch_int_all = pd.concat([df1,df2],ignore_index=True)
    mch_int_all['FOV']=fov
    
    df3=pd.DataFrame()
    df3['Baseline']=reg_pre
    df3['Post']=reg_post
    df3['Tagged']=tagged
    df3['mCherry Intensity (AU)']= mch_f_pre_reg
    df3['Session']='Baseline'
    df3['ROI']=reg_pre
    df4=pd.DataFrame()
    df4['Baseline']=reg_pre
    df4['Post']=reg_post
    df4['Tagged']=tagged
    df4['mCherry Intensity (AU)']=mch_f_post_reg
    df4['Session']='Post'
    df4['ROI']=reg_post
    mch_int_reg = pd.concat([df3,df4],ignore_index=True)
    mch_int_reg['FOV']=fov
    
    metadata = pd.read_csv(file_key)
    group = metadata['Group'].loc[metadata['Animal']==animal].values[0]
    mch_int_all['Animal']=[animal]*mch_int_all.shape[0]
    mch_int_all['Group']=[group] * mch_int_all.shape[0]
    
    mch_int_reg['Animal']=[animal]*mch_int_reg.shape[0]
    mch_int_reg['Group']=[group] * mch_int_reg.shape[0]
    return mch_int_all, mch_int_reg

def mch_int_sessions_animals(animals,fov_lists,file_key,base_dir):
    """
    Compute mCherry intensity across sessions for multiple animals and FOVs.

    Parameters
    ----------
    animals : list of str
        Animal identifiers.
    fov_lists : list of list of str
        FOV identifiers per animal; `fov_lists[i]` gives the FOVs for
        `animals[i]`.
    file_key : str
        Path to the metadata CSV for the experiment.
    base_dir : str
        Base directory for the experiment.

    Returns
    -------
    df : pandas.DataFrame
        Concatenated per-cell mCherry intensity (all detected cells) across
        all animals/FOVs, as returned by `mch_int_across_sessions`.
    df_reg : pandas.DataFrame
        Concatenated per-cell mCherry intensity (registered cells only)
        across all animals/FOVs.
    """
    df=pd.DataFrame()
    df_reg = pd.DataFrame()
    for a,ani in tqdm(enumerate(animals)):
        for f in fov_lists[a]:
            mch_int_all,mch_int_reg = mch_int_across_sessions(ani,f,file_key,base_dir)
            df = pd.concat([df,mch_int_all],ignore_index=True)
            df_reg = pd.concat([df_reg,mch_int_reg],ignore_index=True)
    return df,df_reg