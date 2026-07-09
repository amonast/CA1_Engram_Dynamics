import os
import sys
from tkinter import filedialog
import pickle
import pandas as pd
import tifffile
import matplotlib.pyplot as plt
import numpy as np
import warnings
warnings.filterwarnings("ignore", message=r"Passing", category=FutureWarning)
import caiman as cm
from caiman.motion_correction import MotionCorrect
os.environ['CAIMAN_TEMP'] = '.'

__all__ = ["get_gcamp_temp","reg_mcherry","reg_mcherry_batch"]
#################################
def main():
    """
    Register a single mCherry (1040 nm) TSeries to its GCaMP template via CLI/script config.

    Depending on `batch`, either runs `reg_mcherry_batch` for a hardcoded list
    of animals, or registers a single TSeries (path from `sys.argv[1]`) to a
    hardcoded GCaMP template file.

    Parameters
    ----------
    None
        Reads the mCherry TSeries path from `sys.argv[1]` when not in batch
        mode.

    Returns
    -------
    None

    Notes
    -----
    `file_key`, `base_dir`, `batch`, and `temp_file` are hardcoded within the
    function body.
    """
    file_key='/projectnb/sramirezlab/amonast/Tone2P/Data_info_TFC.csv'
    base_dir = '/projectnb/sramirezlab/amonast/Tone2P'
    batch = False# if sys.argv[1]=='batch' else False
    if batch:
        reg_mcherry_batch(base_dir,file_key,animals=['M8BL2'],
                          new_params={'max_deviation_rigid':2,'pw_rigid':True, 'max_shifts':(20,20),
                                'strides':(84, 84), 'overlaps':(18, 18)},append='Ch1')
    else:
        #TSeries_1040 = filedialog.askdirectory(initialdir=os.path.join(base_dir,'mCherry'))
        TSeries_1040 = sys.argv[1]
        #temp_file = get_gcamp_temp(TSeries_1040,file_key,base_dir,save_template=True)[0]
        temp_file='/projectnb/sramirezlab/amonast/Tone2P/GCaMP/939L/TSeries-05122025-037/suite2p/TSeries-05122025-037-AVG-template.tif'
        #filedialog.askopenfilename(initialdir=os.path.join(base_dir,'mCherry'))
        os.chdir(TSeries_1040)
        reg_mcherry(TSeries_1040,file_key,base_dir,play_movie=False,ds_ratio=.5,temp_file=temp_file,
                    new_params={'max_deviation_rigid':5,'pw_rigid':True, 'max_shifts':(50,50),
                                'strides':(96, 96), 'overlaps':(24, 24)},append='Ch1')
#############################

def get_gcamp_temp(TSeries_1040,file_key,base_dir,save_template=False):
    """
    Locate the saved GCaMP motion-correction template/opts files matching an mCherry TSeries.

    Parameters
    ----------
    TSeries_1040 : str
        Path to the mCherry (1040 nm) TSeries folder; used to look up the
        corresponding animal and GCaMP TSeries via `file_key`.
    file_key : str
        Path to the metadata CSV for the experiment.
    base_dir : str
        Base directory containing 'GCaMP/{animal}/caiman_output/motion_correct'.
    save_template : bool, optional
        If True, also save the GCaMP template image as a `.tif` file in
        `TSeries_1040` (default False).

    Returns
    -------
    temp_file : str
        Path to the matching GCaMP `.mc.pkl` file.
    opts_file : str
        Path to the matching GCaMP `.mc_opts.pkl` file.
    """
    info=pd.read_csv(file_key)
    TSeries_mch = TSeries_1040.split(os.path.sep)[-1]
    ani =  info['Animal'].loc[info['TSeries_mch'] == TSeries_mch].values[0]
    session =  info['Session'].loc[info['TSeries_mch'] == TSeries_mch].values[0]
    T_df = info['TSeries_g'].loc[(TSeries_mch==info['TSeries_mch'])]
    TSeries_g = T_df.values[0]
    mc_base = os.path.join(base_dir,'GCaMP', ani, 'caiman_output', 'motion_correct')

    mc_path = [os.path.join(mc_base,f) for f in os.listdir(mc_base) if TSeries_g in f]
    if not os.path.isdir(mc_base):
        print('GCaMP motion correct path doesnt exist. Check folder structure.')

    temp_file = [f for f in mc_path if 'mc.pkl' in f][0]
    opts_file = [f for f in mc_path if 'mc_opts.pkl' in f][0]

    with open(temp_file,'rb') as file:
        mc=pickle.load(file)
    try:
        template = mc.total_template_els
    except AttributeError:
        template = mc.total_template_rig

    if save_template:
        tifffile.imwrite(os.path.join(TSeries_1040,os.path.split(temp_file)[-1].split('_')[1])+'_template.tif',template)

    return temp_file, opts_file

def reg_mcherry_batch(base_dir,file_key,animals=None,new_params=None,append='red_green'):
    """
    Run mCherry-to-GCaMP registration for all mCherry TSeries across one or more animals.

    Parameters
    ----------
    base_dir : str
        Base directory containing the 'mCherry/{animal}' subfolders.
    file_key : str
        Path to the metadata CSV for the experiment.
    animals : list of str, optional
        Animal identifiers to process; if None, uses every subfolder found
        directly under `base_dir` (default None).
    new_params : dict, optional
        Motion correction parameter overrides (keys 'max_shifts',
        'max_deviation_rigid', 'strides', 'overlaps', 'pw_rigid') passed
        through to `reg_mcherry` (default None, uses parameters from the
        GCaMP motion correction reference).
    append : str, optional
        Channel filename suffix convention passed through to `reg_mcherry`
        (default 'red_green').

    Returns
    -------
    None

    Notes
    -----
    For each TSeries, calls `reg_mcherry` with `plot_reg=False` and
    `save_reg=True`.
    """
    df = pd.read_csv(file_key)
    if animals == None:
        animals = [os.path.join(base_dir,ani) for ani in os.listdir(base_dir)]
    else:
        animals = [os.path.join(base_dir,'mCherry',ani) for ani in animals]
    for a in animals:
        os.makedirs(a,exist_ok=True)
        ani = os.path.split(a)[-1]
        series_all = df['TSeries_mch'].loc[(df['Animal']==ani)].values
        TSeries_all = [os.path.join(base_dir,'mCherry',ani,series) for series in series_all]
        TSeries_all.sort()
        for T in TSeries_all:
            os.chdir(T)
            print('Processing '+ T)
            reg_mcherry(T,file_key,base_dir,plot_reg=False,save_reg=True,new_params=new_params,append=append)

def reg_mcherry(TSeries,file_key,base_dir,temp_file=None,plot_reg:bool=False,save_reg:bool=True,
                play_movie:bool=False,ds_ratio=.5,new_params:dict=None,append='red_green'):
    """
    Motion-correct an mCherry TSeries's green channel to a GCaMP template, then apply the same shifts to the red (mCherry) channel.

    Parameters
    ----------
    TSeries : str
        Path to the mCherry TSeries subfolder; should contain the green and
        red channel TIFFs (named by `append` convention).
    file_key : str
        Path to the metadata CSV for the experiment.
    base_dir : str
        Base directory for the experiment, containing 'mCherry/{animal}'
        subfolders.
    temp_file : str, optional
        Path to an alternative template file to use instead of the 920 nm
        GCaMP motion-correction reference: a `.pkl` (saved `mc` object) or
        `.tif` (template image) file. If given, `new_params` must also be
        provided. If None, the template and parameters are loaded from the
        matching GCaMP motion correction output via `get_gcamp_temp`
        (default None).
    plot_reg : bool, optional
        If True, plot the 920 nm template alongside the registered 1040 nm
        GCaMP and mCherry channels (default False).
    save_reg : bool, optional
        If True, save the motion correction object, registration figure, and
        both channel template images/movies (default True).
    play_movie : bool, optional
        If True, play the registered green/red movies concatenated side by
        side (default False).
    ds_ratio : float, optional
        Downsampling ratio used when playing movies (default 0.5).
    new_params : dict, optional
        Motion correction parameters (keys 'max_shifts',
        'max_deviation_rigid', 'strides', 'overlaps', 'pw_rigid') to use
        instead of the GCaMP template's motion correction reference (default
        None).
    append : str, optional
        Channel filename suffix convention: 'red_green' looks for files
        ending in '_red.tif'/'_green.tif'; any other value looks for
        '_Ch1.tif'/'_Ch2.tif' (default 'red_green').

    Returns
    -------
    m_red : caiman.movie, shape (T, d1, d2)
        Red (mCherry) channel movie, registered to the GCaMP template.
    temp_red : ndarray, shape (d1, d2)
        Red channel mean projection.
    m_green : caiman.movie, shape (T, d1, d2)
        Green channel movie, registered to the GCaMP template.
    temp_green : ndarray, shape (d1, d2)
        Green channel mean projection after registration to the GCaMP
        template.
    mc_mcherry : caiman.motion_correction.MotionCorrect
        Motion correction object used to register the 1040 nm data to the
        920 nm template.

    Notes
    -----
    Not tested with multiple Ch1/Ch2 files in a single TSeries subfolder.
    """
    path = TSeries.split(os.path.sep)[-1]

    if temp_file is None:
        temp_file,opts_file = get_gcamp_temp(TSeries,file_key,base_dir)
        with open(temp_file,'rb') as file:
            mc=pickle.load(file)
        template = mc.total_template_els
        # load motion correction params +  motion correct to 920 data template
        with open(opts_file,'rb') as file:
            opts=pickle.load(file)
    else:
        if temp_file.endswith('.pkl'):
            with open(temp_file,'rb') as file:
                mc=pickle.load(file)
            template = mc.total_template_els if mc.pw_rigid==True else mc.total_template_rig
        elif temp_file.endswith('.tif'):
            template = tifffile.imread(temp_file)

    #%%  load color  data
    if append!='red_green':
        green_files = [os.path.join(TSeries,f) for f in os.listdir(TSeries) if "_Ch2.tif" in f]
        red_files = [os.path.join(TSeries,f) for f in os.listdir(TSeries) if "_Ch1.tif" in f]
    else:
        green_files = [os.path.join(TSeries,f) for f in os.listdir(TSeries) if "_green.tif" in f]
        red_files = [os.path.join(TSeries,f) for f in os.listdir(TSeries) if "_red.tif" in f]
    #%% start cluster
    if 'dview' in locals():
        cm.stop_server(dview=dview)
    c, dview, n_processes = cm.cluster.setup_cluster(backend='local', n_processes=None, single_thread=False)
    #%%
    # use parameters from gcamp motion correction if none specificied in input
    if new_params is None:
        try:
            mc_mcherry = MotionCorrect(green_files,dview=dview, max_shifts=mc.max_shifts, pw_rigid=mc.pw_rigid,
                      strides=mc.strides, overlaps=mc.overlaps,
                      max_deviation_rigid=mc.max_deviation_rigid,
                      shifts_opencv=mc.shifts_opencv, nonneg_movie=True,
                      border_nan=mc.border_nan)
        except NameError:
            print("mc object is not defined, pass new_params dict with motion correct params dict")
    
    else: # else use the new params defined in input args 
        pw_rigid = new_params['pw_rigid']
        max_shifts = new_params['max_shifts']
        max_deviation_rigid =new_params['max_deviation_rigid']
        strides = new_params['strides']
        overlaps = new_params['overlaps']

        mc_mcherry = MotionCorrect(green_files,dview=dview, max_shifts=max_shifts, pw_rigid=pw_rigid,
                      strides=strides, overlaps=overlaps,max_deviation_rigid=max_deviation_rigid,nonneg_movie=True,border_nan='copy')
    print('motion correcting template channel')    
    mc_mcherry.motion_correct(template=template,save_movie=True)

    if mc_mcherry.pw_rigid:
        append = "_els_"
    else:
        append="_rig_"

    base_name = red_files[0].split(os.path.sep)[-1].split('.')[-2]+ append
    #%%
    print('applying shifts to second channel')
    mmap_red = mc_mcherry.apply_shifts_movie(red_files, save_memmap=True, order='F',
                            save_base_name=os.path.join(os.path.split(red_files[0])[0],base_name))
    #%%
    # load green movie + calculate average images
    if mc_mcherry.pw_rigid == True:
        m_green = cm.load(mc_mcherry.fname_tot_els)
    elif mc_mcherry.pw_rigid == False:
        m_green = cm.load(mc_mcherry.fname_tot_rig)

    temp_green=m_green.mean(0)
    m_red = cm.load(mmap_red)
    temp_red=m_red.mean(0)
    if play_movie:
        cm.concatenate([m_green.resize(1, 1, ds_ratio),
                        m_red.resize(1, 1, ds_ratio)], axis=2).play(fr=60, gain=1, magnification=1,
                                                                    offset=0)  # press q to exit
    #%%
    if plot_reg:
        fig, ax = plt.subplots(nrows=1, ncols=3,figsize=(15,5))
        ax[0].imshow(template,cmap='gist_gray')
        ax[0].axis('off')
        ax[0].set_title('920 nm GCaMP Template')
        ax[1].imshow(temp_green,cmap='gist_gray')
        ax[1].set_title('1040 nm GCaMP')
        ax[1].axis('off')
        ax[2].imshow(temp_red,cmap='gist_gray')
        ax[2].set_title('1040 nm mCherry')
        ax[2].axis('off')
        fig.tight_layout()
        if save_reg:
            fig.savefig(os.path.join(TSeries, TSeries.split(os.path.sep)[-1] + '_mChReg.png'))

    #%%
    cm.stop_server(dview=dview) #  stop  server
    #%%
    if save_reg:
        mchreg_file = os.path.join(TSeries, base_name)

        if hasattr(mc_mcherry,'dview'):
            del(mc_mcherry.dview)
        with open(mchreg_file+'mc.pkl','wb') as file:
            pickle.dump(mc_mcherry,file,protocol=pickle.HIGHEST_PROTOCOL)
        
        try:
            if mc in locals():
                tifffile.imwrite(os.path.join(TSeries,os.path.split(temp_file)[-1].split('_')[1])+'_template.tif',template)
        except UnboundLocalError:
            tifffile.imwrite(os.path.join(TSeries,os.path.split(temp_file)[-1].split('.')[0])+'_template.tif',template)
        tifffile.imwrite(mchreg_file+'.tif',temp_red)
        tifffile.imwrite(os.path.join(TSeries,green_files[0].split('.')[0].split(os.path.sep)[-1]+append+'.tif'),temp_green)

        np.savez(mchreg_file+'movies.npz',m_red=m_red,temp_red=temp_red,m_green=m_green,temp_green=temp_green)
    return m_red,temp_red,m_green,temp_green,mc_mcherry
#%%
if __name__=='__main__':
    main()