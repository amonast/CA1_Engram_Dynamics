import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
from tkinter import filedialog
from datetime import datetime
import pickle
import tifffile
import shutil
try:
    cv2.setNumThreads(0)
except():
    pass
try:
    if __IPYTHON__:
        # this is used for debugging purposes only. allows to reload classes
        # when changed
        get_ipython().magic('load_ext autoreload')
        get_ipython().magic('autoreload 2')
except NameError:
    pass
import warnings
warnings.filterwarnings("ignore", message=r"Passing", category=FutureWarning)
import caiman as cm
from caiman.motion_correction import MotionCorrect
from caiman.source_extraction.cnmf import params as params
import pandas as pd
os.environ['CAIMAN_TEMP']='.'
__all__=["load_mc","get_filelist","get_TSeries","get_template_baseline","motion_correct","motion_correct_2color_single"]
##############
def main():
    """
    Motion-correct a hardcoded example TSeries folder as a standalone demo.
    """
    imagespath= '/Users/amonast/Desktop/astro2p/GCaMP/TSeries-08302024--044'
    os.chdir(imagespath)
    tifs = get_filelist(imagespath,'tif')
    fnames = [os.path.join(imagespath,f) for f in tifs]
    fnames.sort()
    print(fnames)
    motion_correct(fnames,template=None,pw_rigid=True,display_movie=True,compute_metrics=False)
################
def get_mc_pkls(ani, fov, file_key, base_dir):
    """
    Map each TSeries name to its saved motion-correction `.mc.pkl` file.

    Parameters
    ----------
    ani : str
        Animal identifier.
    fov : str or int
        Field-of-view identifier.
    file_key : str
        Path to the metadata CSV containing TSeries names.
    base_dir : str
        Base directory containing
        'GCaMP/{ani}/caiman_output/motion_correct'.

    Returns
    -------
    mc_dict : dict
        Dictionary mapping each TSeries name to the path of its
        corresponding `.mc.pkl` file.
    """
    path = os.path.join(base_dir, 'GCaMP', ani, 'caiman_output', 'motion_correct')
    
    # Load CSV file
    info = pd.read_csv(file_key)
    
    # Filter TSeries_g names based on the given animal and FOV
    TSeries_names = info.loc[(info['Animal'] == ani) & (info['FOV'] == fov), 'TSeries_g'].values
    
    # Get list of all files in the directory
    fnames = os.listdir(path)

    # Create a dictionary mapping TSeries_g to its corresponding .mc.pkl file
    mc_dict = {ts: os.path.join(path, mc) for mc in fnames 
        for ts in TSeries_names if mc.endswith('mc.pkl') and ts in mc}
    
    return mc_dict

def load_mc(pklfile):
    """
    Load a pickled CaImAn motion correction object.

    Parameters
    ----------
    pklfile : str
        Path to the pickled `mc` object file.

    Returns
    -------
    mc : caiman.motion_correction.MotionCorrect
        The unpickled motion correction object.
    """
    with open(pklfile,'rb') as file:
        mc=pickle.load(file)

    return mc

def get_filelist(path,filetype):
    """
    List files in a directory whose name contains a given substring.

    Parameters
    ----------
    path : str
        Directory to list.
    filetype : str
        Substring to match in filenames (e.g. a file extension like 'tif').

    Returns
    -------
    filelist : list of str
        Filenames in `path` containing `filetype`.
    """
    filelist = [f for f in os.listdir(path) if filetype in f]
    return filelist

def get_TSeries(path):
    """
    Recursively find subdirectories whose path contains "TSeries".

    Parameters
    ----------
    path : str
        Root directory to search.

    Returns
    -------
    TSeries : list of str
        Paths of all subdirectories (at any depth under `path`) whose path
        contains the substring "TSeries".
    """
    subdirs=[]
    for subdir, dir, files in os.walk(path):
        subdirs.append(subdir)
    TSeries=[T for T in subdirs if "TSeries" in T]

    return TSeries

#####################
def get_template_baseline(TSeries_path,template_session_id,file_key):
    """
    Load the saved motion-correction template and parameters from a reference session.

    Parameters
    ----------
    TSeries_path : str
        Path to a TSeries folder with images; used to identify the animal
        and FOV via `file_key`.
    template_session_id : str or int
        Session name/identifier (as it appears in `file_key`) to use as the
        template/reference session.
    file_key : str
        Path to the metadata CSV for the experiment.

    Returns
    -------
    template : ndarray
        Motion correction template image (`total_template_els` if
        piecewise-rigid, otherwise `total_template_rig`) from the reference
        session's saved `mc` object.
    opts : caiman.source_extraction.cnmf.params.CNMFParams
        Motion correction parameters object from the reference session.
    """
    info=pd.read_csv(file_key)
    TSeries = TSeries_path.split(os.path.sep)[-1]
    ani = info['Animal'].loc[info['TSeries_g'] == os.path.split(TSeries)[-1]].values[0]
    fov = info['FOV'].loc[(info['Animal'] == ani) & (info['TSeries_g'] == os.path.split(TSeries)[-1])].values[0]

    TSeries_template = info['TSeries_g'].loc[(info['Session']==template_session_id)
                                             &(info['Animal']==ani)
                                             &(info['FOV']==fov)].values[0]
    mc_base = os.path.join(os.path.split(TSeries_path)[0], 'caiman_output', 'motion_correct')
    os.makedirs(mc_base,exist_ok=True)
    mc_files = [os.path.join(mc_base,f) for f in os.listdir(mc_base) if TSeries_template in f]
    

    temp_file = [f for f in mc_files if ('mc.pkl' in f)&(TSeries_template in f)][0]
    print("Template file: "+temp_file)
    opts_file = [f for f in mc_files if ('mc_opts.pkl' in f)&(TSeries_template in f)][0]

    with open(temp_file, 'rb') as file:
        mc = pickle.load(file)
    try:
        template = mc.total_template_els
    except:
        template = mc.total_template_rig
    with open(opts_file, 'rb') as file:
        opts = pickle.load(file)

    return template,opts

############################################
def motion_correct(fnames, template=None,pw_rigid=True,
                   display_movie:bool=False,save_avi:bool=False,
                   compute_metrics:bool=True,save_template:bool=False,
                   save_mean_proj:bool=True,mc_opts:dict=None,move_mmaps:bool=False):
    """
    Run CaImAn motion correction on a list of TIFF/mmap files and save the results.

    Sets up a local CaImAn processing cluster, runs rigid or piecewise-rigid
    NoRMCorre motion correction on `fnames` (optionally registering to a
    given `template`), and saves the corrected movie, mean projection,
    and/or template as needed, along with pickled `mc` and `opts` objects.
    Optionally computes and plots registration quality metrics.

    Parameters
    ----------
    fnames : list of str
        Filenames to motion-correct, in acquisition order.
    template : ndarray, optional
        Template image to register to; if None, CaImAn builds its own
        template from the data (default None).
    pw_rigid : bool, optional
        If True, use piecewise-rigid (non-rigid) motion correction; if
        False, use rigid correction (default True).
    display_movie : bool, optional
        If True, display the raw and corrected movies via OpenCV/CaImAn
        playback (default False).
    save_avi : bool, optional
        If True, save the corrected movie as an `.avi` file (default False).
    compute_metrics : bool, optional
        If True, compute and plot registration quality metrics (correlation
        images, shift traces, crispness); this is slow (default True).
    save_template : bool, optional
        If True, save the resulting motion-correction template as a `.tif`
        file (default False).
    save_mean_proj : bool, optional
        If True, save the mean projection of the corrected movie as a
        `.tif` file (default True).
    mc_opts : dict, optional
        Dictionary with keys 'strides', 'overlaps', 'max_shifts', and
        'max_deviation_rigid' overriding the default motion correction
        parameters (default None, uses built-in defaults).
    move_mmaps : bool, optional
        If True, move the resulting `.mmap` files into the per-run output
        subfolder (default False).

    Returns
    -------
    mc : caiman.motion_correction.MotionCorrect
        The motion correction object after running correction.
    opts : caiman.source_extraction.cnmf.params.CNMFParams
        The parameters object used for motion correction.

    Notes
    -----
    Saves outputs under `{parent_of(fnames[0])}/caiman_output/motion_correct`
    and a per-run subfolder `MC_{TSeries}_{timestamp}`, including pickled
    `mc` and `opts` objects (`*_mc.pkl`, `*_mc_opts.pkl`).
    """
#set up data and paths     
    path =os.path.split(fnames[0])[0]
    display_movie = display_movie

    animal=path.split(os.path.sep)[-2]
    TSer = path.split(os.path.sep)[-1]
    output_path = os.path.join(os.path.split(path)[0],'caiman_output','motion_correct')
    
    os.makedirs(os.path.join(os.path.split(path)[0], 'caiman_output', 'motion_correct'),exist_ok=True)


#%% dataset dependent parameters
        # dataset dependent parametersqqq
    fr = 30                             # imaging rate in frames per second
    decay_time = 0.4                    # length of a typical transient in seconds

    # motion correction parameters
    if mc_opts is None:
        strides = (96,96)          # start a new patch for pw-rigid motion correction every x pixels
        overlaps = (24, 24)         # overlap between pathes (size of patch strides+overlaps)
        max_shifts = (30,30)          # maximum allowed rigid shifts (in pixels)
        max_deviation_rigid = 2   # maximum shifts deviation allowed for patch with respect to rigid shifts
    else:
        strides = mc_opts['strides']
        overlaps=mc_opts['overlaps']
        max_shifts = mc_opts['max_shifts']
        max_deviation_rigid=mc_opts['max_deviation_rigid']
    
    pw_rigid = pw_rigid             # flag for performing non-rigid motion correction
    opts_dict = {'fnames': fnames,
                'fr': fr,
                'decay_time': decay_time,
                'strides': strides,
                'overlaps': overlaps,
                'max_shifts': max_shifts,
                'max_deviation_rigid': max_deviation_rigid,
                'pw_rigid': pw_rigid}


    opts = params.CNMFParams(params_dict=opts_dict)
    opts.data['path']=path
    opts.data['animal']=animal
    opts.data['TSeries']=TSer
#%% play original movie 
    if display_movie:
        m_orig = cm.load_movie_chain(fnames)
        ds_ratio = .01  # motion can be perceived better when downsampling in time
        m_orig.resize(1, 1, ds_ratio).play(q_max=99.5, fr=30, magnification=2)   # play movie (press q to exit)

#%% start the cluster (if a cluster already exists terminate it)
    if 'dview' in locals():
        cm.stop_server(dview=dview)
    c, dview, n_processes = cm.cluster.setup_cluster(backend='local', n_processes=None, single_thread=False)

    #%%
    mctype ='Piecewise Rigid' if pw_rigid else 'Rigid'
    print('Performing '+mctype+' Motion Correction')
    os.chdir(path)
    # first we create a motion correction object with the parameters specified
    mc = MotionCorrect(fnames, dview=dview, **opts.get_group('motion')) # note that the file is not loaded in memory

    #%% Run piecewise-rigid motion correction using NoRMCorre
    if template is None:
        mc.motion_correct(save_movie=True)
    else:
        mc.motion_correct(save_movie=True,template=template)
    border_to_0 = 0 if mc.border_nan is 'copy' else mc.border_to_0 # maximum shift to be used for trimming against NaNs
    print('Finished Motion Correction')

#%% Save Results
    print('Saving results')
    now = datetime.now()
    timestamp = str(now.month) + str(now.day) + str(now.year) + '_' + str(now.hour) + str(now.minute) + str(now.second)
    
    mmap_output_folder = os.path.join(path,'MC_'+TSer+'_'+timestamp)
    os.makedirs(mmap_output_folder,exist_ok=True)

#save video
    if (save_avi==False) & (save_mean_proj==False):
        pass
    else:
        if mc.pw_rigid==True:
            mmap_files = mc.fname_tot_els
        else:
            mmap_files = mc.fname_tot_rig

    if (save_avi==True) or (save_mean_proj==True):
        mc_movie = cm.load_movie_chain(mmap_files)
        if save_avi==True:
            mc_movie.save(os.path.join(mmap_output_folder, animal + '_' + TSer + '_' + timestamp + '_mc_movie.avi'))
        #save mean projection    
        if save_mean_proj:
            mean_proj = mc_movie.mean(axis=0)
            tifffile.imwrite(os.path.join(mmap_output_folder, animal + '_' + TSer + '_' + timestamp + '_mc_mean_proj.tif'),mean_proj)
    
    #save template 
    if save_template:
        if mc.pw_rigid==True:
            template_mc=mc.total_template_els
            append='els'
        elif mc.pw_rigid==False:
            template_mc=mc.total_template_rig
            append='rig'
        tifffile.imwrite(os.path.join(mmap_output_folder, animal + '_' + TSer + '_' + timestamp + '_template.tif'),template_mc)
    
    
    # output_path = os.path.join(opts.data['path'].split(os.path.sep)[:-1])+os.path.sep+'caiman_output'+os.path.sep+'motion_correct'
    # if not os.path.isdir(output_path):
    #     os.mkdir(os.path.join(opts.data['path'].split(os.path.sep)[:-1]),'caiman_output')
    #     os.mkdir(os.path.join(opts.data['path'].split(os.path.sep)[:-1]),'caiman_output','motion_correct)

  # move the mmap files!
    if move_mmaps:
        print('moved files to subfolder')
        for mmap in mmap_files:
            source_path = os.path.join(path, mmap)
            destination_path = os.path.join(mmap_output_folder, mmap)
        
            if os.path.exists(source_path):  # Check if the source file exists
                shutil.move(source_path, destination_path)
                print(f"Moved: {source_path} -> {destination_path}")
        opts.data['path']=mmap_output_folder


#  save the pkl file with Motion correction parameters (opss)
    savefile = output_path + os.path.sep + animal + '_' + TSer + '_' + timestamp + '_mc_opts.pkl'
    with open(savefile, 'wb') as file:
        pickle.dump(opts, file, protocol=pickle.HIGHEST_PROTOCOL)
    # to save mc object, must remove the parallel processing pool
    if hasattr(mc, 'dview'):
        del (mc.dview)
#  save the pkl file with MotionCorrection object (mc)
    savefile = output_path + os.path.sep + animal + '_' + TSer + '_' + timestamp + '_mc.pkl'
    with open(savefile, 'wb') as file:
        pickle.dump(mc, file, protocol=pickle.HIGHEST_PROTOCOL)

#if play movie 
    if display_movie:
        m_orig = cm.load_movie_chain(fnames)
        mc_movies = mc.fname_tot_els if pw_rigid else mc.fname_tot_rig
        m_els = cm.load(mc.fname_tot_els)
        cm.concatenate([m_orig.resize(1, 1, ds_ratio) - mc.min_mov * mc.nonneg_movie,
                        m_els.resize(1, 1, ds_ratio)], axis=2).play(fr=60, gain=1, magnification=2,
                                                                        offset=0)  # press q to exit
# Evaluate motion correction
    if compute_metrics:
        print('Computing performance metrics')
        m_orig = cm.load_movie_chain(fnames)
        m_els = cm.load(mc.fname_tot_els) if pw_rigid else cm.load(mc.fname_tot_rig)

        plt.figure(figsize = (20,10))
        plt.subplot(1,2,1); plt.imshow(m_orig.local_correlations(eight_neighbours=True, swap_dim=False))
        plt.subplot(1,2,2); plt.imshow(m_els.local_correlations(eight_neighbours=True, swap_dim=False))

        #visualize elastic shifts

        plt.figure(figsize = (20,10))
        plt.subplot(2, 1, 1)
        plt.plot(mc.x_shifts_els)
        plt.ylabel('x shifts (pixels)')
        plt.subplot(2, 1, 2)
        plt.plot(mc.y_shifts_els)
        plt.ylabel('y_shifts (pixels)')
        plt.xlabel('frames')
        #compute borders to exclude
        bord_px_els = np.ceil(np.maximum(np.max(np.abs(mc.x_shifts_els)),
                                         np.max(np.abs(mc.y_shifts_els)))).astype(int)

    # compute metrics for the results (TAKES TIME!!)
        final_size = np.subtract(mc.total_template_els.shape, 2 * bord_px_els) # remove pixels in the boundaries
        winsize = 100
        swap_dim = False
        resize_fact_flow = .2    # downsample for computing ROF

        tmpl_orig, correlations_orig, flows_orig, norms_orig, crispness_orig = cm.motion_correction.compute_metrics_motion_correction(
            fnames[0], final_size[0], final_size[1], swap_dim, winsize=winsize, play_flow=False, resize_fact_flow=resize_fact_flow)

        tmpl_els, correlations_els, flows_els, norms_els, crispness_els = cm.motion_correction.compute_metrics_motion_correction(
            mc.fname_tot_els[0], final_size[0], final_size[1],
            swap_dim, winsize=winsize, play_flow=False, resize_fact_flow=resize_fact_flow)

        #Plot correlations
        plt.figure(figsize = (20,10))
        plt.subplot(211); plt.plot(correlations_orig); plt.plot(correlations_els)
        plt.legend(['Original','PW-Rigid'])
# Stop server
    cm.stop_server(dview=dview)
  
    #done!
    print('Done!')
    return mc,opts
##################################
def motion_correct_2color_single(TSeries,filetype='tif',template_channel=2,load_mc_obj=False,template=None,
                                 play_movie=False,plot_reg=True,save_reg=True,ds_ratio=None,
                                 max_shifts=(30, 30), pw_rigid=True,mc_opts:dict=None):
    """
    Motion-correct a two-color (dual-channel) TSeries by registering one channel and applying the same shifts to the other.

    Loads the two channel files from `TSeries` (assumed alphabetically
    ordered so that channel 1 comes before channel 2), runs CaImAn motion
    correction on the designated template channel, applies the resulting
    shifts to the other channel, and optionally plots and saves the
    registered mean-projection images and movies.

    Parameters
    ----------
    TSeries : str
        Path to the two-color TSeries folder; should contain exactly two
        image files (tif or mmap) that sort alphabetically as channel 1,
        then channel 2.
    filetype : str, optional
        File extension/substring used to find the channel files (default
        'tif').
    template_channel : {1, 2}, optional
        Which channel to align to the template: 1 to align channel 1, 2 to
        align channel 2 (default 2).
    load_mc_obj : bool, optional
        If True, prompt the user to select a previously saved `mc` object
        (`.pkl`) to reuse its template and parameters (default False).
    template : ndarray, optional
        Template image to register to; if None and `load_mc_obj` is True,
        the template is taken from the loaded `mc` object (default None).
    play_movie : bool, optional
        If True, play the registered green/red movies concatenated side by
        side (default False).
    plot_reg : bool, optional
        If True, plot the mean projection images (and template, if given)
        after registration (default True).
    save_reg : bool, optional
        If True, save the registered movies, mean-projection tifs, `mc`
        object, and registration figure to `TSeries` (default True).
    ds_ratio : float, optional
        Downsampling ratio used when playing movies; if None, defaults to
        0.5 (default None).
    max_shifts : tuple of int, optional
        Maximum allowed rigid shifts, in pixels, used when `mc_opts` is not
        given via `load_mc_obj` (default (30, 30)).
    pw_rigid : bool, optional
        If True, use piecewise-rigid motion correction (default True).
    mc_opts : dict, optional
        Dictionary with keys 'strides', 'overlaps', 'max_shifts', and
        'max_deviation_rigid', used when `load_mc_obj` is False (default
        None).

    Returns
    -------
    m_red : caiman.movie
        Registered red-channel movie.
    m_green : caiman.movie
        Registered green-channel movie.
    mc_mcherry : caiman.motion_correction.MotionCorrect
        The motion correction object used to register the template channel.
    """
    fnames= [os.path.join(TSeries,f) for f in get_filelist(TSeries,filetype)]
    fnames.sort()
    ch1_fname = [fnames[0]]
    ch2_fname = [fnames[1]]
    print('Ch1 file:'+ch1_fname[0])
    print('Ch2 file:'+ch2_fname[0])


    if load_mc_obj:
        temp_file = filedialog.askopenfilename(title='Choose motion correction pkl file')
        with open(temp_file, 'rb') as file:
            mc = pickle.load(file)
        if template is None:
            print('using mc object tempate from loaded baseline file:')
            print(temp_file)
            template = mc.total_template_els if mc.pw_rigid else mc.total_template_rig

    else:
        if template is not None:
            print("using user defined template")

    if template_channel == 2:
        print('template channel is ch2')
        mc_files = ch2_fname
        shift_files = ch1_fname
    elif template_channel == 1:
        print('template channel is ch1')
        mc_files = ch1_fname
        shift_files = ch2_fname

    if 'dview' in locals():
        cm.stop_server(dview=dview)
    c, dview, n_processes = cm.cluster.setup_cluster(backend='local', n_processes=None, single_thread=False)

    if load_mc_obj:  # if loaded a previous mc object saved
        mc_mcherry = MotionCorrect(mc_files, dview=dview, max_shifts=mc.max_shifts, pw_rigid=mc.pw_rigid,
                                   strides=mc.strides, overlaps=mc.overlaps,
                                   max_deviation_rigid=mc.max_deviation_rigid,
                                   shifts_opencv=mc.shifts_opencv, nonneg_movie=True,
                                   border_nan=mc.border_nan)
    else:  # or create new mc object and define params
        print('using user defined mc params')
        strides = mc_opts['strides']
        overlaps=mc_opts['overlaps']
        max_shifts = mc_opts['max_shifts']
        max_deviation_rigid=mc_opts['max_deviation_rigid']

        mc_mcherry = MotionCorrect(mc_files, dview=dview, max_shifts=max_shifts, pw_rigid=pw_rigid,
                                   strides=strides, overlaps=overlaps,
                                   max_deviation_rigid=max_deviation_rigid, nonneg_movie=True,
                                   border_nan='copy')

    mc_mcherry.motion_correct(template=template, save_movie=True)

    if mc_mcherry.pw_rigid:
        append = "_els_"
    else:
        append = "_rig_"

    base_name = shift_files[0].split(os.path.sep)[-1].split('.')[-2] + append

    mmap_shifted = mc_mcherry.apply_shifts_movie(shift_files, save_memmap=True, order='F',
                                                 save_base_name=os.path.join(os.path.split(shift_files[0])[0],
                                                                             base_name))
    #
    if (mc_mcherry.pw_rigid == True) & (template_channel == 2):
        m_green = cm.load(mc_mcherry.fname_tot_els)
        m_red = cm.load(mmap_shifted)
    elif (mc_mcherry.pw_rigid == False) & (template_channel == 2):
        m_green = cm.load(mc_mcherry.fname_tot_rig)
        m_red = cm.load(mmap_shifted)

    elif (mc_mcherry.pw_rigid == True) & (template_channel == 1):
        m_red = cm.load(mc_mcherry.fname_tot_els)
        m_green = cm.load(mmap_shifted)
    elif (mc_mcherry.pw_rigid == False) & (template_channel == 1):
        m_red = cm.load(mc_mcherry.fname_tot_rig)
        m_green = cm.load(mmap_shifted)

    temp_green = m_green.mean(0)
    temp_red = m_red.mean(0)

    if play_movie:
        if ds_ratio==None:
            ds_ratio = 0.5
        cm.concatenate([m_green.resize(1, 1, ds_ratio),
                        m_red.resize(1, 1, ds_ratio)], axis=2).play(fr=60, gain=1, magnification=2,
                                                                    offset=0)  # press q to exit
    if plot_reg:
        if template is not None:
            fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))
            ax[0].imshow(template, cmap='gist_gray')
            ax[0].axis('off')
            ax[0].set_title('920 nm GCaMP Template')
            ax[1].imshow(temp_green, cmap='gist_gray')
            ax[1].set_title('1040 nm GCaMP')
            ax[1].axis('off')
            ax[2].imshow(temp_red, cmap='gist_gray')
            ax[2].set_title('1040 nm mCherry')
            ax[2].axis('off')
            fig.tight_layout()

            if save_reg:
                fig.savefig(os.path.join(TSeries, TSeries.split(os.path.sep)[-1] + '_2colorReg.png'))
        else:
            fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))
            ax[0].imshow(temp_green, cmap='gist_gray')
            ax[0].axis('off')
            ax[0].set_title('Green Channel')
            ax[1].imshow(temp_red, cmap='gist_gray')
            ax[1].set_title('Red Channel')
            ax[1].axis('off')
            fig.tight_layout()

            if save_reg:
                fig.savefig(os.path.join(TSeries, TSeries.split(os.path.sep)[-1] + '_2colorReg.png'))

    cm.stop_server(dview=dview)  # stop  server
    if save_reg:
        mchreg_file = os.path.join(TSeries, base_name)

        if hasattr(mc_mcherry, 'dview'):
            del (mc_mcherry.dview)
        with open(mchreg_file + 'mc.pkl', 'wb') as file:
            pickle.dump(mc_mcherry, file, protocol=pickle.HIGHEST_PROTOCOL)

        if template is not None:  # save original template if exists
            try:
                tifffile.imsave(os.path.join(TSeries, os.path.split(temp_file)[-1].split('_')[1]) + '_template.tif',
                            template)
            except UnboundLocalError:
                pass

        # save mean projection tifs
        tifffile.imwrite(os.path.join(TSeries, ch1_fname[0].split('.')[0].split(os.path.sep)[-1] + append + '.tif'),
                        temp_red)
        tifffile.imwrite(os.path.join(TSeries, ch2_fname[0].split('.')[0].split(os.path.sep)[-1] + append + '.tif'),
                        temp_green)
        # save movies
        tifffile.imwrite(os.path.join(TSeries, ch1_fname[0].split('.')[0].split(os.path.sep)[-1] + append + 'mc_movie.tif'),
                        m_red)
        tifffile.imwrite(os.path.join(TSeries, ch2_fname[0].split('.')[0].split(os.path.sep)[-1] + append + 'mc_movie.tif'),
                        m_green)
        np.savez(mchreg_file + 'movies.npz', m_red=m_red, temp_red=temp_red, m_green=m_green, temp_green=temp_green)
    return m_red,m_green,mc_mcherry

