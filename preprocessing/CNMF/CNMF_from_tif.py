#%%
import bokeh.plotting as bpl
import cv2
import glob
import logging
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import pandas as pd
import re
from datetime import datetime
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
from caiman.source_extraction.cnmf import cnmf as cnmf
from caiman.source_extraction.cnmf import params as params
from caiman.utils.utils import download_demo
from caiman.utils.visualization import plot_contours, view_patches
import sys
from scipy.sparse import csc_matrix
os.environ['CAIMAN_TEMP'] = '.'

__all__=["CNMF","CNMF_batch","CNMF_animal"]
#%%
def main():
    """
    Run CNMF source extraction (from mmap/tif files) for a single animal/FOV/session via CLI arguments.

    Parameters
    ----------
    Reads `ani` from `sys.argv[1]`, `fov` from `sys.argv[2]`, and
        `session` from `sys.argv[3]`.
    Notes
    -----
    `base_dir` and `file_key` are hardcoded within the function body. Calls
    `CNMF` with `plotting=False` and `seedCNMF=False`.
    """
    base_dir = '/projectnb/sramirezlab/amonast/Tone2P'
    file_key='/projectnb/sramirezlab/amonast/Tone2P/Data_info_TFC.csv'
    df=pd.read_csv(file_key)
    ani = sys.argv[1]#'160R'
    fov=sys.argv[2]#s#'FOV1'
    session =sys.argv[3]#'Baseline'#int(sys.argv[2])

    animal_path = os.path.join(base_dir,'GCaMP',ani)
    TSer = df['TSeries_g'].loc[(df.Animal==ani)&(df.Session==session)&(df.FOV==fov)].values[0]
    TSeries = os.path.join(animal_path,TSer)
    CNMF(TSeries,plotting=False,new_opts={'seedCNMF':False})
#%%
def CNMF_animal(ani,fov,base_dir,file_key,new_opts=None):
    """
    Run CNMF source extraction for every session of a given animal/FOV.

    Parameters
    ----------
    ani : str
        Animal identifier.
    fov : str
        FOV identifier.
    base_dir : str
        Base directory for the experiment, containing 'GCaMP/{ani}' and its
        motion-correction outputs.
    file_key : str
        Path to the metadata CSV for the experiment.
    new_opts : dict, optional
        CNMF parameter overrides passed through to `CNMF` (default None).
    """
    df = pd.read_csv(file_key)
    mc_path =os.path.join(base_dir,'GCaMP',ani,'caiman_output','motion_correct')
    series_all = df['TSeries_g'].loc[(df['Animal']==ani)&(df['FOV']==fov)].values
    TSeries_all = [os.path.join(base_dir,'GCaMP',series) for series in series_all]
    TSeries_all.sort()
    for TSeries in TSeries_all:
        print(TSeries)
        CNMF(mc_path,TSeries,plotting=False,save_results=True,new_opts=new_opts)

def CNMF_batch(base_directory,animals=None,session_idx=None,params_list=None):
    """
    Run CNMF source extraction across multiple animals/sessions, optionally sweeping parameter sets.

    Parameters
    ----------
    base_directory : str
        Base directory containing 'GCaMP/{animal}' subfolders.
    animals : list of str, optional
        Animal subfolder names to process; if None, uses every subfolder
        found under `base_directory` (default None).
    session_idx : int, optional
        Index into each animal's sorted list of TSeries subfolders,
        specifying a single session to process; if None, processes every
        session (default None).
    params_list : list of dict, optional
        List of CNMF parameter-override dictionaries to try in sequence for
        each session; if None, uses default CNMF parameters (default None).

    Returns
    -------
    None

    Examples
    --------
    >>> session_idx = 0
    >>> params_list = [{'K': 7, 'gSig': [4, 4]}, {'K': 5, 'gSig': [4, 4]},
    ...                {'K': 10, 'gSig': [4, 4]}]
    >>> CNMF_batch(base_dir, animals, session_idx=session_idx, params_list=params_list)
    """
    if animals is None:
        animals = [os.path.join(base_directory, 'GCaMP',ani) for ani in os.listdir(base_directory)]

    for a in animals:
        ani = os.path.join(base_directory,'GCaMP',a)
        mc_path =os.path.join(ani,'caiman_output','motion_correct')
        TSeries_all = [os.path.join(ani,T) for T in os.listdir(ani) if 'caiman_output' not in T]
        TSeries_all.sort()
        if session_idx is None: # if not specifying one session, will do all sessions in the folder
            for T in TSeries_all:
                print("Processing: "+ T)
                if params_list is not None:
                    for params_dict in params_list:
                        print(params_dict)
                        CNMF(mc_path,T,plotting=True,save_results=True,new_opts=params_dict)
                elif params_list is None:
                    params_dict=None
                    print(T)
                    CNMF(mc_path,T,plotting=True,save_results=True,new_opts=params_dict)       
        else: #else it will specify a particular sesssion, indexed from the sorted list of subfolders
            T = TSeries_all[session_idx]
            if params_list is not None:
                for i,params_dict in enumerate(params_list):
                        print(f"parameter set +{i})+ of +{len(params_list)}")
                        print(T)
                        CNMF(mc_path,T,plotting=True,save_results=True,new_opts=params_dict)
            elif params_list is None:
                params_dict=None
                print(T)
                CNMF(mc_path,T,plotting=True,save_results=True,new_opts=params_dict)

#%%
def CNMF(TSeries,plotting=False,save_results=True,new_opts=None):
    """
    Run CaImAn CNMF source extraction directly from mmap or TIFF files (no motion-correction pickle required).

    Sets up default CNMF parameters (optionally overridden by `new_opts`),
    locates existing `.mmap` files (or raw registered TIFFs as a fallback)
    for `TSeries`, memory-maps the movie, runs patch-wise CNMF (optionally
    seeded from Cellpose segmentation masks), refits and evaluates the
    resulting components, computes dF/F, and optionally plots and saves the
    results.

    Parameters
    ----------
    TSeries : str
        Path to the TSeries folder to process.
    plotting : bool, optional
        If True, plot detected component contours and view final traces
        (default False).
    save_results : bool, optional
        If True, save contour plots and the final CNMF results `.hdf5` file
        (default True).
    new_opts : dict, optional
        CNMF parameter overrides. The special key 'seedCNMF' (bool)
        controls whether CNMF is seeded with Cellpose segmentation masks
        instead of running patch-wise initialization; remaining keys are
        passed to `opts.change_params` (default None).

    Returns
    -------
    cnm2 : caiman.source_extraction.cnmf.cnmf.CNMF
        The fitted and evaluated CNMF object, with dF/F computed and only
        accepted components retained.

    Notes
    -----
    Saves outputs (contour plots, results `.hdf5`) under
    `{parent_of(TSeries)}/caiman_output/cnmf`. Unlike `CNMF.py`'s `CNMF`
    function, this version never loads a motion-correction `mc` object, so
    `plotting=True` will raise a `NameError` at the point it references
    `mc.total_template_els`/`mc.pw_rigid`.
    """
    animal_path = os.path.split(TSeries)[0]
    animal = os.path.split(animal_path)[-1]
    TSer = 'TSeries'+os.path.split(TSeries)[-1].split('TSeries', 1)[1].strip()
    #Set up parameters 
        # parameters for source extraction and deconvolution
    p = 1                       # order of the autoregressive system
    gnb = 2                     # number of global background components
    merge_thr = 0.85            # merging threshold, max correlation allowed
    rf = 25                     # half-size of the patches in pixels. e.g., if rf=25, patches are 50x50
    stride_cnmf = 6             # amount of overlap between the patches in pixels
    K = 7                       # number of components per patch
    gSig = [5,5]               # expected half size of neurons in pixels
    method_init = 'greedy_roi'  # initialization method (if analyzing dendritic data using 'sparse_nmf')
    ssub = 1                    # spatial subsampling during initialization
    tsub = 2                    # temporal subsampling during intialization
    
    # parameters for component evaluation
    min_SNR = 2.0               # signal to noise ratio for accepting a component
    rval_thr = 0.85             # space correlation threshold for accepting a component
    cnn_thr = 0.99              # threshold for CNN based classifier
    cnn_lowest = 0.1            # neurons with cnn probability lower than this value are rejected
    
    #seed cnmf parameters
    seedCNMF=False #false by default
    only_init=False if seedCNMF else True
    rf=None if seedCNMF else rf

    #set cnmf default parameters 
    cnmf_opts_dict={'nb': gnb,
                'rf': rf,
                'gSig':gSig,
                'K': K,
                'stride': stride_cnmf,
                'method_init': method_init,
                'rolling_sum': True,
                'only_init': only_init,
                'ssub': ssub,
                'tsub': tsub,
                'merge_thr': merge_thr,
                'min_SNR': min_SNR,
                'rval_thr': rval_thr,
                'use_cnn': True,
                'min_cnn_thr': cnn_thr,
                'cnn_lowest': cnn_lowest}
    
    opts = params.CNMFParams(params_dict=cnmf_opts_dict)
    opts.data['path']=TSeries
    opts.data['animal']=animal
    opts.data['TSeries']=TSer
    opts.data['seedCNMF']=seedCNMF
#%%update cnmf params object if passed new opts 
    if new_opts is not None: 
        # seed cnmf parameters
        value = new_opts.get('seedCNMF')
        if value is not None:
            print("seedCNMF is ", value)
            seedCNMF=value
            opts.data['seedCNMF']=seedCNMF
        else:
            print("seedCNMF not present, using default " + str(seedCNMF))
            opts.data['seedCNMF']=seedCNMF 
        
        if seedCNMF:
            opts.patch['rf']=None
            opts.patch['only_init']=False
        
        #change params for any new ones passed in input arguments
        opts.change_params(new_opts,verbose=True)

#%% load in mc object if present; if it is it will get the filenames for CNMF from this mc pkl file
    border_to_0 = 0 
    fnames = glob.glob(f"{TSeries}/*.mmap")
    
    if len(fnames)==0:
        print('No mmaps file found. using list of tif files present:')
        fnames = glob.glob(f"{TSeries}/suite2p/plane0/reg_tif/*.tif")   
    fnames = sorted(fnames, key=lambda p: int(re.search(r'file(\d+)_chan0', p).group(1)))
    print('File list...')
    for i in fnames:
        print(i)

#%%create cnmf output path
    output_path = os.path.join(os.path.split(opts.data['path'])[0], 'caiman_output', 'cnmf')
    
    # make output path to save results
    os.makedirs(output_path,exist_ok=True)
 
    print('Starting cluster')
    #%% Setup Cluster
    if 'dview' in locals():
        cm.stop_server(dview=dview)
    c, dview, n_processes = cm.cluster.setup_cluster(
        backend='local', n_processes=None, single_thread=False)

    #%%Memory mapping
    # The cell below  memory maps the file in order `'C'` and then loads the new memory mapped file.The saved files  from
    # motion correction are memory mapped files stored in 'F' order their paths are stored ibn mc.mmap_file
    # if C order memmap file exists already use it 
    try:
        fname_new = [os.path.join(TSeries,f) for f in os.listdir(TSeries) if 'memmap__' in f][0]
        print('Loading memmap C file')
        print(fname_new)
    
    #memory map the file in order 'C'
    except:
        print('Writing mmap files from motion corrected files: '+ fnames[0])
        fname_new = cm.save_memmap(fnames, base_name='memmap_', order='C',
                                border_to_0=border_to_0, dview=dview)  # exclude borders
    
    #reshape movie data 
    Yr, dims, T = cm.load_memmap(fname_new)
    images = np.reshape(Yr.T, [T] + list(dims), order='F')  # load frames in python format (T x X x Y)
    print(fname_new) # load frames in python format (T x X x Y)
    
    #%% if seeding CNMF with spatial masks
    
    if seedCNMF:
        seg_file = glob.glob(f"{TSeries}/*{TSer}*seg.npy")
        if len(seg_file)==0:
            seg_file=glob.glob(f"{TSeries}/{TSer}*seg.npy")
        print('masks for seeded cnmf from cellpose output: '+seg_file[0])
        seg=np.load(seg_file[0],allow_pickle=True).item()
        masks_2d=seg['masks']

        def masks_to_sparse(masks_2d):
            """
            Convert a labeled 2D segmentation mask into a dense boolean spatial component matrix.

            Parameters
            ----------
            masks_2d : ndarray
                2D array where each unique positive integer value represents one
                ROI's pixels (0 is background).

            Returns
            -------
            A : ndarray of bool, shape (masks_2d.size, num_components)
                Spatial component matrix in Fortran (column-major) flattened pixel
                order, one column per ROI (excluding background).
            """
            # Get the number of unique ROIs (excluding 0 if it's the background)
            unique_labels = np.unique(masks_2d)
            unique_labels = unique_labels[unique_labels > 0]  # Exclude background (label 0)
            num_components = len(unique_labels)

            # Initialize an empty dense matrix to store pixel-to-ROI mapping
            A = np.zeros((masks_2d.size, num_components), dtype=bool)

            # Loop over each unique ROI and create binary masks
            for i, label in enumerate(unique_labels):
                temp = (masks_2d == label)  # Binary mask for the current ROI
                A[:, i] = temp.flatten('F')  # Flatten in column-major order (Fortran-style)

            # Convert A to a sparse matrix for memory efficiency
            #A_sparse = csc_matrix(A)

            return A

        Aseed = masks_to_sparse(masks_2d)
    
    #plt.figure(); plt.imshow(np.reshape(A_csc.toarray().max(axis=1), dims, order='F'),origin='upper')
    
    print('Running CNMF')
    # %% RUN CNMF ON PATCHES
    # First extract spatial and temporal components on patches and combine them
    # for this step deconvolution is turned off (p=0). If you want to have deconvolution within each patch change
    # params.patch['p_patch'] to a nonzero value
    Ain=Aseed if seedCNMF else None
    cnm = cnmf.CNMF(n_processes, params=opts, dview=dview,Ain=Ain)
    cnm = cnm.fit(images)

    # %% RE-RUN seeded CNMF on accepted patches to refine and perform deconvolution
    cnm2 = cnm.refit(images, dview=dview)
    
    #%% calculate correlation image
    if (dims != (256,256)) or (T>100000): #if big data [not 256x256 or greater than 100kframes] subsample to make Correlation image
        rs_images = images.transpose(1, 2, 0)
        ds_images = rs_images[:,:,::2]
        Cn = cm.local_correlations(ds_images)
    else:
        Cn = cm.local_correlations(images.transpose(1, 2, 0))
    Cn[np.isnan(Cn)] = 0
    
    cnm2.estimates.Cn = Cn # Update cnm2 object with Cn image

#%% COMPONENT EVALUATION
    # the components are evaluated in three ways:
    #   a) the shape of each component must be correlated with the data
    #   b) a minimum peak SNR is required over the length of a transient
    #   c) each shape passes a CNN based classifier

    cnm2.estimates.evaluate_components(images, cnm2.params, dview=dview)

#%% PLOT Evaluated Spatial COMPONENTS
    now = datetime.now()  # generate a time stamp
    timestamp = str(now.month) + str(now.day) + str(now.year) + '_' + str(now.hour) + str(now.minute) + str(now.second)

    if plotting:
        templ=mc.total_template_els if mc.pw_rigid else mc.total_template_rig
        cnm2.estimates.plot_contours(img=Cn, idx=cnm2.estimates.idx_components,cmap='gray')
        if save_results:
            plt.savefig(output_path+os.path.sep+f"{opts.data['animal']}_{opts.data['path'].split(os.path.sep)[-1]}_cnmf_ROIS_Cn_{timestamp}.png")
        cnm2.estimates.plot_contours(img=templ, idx=cnm2.estimates.idx_components,cmap='gray')
        if save_results:
            plt.savefig(output_path+os.path.sep+f"{opts.data['animal']}_{opts.data['path'].split(os.path.sep)[-1]}_cnmf_ROIS_tpl_{timestamp}.png")

#%% View Components with Spatial + Temporal traces (not dff)
    # # accepted components
    # cnm2.estimates.view_components(img=Cn, idx=cnm2.estimates.idx_components)
    #
    # # rejected components
    # if len(cnm2.estimates.idx_components_bad) > 0:
    #     cnm2.estimates.view_components(img=Cn, idx=cnm2.estimates.idx_components_bad)
    # else:
    #     print("No components were rejected.")

#%% Select only good components
    cnm2.estimates.select_components(use_object=True)

#%% Extract DF/F values
    print('Calculating DF/F...')
    cnm2.estimates.detrend_df_f(quantileMin=8, frames_window=500)

    # %% Show final traces
    if plotting:
        cnm2.estimates.view_components(img=Cn)

    # %% Stop server Save CNMF results
    cm.stop_server(dview=dview)
    if save_results:
        print('Saving results')
        cnm2.save(output_path+os.path.sep+f"{opts.data['animal']}_{opts.data['path'].split(os.path.sep)[-1]}_cnmfresults_{timestamp}.hdf5")

    return cnm2

#%%
if __name__=='__main__':
    main()