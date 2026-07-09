
import bokeh.plotting as bpl
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
from tkinter import filedialog
from datetime import datetime
import pickle
import tifffile
from motion_correction_functions import *
os.environ['CAIMAN_TEMP'] = '.'

#from metadata import parseExperimentXML
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
import sys
import glob

__all__=["load_mc","get_filelist","get_TSeries","get_template_baseline","motion_correct","motion_correct_2color_single"]
##############
def main():
    """
    Motion-correct a single animal/session/FOV's TSeries, driven by CLI arguments.

    Reads animal, session, and FOV from `sys.argv`, locates the
    corresponding TIFF files, optionally loads a template from a reference
    session (`baseline_session_name`), and runs one or two rounds of motion
    correction (`N_MC`) via `motion_correct`, either with a single default
    parameter set or a batch sweep of parameter sets (`mc_params_batch`).

    Parameters
    ----------
    None
        Reads `animal` from `sys.argv[1]`, `session` from `sys.argv[2]`, and
        `fov` from `sys.argv[3]`.

    Returns
    -------
    None

    Notes
    -----
    `baseline_session_name`, `use_template`, `base_dir`, `file_key`,
    `batch`, `pw_rigid`, `N_MC`, and the motion correction parameter
    dictionaries are hardcoded within the function body.
    """
    #define animal and experiment variables
    animal = sys.argv[1]
    baseline_session_name='D3S2' #name of the reference session for template registration
    session = sys.argv[2]
    fov = sys.argv[3]
    use_template=False
    print(animal+' '+ session+ ' '+fov)
    base_dir ="/projectnb/sramirezlab/amonast/PVE"
    file_key = "/projectnb/sramirezlab/amonast/PVE/data_PVE.csv"
    info=pd.read_csv(file_key)

    ### switch to path with image files
    gcamp_path = os.path.join(base_dir,'GCaMP',animal)
    TSeries = info['TSeries_g'].loc[(info.Animal==animal)&(info.Session==session)&(info.FOV==fov)].values[0]
    path = glob.glob(f"{gcamp_path}/{TSeries}")
    os.chdir(path[0])

    ## get template if not the first session
    if use_template:
        if session !=baseline_session_name:
            template = get_template_baseline(path[0],baseline_session_name,file_key)[0]
            #path2 = '/projectnb/sramirezlab/amonast/PVE/GCaMP/F6R/TSeries-11192024-092/MC_TSeries-11192024-092_2172025_173731/MC_MC_TSeries-11192024-092_2172025_173731_2172025_195353'
            #template = tifffile.imread(glob.glob(f"{path2}/*mc_mean_proj.tif"))
        else:
            template=None
    else:
        template=None
    #get the files and store them in a list
    files = get_filelist(path[0], 'tif')
    fnames = [os.path.join(path[0], f) for f in files]
    fnames.sort()

    #set motion correction settings
    batch=False #if you want to run multiple batches
    pw_rigid=True #for non-rigid or rigid motion correction
    N_MC=1 # number of rounds of motion correction

    if batch:
        print('batch')
        mc_params_batch=[
                   {'strides':(64,64),
                'overlaps': (18,18),
                'max_shifts': (30,30),
                'max_deviation_rigid': 3,
                'pw_rigid': pw_rigid},
                  {'strides': (96,96),
                'overlaps': (18,18),
                'max_shifts': (30,30),
                'max_deviation_rigid': 3,
                'pw_rigid': pw_rigid},
                {'strides': (84,84),
                'overlaps': (20,20),
                'max_shifts': (30,30),
                'max_deviation_rigid': 3,
                'pw_rigid': pw_rigid}]

        for params in mc_params_batch:
            #fnames=fnames[5:]+fnames[0:5]
            print(fnames[0])
            motion_correct(fnames,template=template,pw_rigid=pw_rigid,
                    display_movie=False,save_avi=False,compute_metrics=False,
                    mc_opts = params,move_mmaps=True)
    else: # no batch, default settings 
        params = {'strides': (84,84),
                'overlaps': (18,18),
                'max_shifts': (10,10),
                'max_deviation_rigid': 2,
                'pw_rigid': pw_rigid}
            
        mc,opts=motion_correct(fnames,template=template,pw_rigid=pw_rigid,
                    display_movie=False,save_avi=False,compute_metrics=False,
                    mc_opts=params,move_mmaps=True)
    if N_MC==2:
        params = {'strides': (96,96),
                    'overlaps': (24,24),
                    'max_shifts': (30,30),
                    'max_deviation_rigid': 3,
                    'pw_rigid': pw_rigid}
        fnames2 = get_filelist(opts.data['path'], 'mmap')
        motion_correct(fnames2,template=template,pw_rigid=pw_rigid,
                        display_movie=False,save_avi=False,compute_metrics=False,
                        mc_opts=params,move_mmaps=True)
    print('done!')

if __name__=='__main__':
    main()
