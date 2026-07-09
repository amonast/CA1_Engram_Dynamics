
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

__all__=["load_mc","get_filelist","get_TSeries","get_template_baseline","motion_correct","motion_correct_2color_single"]
##############
def main():
    """
    Motion-correct GCaMP TSeries for a hardcoded animal/FOV via an interactive folder picker.

    Prompts the user to select a TSeries folder, loads the Baseline motion
    correction template for that animal/FOV, and runs rigid motion correction
    on all TIFFs in the folder (saving a mean projection but not an AVI or
    metrics). An alternate, currently-disabled code path (`choose=False`)
    motion-corrects a fixed list of sessions (D0-D4) in sequence, aligning
    each to the prior session's template.

    Parameters
    ----------
    None

    Returns
    -------
    None

    Notes
    -----
    `animal`, `fov`, `base_dir`, `file_key`, and `choose` are hardcoded
    within the function body.
    """
    animal = 'M5L'#sys.argv[1]
    fov = 'FOV1'#sys.argv[2]
    base_dir ="/projectnb/sramirezlab/amonast/Tone2P/"
    file_key = "/projectnb/sramirezlab/amonast/Tone2P/Data_info_TFC.csv"
    imagespath = os.path.join(base_dir,'GCaMP',animal)
    df = pd.read_csv(file_key)
    choose=True

    if choose==True:
        path =filedialog.askdirectory(initialdir='/projectnb/sramirezlab/amonast/Tone2P/GCaMP')
        os.chdir(path)
        template = get_template_baseline(path,'Baseline',file_key)[0]
        files = get_filelist(path, 'tif')
        fnames = [os.path.join(path, f) for f in files if 'els' not in f]
        fnames.sort()
        motion_correct(fnames,template=None,pw_rigid=False,display_movie=False,
                       save_avi=False,compute_metrics=False,save_template=False,save_mean_proj=True)
        # mmaps = [os.path.join(path,m) for m in os.listdir(path) if '.mmap' in m]
        # mmaps.sort()
        # motion_correct(mmaps,template=template,pw_rigid=False,display_movie=False,save_avi=False,compute_metrics=False)
        #mmaps2=  [os.path.oin(path,m) for m in os.listdir(path) if m.endswith('rig__d1_512_d2_512_d3_1_order_F_frames_2000.mmap')]
        #motion_correct(mmaps2,template=template,pw_rigid=True,display_movie=False,save_avi=True,compute_metrics=False)

        print('All done!')
    elif choose==False:
        #ignore for now until ready to do multiple @ once
        TSeries = []
        for session in ['D0','D1','D2','D3','D4']:
            series = df['TSeries_g'].loc[(df['Animal']==animal)&(df['FOV']==fov)&(df['Session']==session)].values[0]
            TSeries.append(series)
        print(TSeries)
        subfolders = [os.path.join(imagespath,T) for T in TSeries]
        tifs = get_filelist(subfolders[0],'tif')
        fnames = [os.path.join(subfolders[0],f) for f in tifs]
        fnames.sort()
        print('motion correcting: '+ fnames[0])
        print(fnames)
        motion_correct(fnames,template=None,pw_rigid=True,display_movie=False,save_avi=True,compute_metrics=False)
        print('motion correcting to baseline template: '+ subfolders[1])
        tifs2 = get_filelist(subfolders[1], 'tif')
        fnames2 = [os.path.join(subfolders[1], f) for f in tifs2 if 'movie' in f]
        fnames2.sort()
        print(fnames2)
        template = get_template_baseline(subfolders[1],'Baseline',file_key)[0]
        motion_correct(fnames2,template=None,pw_rigid=True,display_movie=False,save_avi=True,compute_metrics=False)
        tifffile.imwrite(os.path.join(imagespath,TSeries[0])+'_template.tif',template)
        print('All done!')

if __name__=='__main__':
    main()