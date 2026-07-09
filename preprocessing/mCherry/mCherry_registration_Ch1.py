#%%
import os
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
from motion_correct import motion_correct_2color_single, get_filelist
#%%
__all__ = ["get_gcamp_temp","reg_mcherry","reg_mcherry_batch"]
#################################
def main():
    """
    Motion-correct a hardcoded two-color mCherry TSeries and save the registered movies.

    Loads a fixed GCaMP template TIFF, runs
    `motion_correct_2color_single` on a hardcoded mCherry TSeries, and then
    saves the registered channel movies via `save_npz_movies`.

    Parameters
    ----------
    None

    Returns
    -------
    None

    Notes
    -----
    `file_key`, `base_dir`, `TSeries_1040`, and `temp_file` are hardcoded
    within the function body.
    """
    file_key='/projectnb/sramirezlab/akemiito/TD/data_TD.csv'
    base_dir = '/projectnb/sramriezlab/akemiito/TD'
   
    TSeries_1040 = '/projectnb/sramirezlab/akemiito/TD/mCherry/TD6/062024-TD6-D0-MCreal1'
    filedialog.askdirectory(initialdir=os.path.join(base_dir,'mCherry'))
    temp_file='/Volumes/AM_SSD1/Caiman/mCherry/217N/TSeries-07162023-1040-093/TSeries-07162023-1040-093_Ch1_els__0.tif'
    template = tifffile.imread(temp_file)
    print(type(template))
    #motion_correct_2color_single(TSeries_1040,filetype='tif',template_channel=1,load_mc_obj=False,template=template,
    motion_correct_2color_single(TSeries_1040,filetype='tif',load_mc_obj=False,
                                 play_movie=True,plot_reg=False,save_reg=True,ds_ratio=.5,
                                 max_shifts=(50, 50), pw_rigid=True,strides=(200, 200), overlaps=(18, 18),
                                 max_deviation_rigid=40)

    save_npz_movies(TSeries_1040)
    print('done!')
#############################
def save_npz_movies(TSeries_1040):
    """
    Load registered two-channel `.mmap` movies and save them (with mean projections) to an `.npz` file.

    Parameters
    ----------
    TSeries_1040 : str
        Path to the TSeries folder containing the two registered `.mmap`
        files (channel 1 and channel 2, in sorted order).

    Returns
    -------
    None

    Notes
    -----
    Writes an `.npz` file (named `{TSeries basename}_Ch1_{els|rig}_movies.npz`
    in the current working directory) containing `m_red`, `temp_red`,
    `m_green`, and `temp_green`.
    """
    mmaps = [os.path.join(TSeries_1040,f) for f in os.listdir(TSeries_1040) if f.endswith('mmap')]
    mmaps.sort()
    Ch1_file = mmaps[0]
    Ch2_file = mmaps[1]

    m_red = cm.load(Ch1_file)
    m_green = cm.load(Ch2_file)
    temp_green = m_green.mean(0)
    temp_red = m_red.mean(0)

    if 'els' in Ch1_file:
        mctype='els'
    else:
        mctype='rig'
    np.savez(os.path.split(TSeries_1040)[-1]+'_Ch1_'+mctype+'_movies.npz',m_red=m_red,temp_red=temp_red,m_green=m_green,temp_green=temp_green)

#%%
if __name__=='__main__':
    main()