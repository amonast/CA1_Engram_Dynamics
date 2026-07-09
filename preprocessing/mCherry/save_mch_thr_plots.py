import caiman as cm
import pandas as pd
import tifffile
import os
import numpy as np

from caiman.source_extraction.cnmf.cnmf import load_CNMF
from mcherry_functions import load_mch_image,mch_int
import sys
sys.path.extend(['/Users/amonast/Documents/GitHub/Engram_2P/Engram_2P'])
from rois.rois import get_h5,load_cnmf
from visualization.plotting import get_mask_rois,rois_plot
import matplotlib.pyplot as plt
import holoviews as hv
from bokeh.resources import INLINE
import panel as pn
hv.extension('bokeh')

def main():
    """
    Explore mCherry intensity histograms and plot tagging threshold overlays for one animal/FOV.

    Loads registered cell indices and mCherry intensities for cells unique
    to Baseline or the post session, and either plots a histogram of
    mCherry intensities (`plot_thr=False`) or, for a range of candidate
    threshold values, plots each threshold line over the histogram along
    with ROI contour overlays split into above/below-threshold groups
    (`plot_thr=True`), saving each as an interactive HTML panel.

    Workflow
    --------
    1. Run with `plot_thr=False` to save a histogram of mCherry intensities
       and inspect it.
    2. Choose a threshold range and step size, then re-run with
       `plot_thr=True` to visualize candidate thresholds.

    Saves output HTML panels to
    `{base_dir}/Tagging/mcherry_threshold_plots/{ani}_{fov}/`.
    """
    base_dir = '/projectnb/sramirezlab/amonast/Tone2P'
    file_key = '/projectnb/sramirezlab/amonast/Tone2P/Data_info_TFC.csv'
    ani = '939L'
    fov='FOV1'
    plot_thr=True
    range_ = [35,55]
    step = 5
    session_post='Recall1'
    sessions=['Baseline','Recall1','Recall2']
    save_path = os.path.join(base_dir,'Tagging','mcherry_threshold_plots',ani+'_'+fov)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    info = pd.read_csv(file_key)
    ind_csv=os.path.join(base_dir,'Tagging','CellReg_output',ani+'_'+fov+'_reg_indices.csv')
    TSeries_all = [info['TSeries_g'].loc[(info['Session']==s)&(info['Animal']==ani)&(info['FOV']==fov)].values[0] for s in sessions]


    mch_info = info.loc[(info['Animal']==ani)&(info['FOV']==fov)]
    TSeries_mch = mch_info['TSeries_mch'].loc[(mch_info['Session']==session_post)].values[0]
    #mch_path = os.path.join(base_dir,'mCherry',ani,TSeries_mch,TSeries_mch+'_Ch1_els_.tif')

    ind_df=pd.read_csv(ind_csv)
    cnms = [load_cnmf(os.path.join(base_dir,'GCaMP',ani,TSeries)) for TSeries in TSeries_all]
    #mch_im = tifffile.imread(mch_path)


    #mch_im_pre = load_mch_image(ani,fov,'Baseline',file_key,base_dir,channel='red')
    mch_im_post = load_mch_image(ani,fov,session_post,file_key,base_dir,channel='red')
    cnms_list = [cnms[1],cnms[1],cnms[0]]

    mch_ints_post = mch_int(cnms[1],mch_im_post,idx=get_roi_inds(ind_df,'Only',session_post))
    mch_ints_post_reg= mch_int(cnms[1],mch_im_post,idx=get_roi_inds(ind_df,'Both',session_post))
    post_mch_ints = np.concatenate([mch_ints_post,mch_ints_post_reg])

    pre_mch_ints = mch_int(cnms[0],mch_im_post,idx=get_roi_inds(ind_df,'Only','Baseline'))

    post_rois = np.concatenate([get_roi_inds(ind_df,'Only',session_post),get_roi_inds(ind_df,'Both',session_post)])
    pre_rois = get_roi_inds(ind_df,'Only','Baseline')

    mch_ints = np.concatenate([mch_ints_post,mch_ints_post_reg,pre_mch_ints])
    frequencies, edges = np.histogram(mch_ints, 50)
    
    if not plot_thr:
        print('plotting histogram')
        int_hist = hv.Histogram((edges, frequencies))
        pn.pane.HoloViews(int_hist).save(os.path.join(save_path,ani+'_'+fov+'_'+'intensities_hist'),embed=True,resources=INLINE)

    if plot_thr:
        lines = np.arange(range_[0],range_[1],step)
        plot_dict = {}
        scale_factor = 2

        for i,line in enumerate(lines):
            hist = hv.Histogram((edges, frequencies)) * hv.VLine(line).opts(color='red')
        
            high=[]
            for ii,i in enumerate(post_rois):
                if post_mch_ints[ii]>=line:
                    high.append(True)
                else:
                    high.append(False)
            low=[not h for h in high]

            plots_post= hist+rois_plot(cnms[1],mch_im_post,max_pct=95,idxs=post_rois[low],line_color='c',roi_style='contour',scale_factor=scale_factor,display_numbers=True,text_color='blue') + rois_plot(cnms[1],mch_im_post,max_pct=95,idxs=post_rois[high],roi_style='contour',line_color='m',scale_factor=scale_factor,text_color='blue',display_numbers=True) 
            pn.pane.HoloViews(plots_post).save(os.path.join(save_path,ani+'_'+fov+'_'+session_post+'_mcherry_thr_components_Post:'+str(line)),embed=True, resources=INLINE)
        
            high_pre=[]
            for ii,i in enumerate(pre_rois):
                if pre_mch_ints[ii]>=line:
                    high_pre.append(True)
                else:
                    high_pre.append(False)
            low_pre=[not h for h in high_pre]

            plots_pre= hist+rois_plot(cnms[0],mch_im_post,max_pct=95,idxs=pre_rois[low_pre],line_color='c',roi_style='contour',scale_factor=scale_factor,display_numbers=True,text_color='blue') + rois_plot(cnms[0],mch_im_post,max_pct=95,idxs=pre_rois[high_pre],roi_style='contour',line_color='m',scale_factor=scale_factor,text_color='blue',display_numbers=True) 
            pn.pane.HoloViews(plots_pre).save(os.path.join(save_path,ani+'_'+fov+'_'+session_post+'_mcherry_thr_components_Pre:'+str(line)),embed=True, resources=INLINE)


def get_roi_inds(ind_df,roi_set,session):
    """
    Get registered ROI indices for a session, split by whether cells appear only in that session or are matched with Baseline.

    Parameters
    ----------
    ind_df : pandas.DataFrame
        Cell registration index table with a 'Baseline' column and one
        column per other session; -1 indicates the cell is absent in that
        session.
    roi_set : {'Only', 'Both'}
        'Only' returns indices for cells present in `session` but absent
        from 'Baseline' (or vice versa when `session == 'Baseline'`); 'Both'
        returns indices for cells present in both `session` and 'Baseline'.
    session : str
        Session column to get indices for.

    Returns
    -------
    idxs : ndarray
        Registration indices (from the `session` column, or 'Baseline' when
        `session == 'Baseline'`) matching the requested `roi_set` criterion.

    Notes
    -----
    When `session == 'Baseline'`, the 'Only' branch compares the 'Baseline'
    column against itself (`ind_df['Baseline'] == -1`), which is likely
    unintended — it does not actually identify cells unique to Baseline
    relative to another session.
    """
    if session == 'Baseline':
        if roi_set == 'Only':
            idxs = ind_df['Baseline'].loc[ind_df[session]==-1].values
        elif roi_set=='Both':
            idxs = ind_df['Baseline'].loc[(ind_df['Baseline']!=-1)&(ind_df[session]!=-1)].values
    else:
        if roi_set == 'Only':
            idxs = ind_df[session].loc[ind_df['Baseline']==-1].values
        elif roi_set =='Both':
            idxs = ind_df[session].loc[(ind_df['Baseline']!=-1)&(ind_df[session]!=-1)].values
    
    return idxs

if __name__=='__main__':
    main()