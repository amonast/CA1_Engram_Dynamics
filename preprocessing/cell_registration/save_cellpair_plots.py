import numpy as np
import caiman as cm
import os
import matplotlib.pyplot as plt
from caiman.source_extraction.cnmf import cnmf as cnmf
from caiman.utils.visualization import get_contours,plot_contours
import sys 
sys.path.extend(['/Users/amonast/Documents/GitHub/Engram_2P/Engram_2P'])
from visualization.plotting import roi_plot,load_mch_image
#from Engram_2P.CNMF import get_traces,get_deconv_traces
import pandas as pd
import holoviews as hv
hv.extension('bokeh')
import panel as pn
from bokeh.resources import INLINE
import tifffile
import itertools
from collections import defaultdict

def main():
    base_dir = '/Volumes/AM_SSD3/Tone2P'
    file_key = base_dir + '/Data_info_TFC.csv'
    ani,fov='639N','FOV1'
    # for a,ani in enumerate(animals):
    #     for fov in FOVs[a]:
    save_roi_spatial_plots(ani,fov,file_key,base_dir,mch_ref='Recall1')


def save_roi_spatial_plots(ani,fov,file_key,base_dir,mch_ref='Recall1',n_s_common=2):
    '''
    ani animal string
    fov, 'FOV1' 
    file_key: csv with session filenames
    base_dir: home directory
    mch_ref: name of session for mCherry image (post-tag)
    n_s_common: number of sessions in common (i.e 2 for all combinations of 2 sessions (D0,D1),(D0,D2)... (D3,D4), 
                                                3 all combinations of 3 sessions)(D0,D1,D2), (D0,D1,D3)....
    '''
    os.makedirs(os.path.join(base_dir,'ROIs','roi_maps'),exist_ok=True)
    info = pd.read_csv(file_key)
    ind_csv=os.path.join(base_dir,'Tagging',ani+'_'+fov+'_indices_split.csv')
    h5path = os.path.join(base_dir,'GCaMP',ani,'caiman_output','cnmf')
    sessions = info['Session'].loc[info['Animal']==ani].unique()
    TSeries_all = [info['TSeries_g'].loc[(info['Session']==session)&(info['Animal']==ani)&(info['FOV']==fov)].values[0] for session in sessions]
    h5_list = [os.path.join(h5path,f) for f in os.listdir(h5path) if (('hdf5' in f)&(f.startswith('.') == False))]
    h5_list_sorted = sorted(h5_list,key=lambda fname: next((i for i, t in enumerate(TSeries_all) if t in fname), 
                                                           len(TSeries_all))
)
    cnmf_all = [cnmf.load_CNMF(h) for h in h5_list_sorted]

    mch_info = info.loc[(info['Animal']==ani)&(info['FOV']==fov)]
    TSeries_mch = mch_info['TSeries_mch'].loc[(mch_info['Session']==mch_ref)].values[0]
    mch_im = load_mch_image(ani,fov,mch_ref,file_key,base_dir)
    ind_df=pd.read_csv(ind_csv,index_col=0)

    flip='y' # 'y' to flip vertically; x to flip horizontally, None to ignore
    try:
        im_paths=[os.path.join(base_dir,'GCaMP',ani,'caiman_output','motion_correction',f"{TSeries}-AVG-template.tif") for TSeries in TSeries_all]
        mean_list = [tifffile.imread(path).astype(np.float32) for path in im_paths]

    except:
        im_paths = [os.path.join(base_dir,'CellReg',ani+'_'+fov,session+'_summary_images.npz') for session in sessions]
        mean_list = [np.load(path)['mean_im'].astype(np.float32) for path in im_paths]

    # path = [os.path.join(base_dir,'CellReg',ani+'_'+fov,day+'_summary_images.npz') for day in ['D0','D1','D2','D3','D4']]
    corr_list = [c.estimates.Cn for c in cnmf_all]
    #images_list = [[mean_list[i], corr_list[i],mch_im] for i in range(len(sessions))]
    # roi_list = [ind_df['Baseline'].loc[(ind_df['Baseline']!=-1)&(ind_df['Post']!=-1)].values,
    #             ind_df['Post'].loc[(ind_df['Baseline']!=-1)&(ind_df['Post']!=-1)].values]
    # roi_pre = ind_df['Baseline'].loc[(ind_df['Baseline']!=-1)&(ind_df['Post']==-1)].values
    # roi_post = ind_df['Post'].loc[(ind_df['Baseline']==-1)&(ind_df['Post']!=-1)].values
    hv.output(size=300)

    # Dictionary to store ROIs by the number of sessions they appear in, along with session data
    roi_grouped = defaultdict(list)

    num_sessions = len(sessions)

    # Add A column for global index
    ind_df = ind_df.reset_index()
    ind_df['A-idx'] = ['A' + str(i) for i in ind_df.index]

    # Copy the dataframe
    sorted_ind_df = ind_df.copy()

    # Dictionary to store ROIs grouped by session presence
    roi_grouped = defaultdict(list)

    # Populate roi_grouped
    for idx in range(ind_df.shape[0]):
        present_in_sessions = [session for session in sessions if ind_df.at[idx, session] != -1]
        
        if present_in_sessions:
            roi_data = {
                'roi_index': ind_df.at[idx, 'A-idx'],
                'sessions': present_in_sessions,
                'values': {session: ind_df.at[idx, session] for session in present_in_sessions}
            }
            roi_grouped[len(present_in_sessions)].append(roi_data)

    # Flatten roi_grouped to get the sorted order
    sorted_a_indices = [roi['roi_index'] for k in sorted(roi_grouped.keys(), reverse=True) for roi in roi_grouped[k]]

    # Sort the dataframe by the new order
    sorted_ind_df = sorted_ind_df.set_index('A-idx').loc[sorted_a_indices].reset_index(drop=True)
    sorted_ind_df.to_csv(f"{base_dir}/Tagging/{ani}_{fov}_reg_indices_sort.csv")
    # Display the sorted dataframe
    print(sorted_ind_df)
    # Print the grouped ROIs with their session-specific data
    for k in range(num_sessions, 0, -1):
        print(f"ROIs common in {k} sessions:")
        for roi in roi_grouped[k]:
            print(f"  {roi['roi_index']} - Sessions: {roi['sessions']} - Values: {roi['values']}")
    
    # ########## save single session rois ################
       # Save plots for single-session ROIs, one grid per session
    single_session_df = ind_df[(ind_df[sessions] != -1).sum(axis=1) == 1]
    for j, session in enumerate(sessions):
        cnm = cnmf_all[j]
        images_set = [mean_list[j], corr_list[j], mch_im]
        idxs = single_session_df[single_session_df[session] != -1][session].values
        if len(idxs) == 0:
            continue

        n_rois = len(idxs)
        plot_i = [(i, i + 50) for i in np.arange(0, n_rois, 50)]

        for start, stop in plot_i:
            gridspace = hv.GridSpace(kdims=['Images'], group='ROI', label='Neuron')
            for i, im in enumerate(images_set):
                min_pct = 5 if i != 0 else 0
                max_pct = 98
                holomap = hv.HoloMap(kdims='ROI Index')
                for k, idx in enumerate(idxs[start:stop]):
                    panel = roi_plot(cnm, idx, im, min_pct=min_pct, max_pct=max_pct, roi_style='contour', line_color='m')
                    text = hv.Text(10, 20, f"ROI {idx}").opts(text_color='magenta', fontsize=10)
                    holomap[k] = panel * text
                gridspace[i] = holomap

            plotname = f"{base_dir}/ROIs/roi_maps/{ani}_{fov}_roimap_single_session_{session}_{start}-{stop}.html"
            pn.pane.HoloViews(gridspace).save(plotname, embed=True, resources=INLINE)
            print(f"Saved: {plotname}")
        
    ########## save rois in all sessions #####################
        # Step 1: Get ROIs that are common in all sessions
    rois_common_all_sessions = roi_grouped[num_sessions]  # ROIs common in all sessions

        # Extract the indices for these ROIs across all sessions
    roi_values_all_sessions = {session: [roi['values'][session] for roi in rois_common_all_sessions] for session in sessions}

        # Step 2: Plot 50 pairs per HTML file (batching mechanism)
    n_rois = len(rois_common_all_sessions)
    plot_i = [(i, i + 50) for i in np.arange(0, n_rois, 50)]  # Plot in batches of 50

    for start, stop in plot_i:
        gridspace = hv.GridSpace(kdims=['Session', 'Images'], group='ROI', label='Neuron')

        # Step 3: For each session in ind_df
        for j, session in enumerate(sessions):
            cnm = cnmf_all[j]  # Get CNMF for this session
            images_set = [mean_list[j], corr_list[j], mch_im]  # List of images for this session
            
            # Get the ROIs for the current session from the batch (start to stop)
            idxs = roi_values_all_sessions[session][start:stop]

            for i, im in enumerate(images_set):  # Loop through images (mean, corr, mch)
                min_pct = 5 if i != 0 else 0
                max_pct = 99
                holomap = hv.HoloMap(kdims='ROI Index')

                for k, idx in enumerate(idxs):  # Loop through ROIs for the session
                    panel = roi_plot(cnm, idx, im, min_pct=min_pct, max_pct=max_pct, roi_style='contour', line_color='m')
                    text = hv.Text(10, 10, f"ROI {idx}").opts(text_color='magenta', fontsize=10)  # Add ROI label
                    holomap[k] = panel * text  # Add panel to holomap
                
                gridspace[j, i] = holomap  # Add holomap to gridspace (position by session and image type)

        # Save the plot
        plotname = f"{base_dir}/ROIs/roi_maps/{ani}_{fov}_roimap_common_all_sessions_{start}-{stop}.html"
        pn.pane.HoloViews(gridspace).save(plotname, embed=True, resources=INLINE)
        print(f"Saved: {plotname}")
    
    # ##### 2 session matches ######
    ## save rois in 2 sessions 
    pairwise_combinations = list(itertools.combinations(sessions, 2))

        # Initialize the dictionary to hold the session pair-wise combinations and their corresponding ROI values
    roi_sessions_pairwise = defaultdict(list)
    roi_lists_dict = {}

        # Access ROIs that are common in exactly 2 sessions (roi_grouped[2])
        # Generate all pairwise combinations of the sessions (session_combinations)
    for session_pair in itertools.combinations(sessions, 2):
        # Iterate over the ROIs common in exactly 2 sessions
        for roi in roi_grouped[2]:
            # Check if the current pair of sessions are in the 'sessions' list of the ROI
            if set(session_pair).issubset(set(roi['sessions'])):
                # Retrieve the corresponding values for each session in the pair
                roi_values_in_pair = [roi['values'][session] for session in session_pair]
                    
                # Store the roi values for the session pair
                roi_sessions_pairwise[session_pair].append(roi_values_in_pair)
  
                session1_values, session2_values = map(np.array, zip(*roi_sessions_pairwise[session_pair]))
                roi_lists_dict[session_pair]=[session1_values,session2_values]
    
    session_index_map = {session: idx for idx, session in enumerate(sessions)}

    ## Save reg paired cells plots
    # For each pair of 2 sessions,
    for I,(key,value) in enumerate(roi_lists_dict.items()):
        # get the corresponding images for the session pair,
        session_indices = (session_index_map[key[0]],
                            session_index_map[key[1]])
        mean_image_pair = [mean_list[session_indices[0]],
                            mean_list[session_indices[1]]]
        corr_image_pair = [corr_list[session_indices[0]],
                            corr_list[session_indices[1]]]
        images_list = [[m,c,mch_im] for m,c in zip(mean_image_pair,corr_image_pair)]
        
        #get the footprints
        cnmf_pair = [cnmf_all[session_indices[0]],
                        cnmf_all[session_indices[1]]]

        npairs = len(value[0])
        plot_i = []
        for i in np.arange(0,npairs,50): #plot in batches of 50
            plot_i.append((i,i+50))
        
        for p in plot_i:
            start,stop=p
            gridspace = hv.GridSpace(kdims=['Images', 'Session'], group='ROI', label='Neuron')
            for j, image_set in enumerate(images_list):
                cnm = cnmf_pair[j]
                idxs = value[j] 
                for i, im in enumerate(image_set):
                    min_pct=5 if i!=0 else 0
                    max_pct=98
                    holomap = hv.HoloMap(kdims='registration pair index')
                    for k, idx in enumerate(idxs[start:stop]):
                        panel = roi_plot(cnm, idx, im, min_pct=min_pct, max_pct=max_pct,roi_style='contour',line_color='m')
                        text = hv.Text(10, 10, f"ROI {idx}").opts(text_color='magenta', fontsize=10)
                        holomap[k] = panel*text
                    gridspace[i, j] = holomap
            plotname=f"{base_dir}/ROIs/roi_maps/{ani}_{fov}_roimap_regpairs_{key}_{start}-{stop}"
            pn.pane.HoloViews(gridspace).save(plotname,embed=True, resources=INLINE)

if __name__=='__main__':
    main()