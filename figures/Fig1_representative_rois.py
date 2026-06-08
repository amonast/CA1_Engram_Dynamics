
#%%
import pandas as pd 
import sys,os
sys.path.extend(['/Users/amonast/Documents/GitHub/Engram_2P/Engram_2P/preprocess'])
from save_all_rois_fiji import write_roi_zip
from rois import remove_bad_cells
from traces import load_cnmf
from caiman.utils.visualization import get_contours
import numpy as np
#%%
base_dir = '/Volumes/AM_SSD1/Spont2P'
file_key = '/Volumes/AM_SSD1/Caiman/Data_info.csv'
info = pd.read_csv(file_key)
ani = '989N'
fov='FOV1'

ind_df = remove_bad_cells(ani,fov,snr_thr=4.0,file_key=file_key,base_dir=base_dir)
#%%
mch_post = ind_df['Post'].loc[ind_df['Tagged']==1].values
mch_post = mch_post[mch_post!=-1]
non_post = ind_df['Post'].loc[ind_df['Tagged']==0].values
non_post = non_post[non_post!=-1]

TSeries_all = [info['TSeries_g'].loc[(info['Session']=='Baseline')&(info['Animal']==ani)&(info['FOV']==fov)].values[0],
                info['TSeries_g'].loc[(info['Session']=='Post')&(info['Animal']==ani)&(info['FOV']==fov)].values[0]]

############# get rois 
cnms = [load_cnmf(os.path.join(base_dir,'GCaMP',ani,TSeries)) for TSeries in TSeries_all]

coord_post = get_contours(cnms[1].estimates.A,dims = cnms[1].estimates.dims)
coords = []
for a in range(len(coord_post)):
    coord_b= coord_post[a]['coordinates'][1:-1]
    shape = coord_b[~np.isnan(coord_b)].shape[0]
    coord = coord_b[~np.isnan(coord_b)].reshape(int(shape/2),2)
    coords.append(coord)
#%% write all post rois 
save_path = '/Users/amonast/Dropbox (BOSTON UNIVERSITY)/Manuscripts/Engram2P/Figures/Figure1_method'
write_roi_zip(coords_all=coords,idxs=mch_post,path=save_path,name='post-mch',ani=ani,fov=fov,session='Post')
write_roi_zip(coords_all=coords,idxs=non_post,path=save_path,name='post-non',ani=ani,fov=fov,session='Post')


# %% get registered cell rois pre and post 
save_path = '/Users/amonast/Dropbox (BOSTON UNIVERSITY)/Manuscripts/Engram2P/Figures/Figure1_method/representative_cells'
mch_post_reg = ind_df['Post'].loc[(ind_df['Baseline']!=-1)&(ind_df['Post']!=-1)&(ind_df['Tagged']==1)].values
non_post_reg = ind_df['Post'].loc[(ind_df['Baseline']!=-1)&(ind_df['Post']!=-1)&(ind_df['Tagged']==0)].values

mch_pre_reg = ind_df['Baseline'].loc[(ind_df['Baseline']!=-1)&(ind_df['Post']!=-1)&(ind_df['Tagged']==1)].values
non_pre_reg = ind_df['Baseline'].loc[(ind_df['Baseline']!=-1)&(ind_df['Post']!=-1)&(ind_df['Tagged']==0)].values

coord_post = get_contours(cnms[1].estimates.A,dims = cnms[1].estimates.dims)
coords = []
for a in range(len(coord_post)):
    coord_b= coord_post[a]['coordinates'][1:-1]
    shape = coord_b[~np.isnan(coord_b)].shape[0]
    coord = coord_b[~np.isnan(coord_b)].reshape(int(shape/2),2)
    coords.append(coord)

write_roi_zip(coords_all=coords,idxs=mch_post_reg,path=save_path,name='post-mch-reg',ani=ani,fov=fov,session='Post')
write_roi_zip(coords_all=coords,idxs=non_post_reg,path=save_path,name='post-non-reg',ani=ani,fov=fov,session='Post')

coord_pre = get_contours(cnms[0].estimates.A,dims = cnms[1].estimates.dims)
coords = []
for a in range(len(coord_pre)):
    coord_b= coord_pre[a]['coordinates'][1:-1]
    shape = coord_b[~np.isnan(coord_b)].shape[0]
    coord = coord_b[~np.isnan(coord_b)].reshape(int(shape/2),2)
    coords.append(coord)
write_roi_zip(coords_all=coords,idxs=mch_pre_reg,path=save_path,name='pre-mch-reg',ani=ani,fov=fov,session='Baseline')
write_roi_zip(coords_all=coords,idxs=non_pre_reg,path=save_path,name='pre-non-reg',ani=ani,fov=fov,session='Baseline')
