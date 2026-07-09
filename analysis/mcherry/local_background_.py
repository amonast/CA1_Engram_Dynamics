#%%
import caiman as cm 
import pandas as pd
import os,sys
import numpy as np
sys.path.extend(['/Users/amonast/Documents/GitHub/Engram_2P/Engram_2P'])
from mCherry.mcherry_functions import load_mch_image
from rois.rois import load_cnmf,remove_bad_cells
from visualization.plotting import rois_plot
import matplotlib.pyplot as plt
from skimage.draw import polygon2mask
from matplotlib.patches import Polygon
from matplotlib.transforms import Affine2D
from caiman.utils.visualization import get_contours

# %%
def main():
    #set animal/data paths and save path
    base_dir = '/Volumes/AM SSD2/LOU'
    file_key = '/Volumes/AM SSD2/LOU/Data_info_LOU.csv'
    animals=['160R','492N','493R','1912L']
    fov_lists=[['FOV1','FOV2'],['FOV1','FOV2'],['FOV1'],['FOV1','FOV2']]
    
    save_path = '/Users/amonast/Desktop/Caiman/Analysis/mcherry/local_bg'
    for a,ani in enumerate(animals):
        for fov in fov_lists[a]:
            print(ani + ' '+fov)
            save_cell2bg(ani,fov,file_key,base_dir,save_path)

def save_cell2bg(ani,fov,file_key,base_dir,save_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    df_pre,df_post,df_reg = cell2bg_allcells(ani,fov,file_key,base_dir)
    df_pre.to_csv(os.path.join(save_path,ani+'_'+fov+'_local_int_pre.csv'))
    df_post.to_csv(os.path.join(save_path,ani+'_'+fov+'_local_int_post.csv'))
    df_reg.to_csv(os.path.join(save_path,ani+'_'+fov+'_local_int_regcells.csv'))

def cell2bg_allcells(ani,fov,file_key,base_dir):
    info = pd.read_csv(file_key)
    ind_csv=os.path.join(base_dir,'Tagging',ani+'_'+fov+'_indices_split.csv')
    
    #get cnmf and cell idxs 
    TSeries_all = [info['TSeries_g'].loc[(info['Session']=='Baseline')&(info['Animal']==ani)&(info['FOV']==fov)].values[0],
                info['TSeries_g'].loc[(info['Session']=='Post')&(info['Animal']==ani)&(info['FOV']==fov)].values[0]]
    cnms = [load_cnmf(os.path.join(base_dir,'GCaMP',ani,TSeries)) for TSeries in TSeries_all]
    ind_df = remove_bad_cells(ani,fov,snr_thr=4,file_key=file_key,base_dir=base_dir)
    pre = ind_df['Baseline'].loc[ind_df['Baseline']!=-1]
    pre_reg = ind_df['Baseline'].loc[(ind_df['Baseline']!=-1)&(ind_df['Post']!=-1)]
    post = ind_df['Post'].loc[ind_df['Post']!=-1]
    post_reg = ind_df['Post'].loc[(ind_df['Baseline']!=-1)&(ind_df['Post']!=-1)]

    # get mcherry images 
    mch_im_pre = load_mch_image(ani,fov,'Baseline',file_key,base_dir,channel='red')
    mch_im_post = load_mch_image(ani,fov,'Post',file_key,base_dir,channel='red')
    # baseline_im_file = os.path.join(base_dir,'CellReg',ani+'_'+fov,'Baseline_summary_images.npz')
    # post_im_file = os.path.join(base_dir,'CellReg',ani+'_'+fov,'Post_summary_images.npz')
    # mean_list = [np.load(path)['mean_im'] for path in [baseline_im_file,post_im_file]]

    #get coordinates and coms from cnmf obj and store in list 
    coord = get_contours(cnms[0].estimates.A,dims = cnms[0].estimates.dims) ## pre session
    coms_pre = []
    coords_pre = []
    for a in range(len(coord)):
        coms_pre.append(coord[a]['CoM'])
        coords_pre.append(coord[a]['coordinates'][1:-1])

    coord = get_contours(cnms[1].estimates.A,dims = cnms[1].estimates.dims) ## post session
    coms_post = []
    coords_post = []
    for a in range(len(coord)):
        coms_post.append(coord[a]['CoM'])
        coords_post.append(coord[a]['coordinates'][1:-1])

    d0_pre = []
    for i in pre: # loop over pre cells
        bg2cell_i = cell2bg_ratio(mch_im_pre,i,coords_pre,coms_pre,trans_scale=1.5)
        d0_pre.append(bg2cell_i)
    
    d0_pre_reg = []
    for i in pre_reg: # loop over pre cells
        bg2cell_i = cell2bg_ratio(mch_im_pre,i,coords_pre,coms_pre,trans_scale=1.5)
        d0_pre_reg.append(bg2cell_i)
    
    #%%
    d4_post = []
    for i in post: # loop over pre cells
        bg2cell_i = cell2bg_ratio(mch_im_post,i,coords_post,coms_post,trans_scale=1.5)
        d4_post.append(bg2cell_i)
    
    d4_post_reg = []
    for i in post_reg: # loop over pre cells
        bg2cell_i = cell2bg_ratio(mch_im_post,i,coords_post,coms_post,trans_scale=1.5)
        d4_post_reg.append(bg2cell_i)
    

    df_pre = pd.DataFrame()
    df_pre['Tagged'] = ind_df['Tagged'].loc[ind_df['Baseline'].isin(pre)]
    df_pre['Cell'] = pre
    df_pre['Relative Intensity'] = d0_pre

    df_post = pd.DataFrame()
    df_post['Tagged'] = ind_df['Tagged'].loc[ind_df['Post'].isin(post)]
    df_post['Cell'] = post
    df_post['Relative Intensity'] = d4_post

    df_reg = pd.DataFrame()
    df_reg['Tagged'] = np.hstack((ind_df['Tagged'].loc[ind_df['Baseline'].isin(pre_reg)].values,ind_df['Tagged'].loc[ind_df['Baseline'].isin(pre_reg)].values))
    df_reg['Baseline'] = np.hstack((pre_reg,pre_reg))
    df_reg['Post'] = np.hstack((post_reg,post_reg))
    df_reg['Session'] = ['Baseline']*pre_reg.shape[0]+['Post']*post_reg.shape[0]
    df_reg['Relative Intensity'] = np.hstack((d0_pre_reg,d4_post_reg))

    return df_pre,df_post,df_reg

def cell2bg_ratio(image,i,coords,coms,trans_scale=1.5):
    '''
    returns the ratio of border surrounding cell (local background) to cell mean intensity
    image:image to calculate over, red channel
    i: cell index for cnmf object
    coords: list of all coords for all cells in cnmf obj
    coms: list of all coms for all cells in cnmf obj
    trans_scale: factor to stretch cell by to generate outer mask for background calculation
    '''
    #image polygon to make logical masks
    x, y = np.mgrid[:image.shape[1],:image.shape[0]]
    coors_frame=np.hstack((x.reshape(-1, 1), y.reshape(-1,1))) # coors.shape is (4000000,2)
    image_poly = Polygon(coors_frame)

    #original coordinates in image reference frame
    xs = coords[i][:,0] 
    ys = coords[i][:,1]
    coords_o = np.c_[xs,ys]

    com = coms[i] #com of cell
    # center coordinates at 0 
    xxs = xs-com[1]
    yys = ys-com[0]
    coord_norm = np.c_[xxs,yys]
    #make polygons, get coordinates again back in image coordinate space 
    poly_o = Polygon(coord_norm , fill=False, edgecolor='red')
    poly_t = Polygon(coord_norm*trans_scale , fill=False, edgecolor='pink')

    poly_t_xs = poly_t.get_xy()[:,0]+com[1]
    poly_t_ys = poly_t.get_xy()[:,1]+com[0]
    coords_t = np.c_[poly_t_xs,poly_t_ys]
    #get logical masks + reshape into image shape
    maskinner = Polygon(coords_o).contains_points(image_poly.get_xy())
    maskouter = Polygon(coords_t).contains_points(image_poly.get_xy())
    d = int((maskinner.shape[0]-1)**.5)

    mask1 = maskinner[0:maskinner.shape[0]-1].reshape(d,d).T # cell logical mask
    mask2 = maskouter[0:maskinner.shape[0]-1].reshape(d,d).T # stretched outer cell logical mask
    mask3 = np.logical_xor(mask1,mask2) # border only 

    outer = image[mask3].mean()
    inner = image[mask1].mean()

    return inner/outer
    # animals=['589L',
    #         '989N',
    #         '992N',
    #         '992L',
    #         '994R',
    #         '9972R',
    #         '217R',
    #         '217N',
    #         '218L',
    #         '034R',
    #         '149L',
    #         '146R']

    # fov_lists=[['FOV1','FOV2'],
    #             ['FOV1','FOV2'],
    #             ['FOV2'],
    #             ['FOV2'],
    #             ['FOV1','FOV2'],
    #             ['FOV1','FOV2'],
    #             ['FOV1'],
    #             ['FOV1','FOV2'],
    #             ['FOV1','FOV2'],
    #             ['FOV1'],
    #             ['FOV1','FOV2'],
    #             ['FOV2']]
if __name__ == '__main__':
    main()
    