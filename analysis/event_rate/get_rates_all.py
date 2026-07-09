#%%

import numpy as np
import pandas as pd
import sys
sys.path.extend(['/Users/amonast/Documents/GitHub/Engram_2P/Engram_2P'])
from rois.rois import remove_bad_cells
from utilities.traces import get_traces,ev_trace,event_rate, weight_event_rate

#%%
def main():
    file_key,base_dir = '/Volumes/AM SSD2/LOU/Data_info_LOU.csv','/Volumes/AM SSD2/LOU'

    animals =  ['160R','493R','492N','1912L'] 
    
    fov_list= [['FOV1','FOV2'],
               ['FOV1'],
               ['FOV1','FOV2'],
               ['FOV1','FOV2']]

    rate_df = pd.DataFrame()
    for a,ani in enumerate(animals):
        print(ani)
        fovs = fov_list[a]
        for fov in fovs:
            print(fov)
            df_fov = rate_df_fov(ani,fov,file_key,base_dir,ind=None,weighted=True)
            rate_df=pd.concat([rate_df,df_fov],ignore_index=True)
    rate_df.to_csv('/Users/amonast/Desktop/Caiman/Analysis/rates/rates_lou_mice_df_weight_5std_filt.csv')

#%%
def rate_df_fov(ani,fov,file_key,base_dir,ind=None,weighted=False):
    '''

    :param ani: animal name
    :param fov: fov name
    :param file_key: metadata csv with TSeries info
    :param base_dir: base directory for experiment
    :param ind: array of time indices indices if you want to calculate rate based on one time period.
    :return:
    '''
    dff_pre, ev_pre, times_pre, est_pre = get_traces(ani, fov, 'Baseline', file_key, base_dir)
    dff_post, ev_post, times_post, est_post = get_traces(ani, fov, 'Post', file_key, base_dir)

    if ind is None:
        d_pre = ev_trace(np.asarray(ev_pre, dtype=object), np.asarray(times_pre, dtype=object), dff_pre.shape[1])
        d_post = ev_trace(np.asarray(ev_post, dtype=object), np.asarray(times_post, dtype=object), dff_post.shape[1])
    else:
        d_tot_pre = ev_trace(np.asarray(ev_pre, dtype=object), np.asarray(times_pre, dtype=object), dff_pre.shape[1])
        d_tot_post= ev_trace(np.asarray(ev_post, dtype=object), np.asarray(times_post, dtype=object), dff_post.shape[1])
        d_pre = d_tot_pre[:,ind]
        d_post = d_tot_post[:,ind]

    metadata = pd.read_csv(file_key)

    group = metadata.Group.loc[metadata.Animal==ani].values[0]

    #organize cell indices
    ind_df = remove_bad_cells(ani,fov,file_key,base_dir,snr_thr=4,filter_mchleak=True) #filter out the bad cells
    #cells only active in each session
    silent_post = ind_df.Post.loc[((ind_df.Baseline.values == -1) & (ind_df.Post.values != -1))].values
    silent_pre = ind_df.Baseline.loc[((ind_df.Baseline.values!=-1) & (ind_df.Post.values==-1))].values
    #registered cells
    reg_ind = np.hstack((np.expand_dims(ind_df.Baseline.loc[((ind_df.Baseline.values!=-1) & (ind_df.Post.values!=-1))].values,axis=1),
                         np.expand_dims(ind_df.Post.loc[((ind_df.Baseline.values!=-1) & (ind_df.Post.values!=-1))].values,axis=1)))
    print(str(silent_pre.shape[0])+' cells active on baseline only')
    print(str(silent_post.shape[0])+' cells active on post only')
    print(str(reg_ind.shape[0])+' cells active both days')

    #get event rates in same  order as indices
    if weighted:
        r_pre_reg = weight_event_rate(d_pre[reg_ind[:,0],:])
        r_pre_sil = weight_event_rate(d_pre[silent_pre,:])
        r_post_reg = weight_event_rate(d_post[reg_ind[:,1],:])
        r_post_sil = weight_event_rate(d_post[silent_post,:])    
    else:
        r_pre_reg = event_rate(d_pre[reg_ind[:,0],:])
        r_pre_sil = event_rate(d_pre[silent_pre,:])
        r_post_reg = event_rate(d_post[reg_ind[:,1],:])
        r_post_sil = event_rate(d_post[silent_post,:])
    #build dataframes for each cell group
    df_sil_pre = pd.DataFrame()
    df_sil_pre['Baseline'] = silent_pre
    df_sil_pre['Post'] = [-1] * silent_pre.shape[0]
    df_sil_pre['Tagged'] = ind_df.Tagged.loc[ind_df.Baseline.isin(silent_pre)].values
    df_sil_pre['Event Rate'] = r_pre_sil
    df_sil_pre['Session'] = ['Baseline']*silent_pre.shape[0]

    df_reg_pre  = pd.DataFrame()
    df_reg_pre['Baseline'] = reg_ind[:,0]
    df_reg_pre['Post'] = reg_ind[:,1]
    df_reg_pre['Tagged'] = ind_df.Tagged.loc[ind_df.Baseline.isin(reg_ind[:,0])].values
    df_reg_pre['Event Rate'] = r_pre_reg
    df_reg_pre['Session'] = ['Baseline']*reg_ind.shape[0]

    df_sil_post = pd.DataFrame()
    df_sil_post['Baseline'] = [-1] * silent_post.shape[0]
    df_sil_post['Post'] = silent_post
    df_sil_post['Tagged'] = ind_df.Tagged.loc[ind_df.Post.isin(silent_post)].values
    df_sil_post['Event Rate'] = r_post_sil
    df_sil_post['Session'] = ['Post']*silent_post.shape[0]

    df_reg_post = pd.DataFrame()
    df_reg_post['Baseline'] = reg_ind[:,0]
    df_reg_post['Post'] = reg_ind[:,1]
    df_reg_post['Tagged'] = ind_df.Tagged.loc[ind_df.Post.isin(reg_ind[:,1])].values
    df_reg_post['Event Rate'] = r_post_reg
    df_reg_post['Session'] = ['Post'] * reg_ind.shape[0]

    df_fov = pd.concat([df_reg_pre,df_sil_pre,df_sil_post,df_reg_post],ignore_index=True)
    df_fov['Animal'] = [ani]*df_fov.shape[0]
    df_fov['FOV'] = [fov] * df_fov.shape[0]
    df_fov['Group'] = [group]* df_fov.shape[0]
    df_fov= df_fov.rename(columns={'Tagged': 'Population'})
    df_fov['Population'] = df_fov['Population'].replace({1: 'Tagged', 0: 'Non-Tagged'})
    return df_fov

if __name__=='__main__':
    main()
#%%
