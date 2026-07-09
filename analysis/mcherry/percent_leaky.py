import pandas as pd
import sys,os
sys.path.extend(['/Users/amonast/Documents/GitHub/Engram_2P/Engram_2P'])
from rois.rois import remove_bad_cells
from matplotlib import rcParams
import matplotlib.pyplot as plt
import seaborn as sb

def main():
    animals=['589L',
    '989N',
    '992N',
    '992L',
    '994R',
    '9972R',
    '217R',
    '217N',
    '218L',
    '034R',
    '149L',
    '146R','160R','492N','493R','1912L']

    fov_lists=[['FOV1','FOV2'],
    ['FOV1','FOV2'],
    ['FOV2'],
    ['FOV2'],
    ['FOV1','FOV2'],
    ['FOV1','FOV2'],
    ['FOV1'],
    ['FOV1','FOV2'],
    ['FOV1','FOV2'],
    ['FOV1'],
    ['FOV1','FOV2'],
    ['FOV2'],['FOV1','FOV2'],['FOV1','FOV2'],['FOV1'],['FOV1','FOV2']]

    file_key = '/Volumes/AM_SSD1/Spont2P/Data_info.csv'
    base_dir  = '/Volumes/AM_SSD1/Spont2P'
    df_leak = pd.DataFrame()
    df = pd.DataFrame()

    for a,ani in enumerate(animals):
        fovs = fov_lists[a]
        for fov in fovs:
            inds,leak = remove_bad_cells(ani,fov,file_key,base_dir,snr_thr=4.0,return_leaky=True)
            df_leak  = pd.concat([df_leak,leak],ignore_index=True)
            df =  pd.concat([df,inds],ignore_index=True)

    reg_leak = df_leak.loc[(df_leak.Baseline!=-1)&(df_leak.Post!=-1)]
    reg = df.loc[(df.Baseline!=-1)&(df.Post!=-1)]
    n_leak_reg = reg_leak.shape[0] 
    n_good_reg = reg.shape[0] - n_leak_reg
    n_leak_all = df_leak.shape[0]
    n_good_all= df.shape[0] - n_leak_all

    reg = [n_leak_reg,n_good_reg] 
    reg_label = ['Leaky mCh+', ' '] 
    all = [n_leak_all,n_good_all] 
    all_label = ['Leaky mCh+', ' '] 

    rcParams['font.family']='Galvji'
    rcParams['svg.fonttype']='none'
    fig=plt.figure(figsize=(2,2))
    fig.add_subplot(211)
    plt.title('All cells',fontdict={'size':8})
    plt.pie(all, labels=all_label, autopct='%.2f%%',colors=['red','darksalmon'],startangle=45,textprops={'size':8}) 
    fig.add_subplot(212)
    plt.title('Registered cells',fontdict={'size':8})
    plt.pie(reg, labels=reg_label, autopct='%.2f%%',colors=['red','darksalmon'],startangle=45,textprops={'size':8}) 
    plt.show()
    plt.savefig('percent_leakyD0_cells.svg',transparent=True)


    

if __name__=='__main__':
    main()