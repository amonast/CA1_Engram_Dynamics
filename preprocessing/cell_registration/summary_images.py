#%%
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import pickle
import warnings
warnings.filterwarnings("ignore", message=r"Passing", category=FutureWarning)
import sys
sys.path.insert(0,'/projectnb/sramirezlab/amonast/Engram_2P/Engram_2P')
from spatial.rois import save_summary_images_batch
import pandas as pd
import numpy as np
#%%
def main():
    base_dir='/projectnb/sramirezlab/amonast/Tone2P'
    file_key='/projectnb/sramirezlab/amonast/Tone2P/Data_info_TFC.csv'
    animals = ['939L']
    df = pd.read_csv(file_key)

    for ani in animals:
        fovs = np.unique(df.FOV.loc[df.Animal==ani].values)
        for fov in fovs:
            folder = os.path.join(base_dir,'CellReg', ani+'_'+fov)
            os.makedirs(folder,exist_ok=True)

    for ani in animals:
        fovs = np.unique(df.FOV.loc[df.Animal==ani].values)
        for fov in fovs:
            ani_dir = os.path.join(base_dir,'GCaMP',ani)
            os.makedirs(folder,exist_ok=True)
            TSeries_all = [os.path.join(ani_dir,folder) for folder in os.listdir(ani_dir) if ('caiman_output' not in folder)]
            print(TSeries_all)
            TSeries_all.sort()
            save_summary_images_batch(TSeries_all,file_key,base_dir)

if __name__=='__main__':
    main()