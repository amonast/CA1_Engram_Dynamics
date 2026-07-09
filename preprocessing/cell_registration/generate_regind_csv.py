import warnings
warnings.filterwarnings("ignore", message=r"Passing", category=FutureWarning)
import holoviews as hv
import pandas as pd
import numpy as np
import os 
import rois
from datetime import datetime
import pandas as pd
hv.extension('bokeh')

def main():
    base_dir,file_key = '/Volumes/AM_SSD3/Tone2P','/Volumes/AM_SSD3/Tone2P/Data_info_TFC.csv'
    info = pd.read_csv(file_key)
    animals = ['939L']

    for animal in animals:
        sessions = info['Session'].loc[info['Animal']==animal]
        fovs = np.unique(info['FOV'].loc[info['Animal']==animal].values)
        print(fovs)
        for fov in fovs:
            reg_ind, mean_list, std_list, cnm_list = rois.get_fov_data(animal,fov,file_key,base_dir)
            df = pd.DataFrame(reg_ind,columns=sessions)
            os.makedirs(os.path.join(base_dir,"Tagging",'CellReg_output'),exist_ok=True)
            df.to_csv(os.path.join(base_dir,"Tagging",'CellReg_output',animal+'_'+fov+'_reg_indices.csv'))

if __name__=='__main__':
    main()