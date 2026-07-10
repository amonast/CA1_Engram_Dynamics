import sys 
import numpy as np
import os
import ast
import importlib
import matplotlib.pyplot as plt
from rastermap import Rastermap
sys.path.extend(['/projectnb/sramirezlab/amonast/Engram_2P/Engram_2P',
                 '/Users/amonast/Documents/GitHub/Engram_2P/Engram_2P'])
from utilities import animal 
from behavior.timestamps import get_tone_frame_vector


def main():
    base_dir ='/Users/amonast/Desktop/Tone2P' #'/Volumes/AM_SSD1/Caiman'
    file_key ='/Users/amonast/Desktop/Tone2P/Data_info_TFC.csv' #'/Volumes/AM_SSD1/Caiman'
    animals = ['997B','639N', 'M2L','M1N','F5L','F7N','M8BL2','M9BR2','194L','M5L','939L']
    session='Baseline' 
    
    nPCs = 10
    lag = 15
    loc = 0.1
    time_bin = 15
    params = {'n_clusters': 0, 
                'n_splits': 0, 
                'grid_upsample': 0, 
                'mean_time': True, 
                'verbose': True, 
                'verbose_sorting': False, 
                'n_PCs': nPCs, 
                'time_lag_window': lag, 
                'locality': loc, 
                'time_bin': time_bin}
    
    fov='FOV1'
    for ani in animals:
        mouse = animal.animal(ani,fov,file_key=file_key,base_dir = base_dir)
        traces_df = mouse.load_traces(signal='events',sessions=[session])
        dff_dataframe = mouse.load_traces(signal='dff',sessions=[session])
        dff = dff_dataframe.iloc[:,1:].values.T
        timestamps = traces_df['timestamps']
        spks = traces_df.iloc[:,1:].T.values
        spks[spks>0]=1        
        n_neurons, n_time = spks.shape
        print(f"{n_neurons} neurons by {n_time} timepoints")
        
        ## run rastermap        
        model = Rastermap(**params).fit(spks)
        isort = model.isort
        cc_nodes = model.cc.copy()

        ## save output
        output_dir = f'{base_dir}/Analysis/rastermap_batch/baseline/nPCs{nPCs}_lag{lag}_loc{loc}_bin{time_bin}'#os.path.dirname(params_file)
        os.makedirs(output_dir,exist_ok=True)
        # Save plots to that folder
        plt.figure(figsize=(10, 10), dpi=300)
        plt.imshow(cc_nodes, aspect='auto')
        plt.colorbar()
        plt.title('Asymmetric similarity matrix')
        plt.savefig(os.path.join(output_dir, f"{ani}_{fov}_{session}_similarity_matrix.png"), bbox_inches='tight')

        np.save(f"{output_dir}/{ani}_{session}_matrix_.npy",cc_nodes)
        np.save(f"{output_dir}/{ani}_{session}_isort_.npy",isort)
        np.savez(f"{output_dir}/{ani}_{session}_Rastermap_.pkl",model)
        
        fig,ax=plt.subplots(nrows=2,figsize=(15,25),dpi=300)
        ax[0].imshow(dff[isort],vmin=0,vmax=3,aspect='auto',cmap='magma')
        ax[1].imshow(spks[isort],vmin=0,vmax=1,aspect='auto',cmap='magma')
        plt.title('sorted activity')
        plt.savefig(os.path.join(output_dir, f"{ani}_{fov}_{session}_sorted_activity.png"), bbox_inches='tight')


        plt.figure(figsize=(10,10),dpi=300)
        plt.imshow(cc_nodes,aspect='auto',cmap='coolwarm')
        plt.colorbar()
        plt.title('Asymmetric similiarity matrix')
        plt.savefig(os.path.join(output_dir, f"{ani}_{fov}_{session}_similarity_matrix.png"), bbox_inches='tight')
        plt.close('all')

if __name__=='__main__':
    main()
