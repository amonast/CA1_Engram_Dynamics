import numpy as np
import warnings
warnings.filterwarnings("ignore", message=r"Passing", category=FutureWarning)
import os
import pandas as pd
import scipy.io as sio
import glob
import scipy.sparse as sp
try:
    import caiman as cm
    from caiman.source_extraction.cnmf import cnmf as cnmf
except ImportError:
    ImportError('did not find caiman, need it to save footprints from caiman results')
import matplotlib.pyplot as plt
__all__=["reshape_ROIs"]
# %%
def main():
    base_dir = '/Volumes/AM_SSD3/Tone2P'  # path with animal subfolders of gcamp data and caiman output
    file_key ='/Volumes/AM_SSD3/Tone2P/Data_info_TFC.csv'
    metadata = pd.read_csv(file_key)
    #animals = [os.path.join(base_dir, ani) for ani in os.listdir(base_dir)]
    animals=['939L']
    animals_path = [os.path.join(base_dir,'GCaMP',a) for a in animals]
    for ani in animals_path:
        animal = os.path.split(ani)[-1]
        # TSeries_all = [os.path.join(ani, T) for T in os.listdir(ani) if 'TSeries' in T]
        TSeries_all = [os.path.join(ani,TSer) for TSer in metadata['TSeries_g'].loc[metadata['Animal']==animal].values]
        print(TSeries_all)
        for TSeries in TSeries_all:
            print(TSeries)
            fov = metadata['FOV'].loc[metadata['TSeries_g']==os.path.split(TSeries)[-1]].values[0]
            session = metadata['Session'].loc[metadata['TSeries_g']==os.path.split(TSeries)[-1]].values[0]
            reshape_ROIs(TSeries, base_dir,animal,fov,session,save_mat=True)
# %%
# only one hdf5 for each session is permitted in each TSeries subfolder. dont have multiple cnmf outputs in same folder.
def load_cnmf(TSeries):
    '''
    Loads saved cnmf object for a session
    :param TSeries: path to TSeries folder
    :return: cnm, caiman's cnmf object
    '''
    base_path = os.path.split(TSeries)[0]
    T = os.path.split(TSeries)[-1]
    cnmf_path = os.path.join(base_path, 'caiman_output', 'cnmf')
    cnmf_file = [os.path.join(cnmf_path,f) for f in os.listdir(cnmf_path) if (T in f)&('.hdf5' in f)]
    cnm = cnmf.load_CNMF(cnmf_file[0])
    print(cnmf_file[0])
    return cnm

def reshape_ROIs(TSeries, base_dir=None,ani=None,FOV=None,session=None,save_mat: bool = False,dims=None):
    '''
    Reshapes sparse matrix A from cnmf output to N x Y x X dimensions for CellReg in MATLAB.
    TSeries: str, path to TSeries folder 
            OR A, sparse csc matrix from caiman estimates.
    save_mat: bool, set True to write individual .mat file with footprints saved
    returns: A in  N x Y x X dimensions
    '''
    if type(TSeries)==str:
        cnm = load_cnmf(TSeries)
        As = cnm.estimates.A
        dims = cnm.dims
        
        base_path = os.path.split(TSeries)[0]
        T = os.path.split(TSeries)[-1]
        cnmf_path = os.path.join(base_path, 'caiman_output', 'cnmf')
        cnmf_file = [os.path.join(cnmf_path, f) for f in os.listdir(cnmf_path) if T in f]
        A_list = [np.reshape(As[:, i].toarray(), dims, order='F') for i in range(As.shape[1])]
        A = np.asarray(A_list)

        if save_mat:  # if you want to save
            append = os.path.split(cnmf_file[0])[-1].split('.')[0].split('_')[-2] + '_' + \
                    os.path.split(cnmf_file[0])[-1].split('.')[0].split('_')[-1]
            savedir = os.path.join(base_dir, 'CellReg', str(ani) + '_' + str(FOV))

            matfile = os.path.join(savedir,f"{ani}_{T}_{FOV}_{session}_A_{append}.mat")
            if not os.path.exists(savedir):
                print('Made directory '+ savedir)
                os.makedirs(savedir)
            sio.savemat(matfile, {'A': A})
            print('saved' + matfile)

        
    elif sp.isspmatrix_csc(TSeries):
        
        Asp = TSeries
        A_list = [np.reshape(Asp[:, i].toarray(), dims, order='F') for i in range(Asp.shape[1])]
        A = np.asarray(A_list)

    
    return A


# %%
if __name__ == "__main__":
    main()
