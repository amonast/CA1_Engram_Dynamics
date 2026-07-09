#%%
import os
import glob
import pickle
import warnings
warnings.filterwarnings("ignore", message=r"Passing", category=FutureWarning)

__all__ = ["get_h5","write_dff_file"]
import caiman as cm
from caiman.source_extraction.cnmf import cnmf as cnmf

def main():
    base_dir = '/projectnb/sramirezlab/amonast/Tone2P'
    savepath = os.path.join(base_dir, 'deconvolution', 'dff_files_500')
    os.makedirs(savepath,exist_ok=True)
    h5path = get_h5(os.path.join(base_dir,'GCaMP'),['939L'])
    #h5path = [os.path.join('/Users/amonast/Desktop/Caiman/hdf5_results',f) for f in os.listdir('/Users/amonast/Desktop/Caiman/hdf5_results') if '.hdf5' in f]
    for h in h5path:
        print(h)
        write_dff_file(h,savepath=savepath,rewrite_dff=False,frames_window=500,overwrite_h5=False)

#%%
def get_h5(base_dir,animals):
    h5_all = []
    for ani in animals:
        cnmf_path = os.path.join(base_dir,ani,'caiman_output','cnmf')+os.path.sep
        h5files= glob.glob(os.path.join(cnmf_path,'*.hdf5'))
        [h5_all.append(h) for h in h5files]
    return h5_all
#%%
def write_dff_file(f, frames_window=None,rewrite_dff=False,overwrite_h5=False,savepath:str=None):
    '''
    :param f: hdf5 cnmf output file
    :param rewrite_dff: False, True if want to reextract the df/f from the caiman results file
    :param frames_window: if rewrite dff, specific frames window for df/f calculation
    :overwriteh5: if rewrite dff, overwrite the original h5 results file! it not, saves as same filename in savepath
    :param savepath: where to save pkl file with dff and file name, and h5 file name if overwrite h5 is false
    :return: filename of pickle file with df/f traces
    '''
    cnm  = cnmf.load_CNMF(f)
    h5name = os.path.split(f)[-1]
    try:
        if cnm.estimates.F_dff==None:
            cnm.estimates.detrend_df_f(quantileMin=8, frames_window=250)
    except ValueError:
        pass
    if rewrite_dff:
        cnm.estimates.detrend_df_f(quantileMin=8, frames_window=frames_window)
        try:
            if overwrite_h5:
                print('!rewrote df/f in camain results hdf5 file: '+f)
                cnm.save(f)
            else:
                cnm.save(os.path.join(savepath,h5name))
                print('saved new caiman results hdf5 file: '+ os.path.join(savepath,h5name))
        except ValueError:
            print('couldnt resave h5 file')
            pass
    dff = cnm.estimates.F_dff
    fid = h5name.split('_')[-3] + '_' + h5name.split('_')[-2] + '_' + h5name.split('_')[-1]
    animal = h5name.split('_')[0]
    TSeries = h5name.split('_')[1]
    dff_dict = {'animal': animal, 'fid': fid, 'TSeries': TSeries, 'dff': dff}
    append = '_dff.pkl'
    if savepath:
        fname = os.path.join(savepath, h5name.split('.')[0] + append)
    elif savepath is None:
        fname = os.path.join(os.path.split(f)[0], h5name.split('.')[0] + append)

    with open(fname, 'wb') as p:
        print('Saving as '+fname)
        pickle.dump(dff_dict, p, protocol=pickle.HIGHEST_PROTOCOL)
    return fname

#%%

if __name__ == '__main__':
    main()


