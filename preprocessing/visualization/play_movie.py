import tkinter.filedialog as fd
import warnings
warnings.filterwarnings("ignore", message=r"Passing", category=FutureWarning)
import caiman as cm
import os
import sys
from glob import glob
import re
def main():
    """
    Interactively play a motion-corrected or raw imaging movie via a folder picker.

    Prompts the user to select a TSeries folder, optionally overrides
    playback/motion-correction settings from command-line arguments, locates
    the appropriate movie files (motion-corrected `.mmap` or raw `.tif`,
    single-channel GCaMP or two-channel GCaMP/mCherry), and plays them back
    using CaImAn's movie player.

    Parameters
    ----------
    None
        Reads optional overrides from `sys.argv`: 'mc' or 'raw' to force
        motion-corrected vs. raw playback, 'mcherry' to switch to two-color
        mode, and `nmc`/`mag`/`ds`/`fr` (each followed by its value) to set
        the number of motion-correction rounds, magnification, downsample
        ratio, and frame rate respectively.

    Returns
    -------
    None
        Plays the movie via CaImAn's `.play()` (press 'q' to exit); does not
        return a value.

    Notes
    -----
    `gcamp_920`, `mc`, `n_mc`, `mag`, `fr`, and `ds_ratio` are hardcoded
    defaults within the function body, and may be overridden by CLI
    arguments as described above.
    """
    gcamp_920=True #False for 2 color data
    mc =True #True for mmap files, false for tif files
    n_mc=2 #number of rounds of motion correction done on the mmap files
    mag =2#magnification
    fr=60 #frame rate
    ds_ratio=.2#mcherry/shorter videos .5, long videos .1
    
    #images to play
    series = fd.askdirectory(initialdir='/projectnb/sramirezlab/amonast/Tone2P')
    
    #check for command line arguments 
   
    if len(sys.argv)>1:
        #for command line running the command line arguments should be mc
        if 'mc' in sys.argv:
            mc=True
        if 'raw' in sys.argv:
            mc=False
        if 'mcherry' in sys.argv:
            gcamp_920=False
        try:
            n_mc=float(return_next_arg('nmc'))
            mag=float(return_next_arg('mag'))
            ds_ratio=float(return_next_arg('ds'))
            fr=float(return_next_arg('fr'))
        except:
            pass
    print()
    #get the name of the TSeries
    if os.path.split(series)[-1].startswith('T'):
        T = os.path.split(series)[-1]  
    elif os.path.split(series)[-1].startswith('MC_MC_MC'):
        T = os.path.split(series)[-1][9:29]
    elif os.path.split(series)[-1].startswith('MC_MC'):
        T = os.path.split(series)[-1][6:26]
    else:
        T = os.path.split(series)[-1][3:23]
    print("Loading "+series.split(os.path.sep)[-1])

    ### this is the default suffix appended by Caiman to the mmap motion corrected files
    append = '???__d1_???_d2_???_d3_?_order_F_frames_????'

    #this stk_ prefix depends on how image files are preprocessed, for long gcamp data
    if n_mc ==1:
        pattern=f"{series}/stk_????_{T}_{append}.mmap"
    elif n_mc==2:
        pattern=f"{series}/stk_????_{T}_{append}_{append}.mmap"
    elif n_mc==3:
        pattern=f"{series}/stk_????_{T}_{append}_{append}_{append}.mmap"

    if 'dview' in locals():
        cm.stop_server(dview=dview)
    c, dview, n_processes = cm.cluster.setup_cluster(backend='local', n_processes=None, single_thread=False)

    if gcamp_920: #gcamp data
        if mc:
            print('playing mmap files') #get mition corrected files
            mmaps= glob(pattern)
            fnames = [os.path.join(series,f) for f in mmaps]
            fnames.sort()
        else:
            print('playing tif files') #get raw images
            if gcamp_920:
                tifs = [os.path.join(series,f) for f in os.listdir(series) if ('tif' in f)]
            else:
                tifs = [f"{series}+{os.path.sep}+*_Ch1.tif",f"{series}+{os.path.sep}+*_Ch2.tif"]
            fnames = [os.path.join(series,f) for f in tifs]
            try:
                fnames = sorted(fnames, key=lambda p: int(re.search(r'file(\d+)_chan0', p).group(1)))
            except:
                fnames.sort()
        
        ## last effort: if cant find the files just play whatever mmaps are in the folder
        for f in fnames:
            print(f)
        if len(fnames)==0:
            fnames = [os.path.join(series,f) for f in os.listdir(series) if (f.endswith('.mmap'))&('order_C' not in f)]
            fnames.sort()

        movie = cm.load_movie_chain(fnames)
        moviehandle = movie.resize(1, 1, ds_ratio).play(q_min=0.1,q_max=99.5, gain=1,fr=fr, magnification=mag)
    else: #mcherry data: 
        if mc:
            print('playing mmap files')
            mmaps = [os.path.join(series,f) for f in os.listdir(series) if ('order_F_' in f)&('.mmap' in f)]
            mmaps = [m for m in mmaps if ('mmap' in m) &('els__' in m)]
            fnames = [os.path.join(series,f) for f in mmaps]
            fnames.sort()
        else:
            print('playing tif files') 
            #Bruker compatible only:
            tifs = [os.path.join(series,f) for f in os.listdir(series) if ('Ch1.tif' in f) or ('Ch2.tif' in f)]
            fnames = [os.path.join(series,f) for f in tifs]
            fnames.sort()

        for f in fnames:
            print(f)
        movie_ch1 = cm.load(fnames[0])
        movie_ch2 = cm.load(fnames[1])
        movie = cm.concatenate([movie_ch1.resize(1, 1, ds_ratio),
                            movie_ch2.resize(1, 1, ds_ratio)], axis=2).play(fr=fr, gain=1, magnification=mag,
                                                                            offset=0,q_min=0.01,q_max=99.5)  # press q to exit

    cm.stop_server(dview=dview)

def return_next_arg(keyword):
    """
    Get the value following a keyword in the command-line arguments.

    Parameters
    ----------
    keyword : str
        Keyword to search for in `sys.argv`.

    Returns
    -------
    str or None
        The argument immediately following `keyword` in `sys.argv`, if
        `keyword` is present and has a following argument; otherwise None.
    """
    if keyword in sys.argv:
        index = sys.argv.index(keyword)
        try:
            next_arg = sys.argv[index + 1]
            print(f"The argument following '{keyword}' is: {next_arg}")
            return next_arg
        except IndexError:
            print(f'{keyword} not given in command line, using defaults')
    else:
        print(f'{keyword} not given in command line, using defaults')

if __name__=='__main__':
    main()