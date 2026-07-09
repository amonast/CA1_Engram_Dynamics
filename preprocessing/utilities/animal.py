#%%
import numpy as np
import os 
import pickle 
#import pickle5
import tkinter.filedialog as fd
import ast
import pandas as pd
import scipy.sparse as sp
import h5py
import sys
sys.path.extend(['/projectnb/sramirezlab/amonast/Engram_2P/Engram_2P'])
try:
    from rois.rois import remove_bad_cells
except:
    pass
from utilities.traces import ev_trace,bin_traces
from behavior.timestamps import get_timestamps
from behavior.running import load_position,calc_velocity,thr_velocity
#%%

class animal:
    """
    Interface for loading and manipulating one animal's preprocessed 2P imaging data.

    Provides methods for loading traces, cell registration/tagging
    indices, tone/stimulus timestamps, position and velocity data, and
    CNMF spatial components, given a consistent `base_dir` folder
    structure.

    Parameters
    ----------
    animal : str
        Animal identifier.
    fov : str, optional
        Field-of-view identifier (default 'FOV1').
    file_key : str, optional
        Path to the experiment metadata CSV. If None, a file dialog
        prompts the user to select one (default None).
    base_dir : str, optional
        Base directory for the experiment's processed data. If None, a
        directory dialog prompts the user to select one (default None).

    Attributes
    ----------
    animal : str
        Animal identifier.
    fov : str
        Field-of-view identifier.
    base_dir : str
        Base directory for the experiment's processed data.
    file_key : str
        Path to the experiment metadata CSV.
    exp_data : pandas.DataFrame
        Full experiment metadata loaded from `file_key`.
    sessions : ndarray
        Session names recorded for this animal/FOV.
    TSeries_names : list of str
        TSeries name for each session in `sessions`.
    group : str
        Experimental group for this animal.
    tone_data : dict
        Cache of loaded tone-time data, keyed by session name.
    paths : dict
        Dictionary of subdirectory paths (e.g. 'GCaMP', 'Traces',
        'Tagging') under `base_dir`.

    Notes
    -----
    Requires `base_dir` to follow the folder structure set up by the
    internal `init_paths` helper (e.g. `base_dir/Traces/animal/...` for
    trace CSVs and tone-time `.npz` files).

    Examples
    --------
    >>> mouse = animal('M1N', 'FOV1',
    ...                file_key='/Volumes/AM_SSD3/Tone2P/Data_info_TFC.csv',
    ...                base_dir='/Volumes/AM_SSD3/Tone2P')
    >>> traces_df = mouse.load_traces(sessions=['Recall1'])
    >>> df_engram, df_non = mouse.load_traces(sessions=['Recall1'], split=True)
    >>> on_times, off_times = mouse.load_tone_times(session_name='Recall1')
    """
    def __init__(self,animal,fov='FOV1',file_key=None,base_dir=None):
        self.animal=animal
        self.base_dir=base_dir
        self.fov=fov
        self.file_key=file_key
        if self.base_dir is None:
            fd.askdirectory(title='Select experiment home directory',initialdir=os.getcwd())
        if self.file_key is None:
            fd.askopenfilename(title='Select experiment data csv file',initialdir=os.getcwd())
        self.exp_data= pd.read_csv(file_key)
        self.sessions = self.exp_data['Session'].loc[(self.exp_data['Animal']==animal)&(self.exp_data['FOV'] == self.fov) ].values
        self.TSeries_names = [self.exp_data['TSeries_g'].loc[(self.exp_data['Animal'] == self.animal) & \
                                                            (self.exp_data['FOV'] == self.fov) & \
                                                            (self.exp_data['Session'] == session)].values[0] \
                                                            for session in self.sessions]
        self.group = self.exp_data['Group'].loc[self.exp_data['Animal']==self.animal].values[0]
        self.tone_data={}
        def init_paths(self):
            """Build and return the dict of subdirectory paths under `self.base_dir`."""
            path_dict={}
            path_dict['GCaMP']=os.path.join(self.base_dir,'GCaMP',self.animal)
            path_dict['mCherry']=os.path.join(self.base_dir,'mCherry',self.animal)
            path_dict['CellReg']=os.path.join(self.base_dir,'CellReg',self.animal+'_'+fov)
            path_dict['Traces']=os.path.join(self.base_dir,'Traces',self.animal)
            path_dict['Behavior']=os.path.join(self.base_dir,'Behavior',self.animal)
            path_dict['Deconvolution']=os.path.join(self.base_dir,'deconvolution')
            path_dict['Tagging']=os.path.join(self.base_dir,'Tagging')
            self.path_dict=path_dict
            return self.path_dict
        self.paths = init_paths(self)
    
    def save_traces_csv(self,signal='dff',sessions=None,savepath=None):
        """
        Compute and save deconvolution-derived traces to CSV, sorted by engram status.

        For each session, loads the deconvolution output, selects the
        requested signal, reorders cells so tagged (engram) cells come
        first followed by non-tagged cells (dropping cells flagged with
        Score == 3), and writes the result to a per-session CSV. Also
        saves tone/CS onset and offset times to a `.npz` file for TFC
        sessions other than Baseline.

        Parameters
        ----------
        signal : {'dff', 'events', 'model'}, optional
            Signal type to save: raw dF/F, deconvolved event trace, or
            the deconvolution model fit (default 'dff').
        sessions : list of str, optional
            Sessions to process; if None, uses `self.sessions` (default
            None).
        savepath : str, optional
            Directory to save the CSVs and tone-times file to; if None,
            uses `self.paths['Traces']` (default None).

        Returns
        -------
        csvs : list of str
            Paths to the saved CSV files, one per session.

        Notes
        -----
        Sets `self.dec_files` to the list of loaded deconvolution files.
        Also sets `self.dff_csvs` (if `signal == 'dff'`) or
        `self.event_csvs` (if `signal == 'events'`) to the returned list
        of CSV paths.
        """
        if sessions is None:
            sessions=self.sessions
            TSeries_names = self.TSeries_names
        else:
            TSeries_names=[self.exp_data['TSeries_g'].loc[(self.exp_data['Animal'] == self.animal) & \
                                                        (self.exp_data['FOV'] == self.fov) & \
                                                        (self.exp_data['Session'] == session)].values[0] for session in sessions]
        if savepath is None:
            savepath = self.paths['Traces']
        os.makedirs(savepath,exist_ok=True)
        
        inds = self.load_cellreg(sessions=sessions)
        deconv_path = os.path.join(self.paths['Deconvolution'],'deconvolution_results','deconv_results_min5')
        self.dec_files=[]
        csvs=[]
        for t,TSeries in enumerate(TSeries_names):
            #find deconv data
            dfile = [os.path.join(deconv_path, f) for f in os.listdir(deconv_path) if ((TSeries in f)&(f.endswith('l0deconv.pkl')))][0]
            self.dec_files.append(dfile)
            print('deconvolution file: '+ dfile)
            
            sess = sessions[t] 
            #grab deconvolution outputs 
            with open(dfile, 'rb') as file:
                data = pickle.load(file)
            dff = data['dff']
            times = data['times']
            ev = data['events']
            est = data['est']
            
            if signal=='dff':
                act=dff.copy()
            elif signal=='events':
                act=ev_trace(ev,times,T=dff.shape[1])
            elif signal=='model':
                act=est.copy()
            
            #sort activity by engram cells
            engram_dict = self.load_engram_idx() 
            engram_idxs_all = engram_dict[sess]['tagged']
            non_idxs_all = engram_dict[sess]['non-tagged']
            #drop indices that were manually scored to exclude 
            drop_idxs = inds[sess].loc[inds['Score']==3].values
            engram_idxs = [e for e in engram_idxs_all if e not in drop_idxs]
            non_idxs = [n for n in non_idxs_all if n not in drop_idxs]

            # Determine stimulus mode & get timestamps & stimulus times if needed.
            stimulus = ('TFC' in self.group) and (sess != 'Baseline')
            # Get timestamps
            timestamps = get_timestamps(self.animal, self.fov, sess, self.file_key, self.base_dir, stimulus=stimulus)
            if stimulus & (sess=='Recall1'):
                # If stimulus=True, timestamps contains (frame_times, on_times, off_times)
                frame_times, on_times, off_times = timestamps[0], timestamps[1], timestamps[2]
                # Save tone times
                tonetimes_path = os.path.join(savepath, f"{self.animal}_{self.fov}_{sess}_tonetimes.npz")
                np.savez(tonetimes_path, on_times=on_times/1000, off_times=off_times/1000)
                self.paths['tones'] = tonetimes_path
            
            elif stimulus & (sess=='Recall2'):
                # If stimulus=True, timestamps contains (frame_times, on_times, off_times)
                frame_times, on_times_1, off_times_1, on_times_2, off_times_2 = timestamps[0], timestamps[1], timestamps[2],timestamps[3],timestamps[4]
                # Save tone times
                tonetimes_path = os.path.join(savepath, f"{self.animal}_{self.fov}_{sess}_tonetimes.npz")
                np.savez(tonetimes_path, on_times_1=on_times_1/1000, off_times_1=off_times_1/1000,
                         on_times_2=on_times_2/1000, off_times_2=off_times_2/1000)
                self.paths['tones'] = tonetimes_path
            else:
                # If stimulus=False, timestamps is just frame_times
                frame_times = timestamps
                
            #resort the traces as engram first, then non-engram    
            traces = pd.concat([pd.DataFrame(act[engram_idxs].T,index=(frame_times/1000).tolist()),
                                pd.DataFrame(act[non_idxs].T,index=(frame_times/1000).tolist())],axis=1) 
            traces.columns = [(True, int(ei)) for ei in engram_idxs] + [(False, int(ni)) for ni in non_idxs]#column is binary
            traces=traces.reset_index(drop=False).rename(columns={'index':'timestamps'})
            fname= os.path.join(savepath,f"{self.animal}_{self.fov}_{sess}_{signal}.csv")
            csvs.append(fname)
            traces.to_csv(fname)
            print('saved: '+fname)
            
        if signal=='dff':
            self.dff_csvs = csvs
        elif signal=='events':
            self.event_csvs=csvs
        return csvs
    
    def load_traces(self, sessions=None, signal='dff', path=None, split=False,zscore=False):
        """
        Load per-session trace CSVs, computing them first if not already saved.

        Parameters
        ----------
        sessions : list of str, optional
            Sessions to load; if None, uses `self.sessions` (default
            None).
        signal : {'dff', 'events', 'model'}, optional
            Signal type to load (default 'dff').
        path : str, optional
            Directory containing the trace CSVs; if None, uses
            `self.paths['Traces']` (default None).
        split : bool, optional
            If True, split each session's traces into separate engram
            (tagged) and non-engram (non-tagged) dataframes based on
            column labels (default False).
        zscore : bool, optional
            If True, Z-score each cell's trace (per column, excluding
            the timestamp column) before returning (default False).

        Returns
        -------
        traces : pandas.DataFrame or dict of pandas.DataFrame
            If `split` is False: a single dataframe when one session is
            requested, or a dict mapping session name to dataframe when
            multiple sessions are requested.
        df_engram, df_non, timestamps : pandas.DataFrame, pandas.DataFrame, pandas.Series
            Returned instead of `traces` when `split` is True and a
            single session is requested.
        split_traces, timestamps : dict, pandas.Series
            Returned instead of `traces` when `split` is True and
            multiple sessions are requested; `split_traces` maps session
            name to `(df_engram, df_non)`.

        Notes
        -----
        If the trace CSVs don't already exist on disk, they are computed
        and saved via `save_traces_csv` before being loaded.
        """
        if sessions is None:
            sessions = self.sessions
        if path is None:
            path = self.paths['Traces']
        
        try:
            if len(sessions) == 1:
                traces = pd.read_csv(f"{path}/{self.animal}_{self.fov}_{sessions[0]}_{signal}.csv", index_col=0)
            else:
                self.traces = {sessions[i]: pd.read_csv(f"{path}/{self.animal}_{self.fov}_{sessions[i]}_{signal}.csv", index_col=0) for i in range(len(sessions))}
                traces = self.traces
        except:
            csv_paths = self.save_traces_csv(signal=signal, sessions=sessions)
            self.traces = {sessions[i]: pd.read_csv(f"{path}/{self.animal}_{self.fov}_{sessions[i]}_{signal}.csv", index_col=0) for i in range(len(sessions))}
            if len(sessions)==1:
                traces = self.traces[sessions[0]]
            elif sessions is None:
                traces=self.traces
            else:
                traces = {key: self.traces[key] for key in sessions if key in self.traces}
        
        if zscore:
            def zscore_df(df):
                df_z = df.copy()
                timestamp_col = df_z.columns[0]
                for col in df_z.columns[1:]:
                    df_z[col] = (df_z[col] - df_z[col].mean()) / df_z[col].std()
                return df_z

            if isinstance(traces, dict):
                traces = {k: zscore_df(v) for k, v in traces.items()}
            else:
                traces = zscore_df(traces)
            
        if split:
            if isinstance(traces, dict):  # If traces is a dictionary (multiple sessions)
                split_traces = {}
                for session, trace_df in traces.items():
                    # Split each session's trace
                    trace_df.columns = [ast.literal_eval(col) if isinstance(col, str) and col.startswith('(') else col for col in trace_df.columns]
                    engram_cols = [col for col in trace_df.columns if col[0] == True]
                    non_cols = [col for col in trace_df.columns if col[0] == False]
                    df_engram = trace_df[engram_cols]
                    df_non = trace_df[non_cols]
                    timestamps = trace_df[trace_df.columns[0]]
                    split_traces[session] = (df_engram, df_non)
                return split_traces, timestamps
            else:  # If traces is a single session dataframe
                traces.columns = [ast.literal_eval(col) if isinstance(col, str) and col.startswith('(') else col for col in traces.columns]
                engram_cols = [col for col in traces.columns if col[0] == True]
                non_cols = [col for col in traces.columns if col[0] == False]
                df_engram = traces[engram_cols]
                df_non = traces[non_cols]
                timestamps = traces[traces.columns[0]]
                return df_engram, df_non,timestamps
        return traces    
    
    def get_engram_bool(self, session='Recall1', signal='dff'):
        """
        Return a boolean engram/non-engram label for each cell in a session's traces.

        Assumes traces are ordered with engram cells first, followed by
        non-engram cells (as produced by `save_traces_csv`).

        Parameters
        ----------
        session : str, optional
            Session to extract engram labels from (default 'Recall1').
        signal : str, optional
            Signal type used in the saved trace file (default 'dff').

        Returns
        -------
        is_engram : list of bool
            True for engram (tagged) cells, False for non-engram cells,
            in column order.
        """
        traces_df = self.load_traces(sessions=[session], signal=signal)
        colnames = traces_df.columns[1:]  # Skip timestamp
        is_engram = [ast.literal_eval(col)[0] if isinstance(col, str) else col[0] for col in colnames]
        return is_engram
    
    def load_engram_idx(self,sessions=None,filter=True):
        """
        Load original CNMF/CellReg ROI indices for tagged and non-tagged cells, per session.

        Parameters
        ----------
        sessions : str or list of str, optional
            Session(s) to get indices for; if None, uses `self.sessions`
            (default None).
        filter : bool, optional
            Currently unused (default True).

        Returns
        -------
        engram_dict : dict
            Dictionary keyed by session name, each value a dict with keys
            'tagged' and 'non-tagged' mapping to arrays of original ROI
            indices (from CNMF/CellReg) for that session, excluding cells
            absent in that session (registration index == -1).
        """
        ind_df = self.load_cellreg()
        if type(sessions)==str:
            sessions=[sessions] 
        if sessions is not None:
            print('getting mcherry+ cells for session ' + str(sessions))
        else:
            sessions=self.sessions
        engram_dict={}
        for n,sess in enumerate(sessions):
            engram_dict[sess]={}
            # get engram indices from that session
            engram_dict[sess]['tagged']=ind_df[sess].loc[(ind_df[sess]!=-1)&(ind_df['Tagged']==1)].values 
            # get non-engram indices from that session
            engram_dict[sess]['non-tagged']=ind_df[sess].loc[(ind_df[sess]!=-1)&(ind_df['Tagged']==0)].values     
        return engram_dict
    
    ### cellreg functions ##### 
    def load_cellreg(self,sessions=None,drop_bad_cells=True):
        """
        Load the cell registration / tagging index table for this animal and FOV.

        Tries a series of candidate CSV filenames (index-split, sorted/
        annotated, or raw CellReg output) in order, using the first that
        exists.

        Parameters
        ----------
        sessions : list of str, optional
            Currently unused (default None).
        drop_bad_cells : bool, optional
            If True, drop rows where 'Score' == 3 (default True).

        Returns
        -------
        ind_df : pandas.DataFrame
            Cell registration/index dataframe, optionally with low-quality
            (Score == 3) cells removed.

        Raises
        ------
        FileNotFoundError
            If none of the candidate index CSV files can be loaded.
        """
        file_candidates = [f"{self.paths['Tagging']}/{self.animal}_{self.fov}_indices_split.csv",
                           f"{self.paths['Tagging']}/{self.animal}_{self.fov}_reg_indices_sort_annotate.csv",
                            f"{self.paths['Tagging']}/CellReg_output/{self.animal}_{self.fov}_reg_indices.csv"]

        ind_df = None
        for filename in file_candidates:
            try:
                ind_df = pd.read_csv(os.path.join(self.paths['Tagging'], filename), index_col=0)
                print(os.path.join(self.paths['Tagging'], filename))
                break  # Successfully loaded, exit the loop
            except FileNotFoundError:
                continue  # Try next file

        if ind_df is None:
            raise FileNotFoundError("None of the candidate cell index csv files could be loaded.")
        if drop_bad_cells:
            ind_df = ind_df.loc[ind_df['Score']!=3]
        return ind_df

    #### Tone functions #####
    
    # get timestamps for frames, and tone_times if needed
    def load_tone_times(self,session_name='Recall1'):
        """
        Load tone (CS) onset/offset times for a session.

        Parameters
        ----------
        session_name : {'Recall1', 'Recall2'}, optional
            Session to load tone times for (default 'Recall1').

        Returns
        -------
        on_times, off_times : ndarray
            CS1 onset and offset times, if `session_name` is 'Recall1'.
        on_times_1, off_times_1, on_times_2, off_times_2 : ndarray
            CS1 and CS2 onset/offset times, if `session_name` is 'Recall2'.

        Notes
        -----
        Sets `self.tone_data[session_name]` to the loaded `.npz` archive.
        """
        npz_file=f"{self.paths['Traces']}/{self.animal}_{self.fov}_{session_name}_tonetimes.npz"
        tone_data = np.load(npz_file)
        self.tone_data[session_name]=tone_data

        if session_name=='Recall1':
            on_times = tone_data['on_times']
            off_times = tone_data['off_times']
            return on_times,off_times
        elif session_name=='Recall2':
            on_times_1 = tone_data['on_times_1']
            on_times_2 = tone_data['on_times_2']
            off_times_1 = tone_data['off_times_1']
            off_times_2 = tone_data['off_times_2']
            return on_times_1,off_times_1,on_times_2,off_times_2
    
    def make_trial_table(self,session='Recall2'):
        """
        Build a table of trial start/stop times, ordered by onset, for a session.

        Parameters
        ----------
        session : {'Recall1', 'Recall2'}, optional
            Session to build the trial table for (default 'Recall2').

        Returns
        -------
        pandas.DataFrame
            Columns 'trial' (global trial number in temporal order), 'type'
            ('cs1' or 'cs2'), 'type_trial' (trial number within that type),
            'start', and 'stop'.

        Notes
        -----
        Sets `self.trials_table` to the full trial dataframe (same columns
        as the returned dataframe).
        """
        if session == 'Recall1':
            cs1_starts, cs1_stops = self.load_tone_times(session_name=session)
            cs2_starts,cs2_stops = None,None
        elif session == 'Recall2':
            cs1_starts, cs1_stops, cs2_starts,cs2_stops  = self.load_tone_times(session_name=session)
        
        cs1 = pd.DataFrame({
            "type": "cs1",
            "type_trial": np.arange(len(cs1_starts), dtype=int),
            "start": cs1_starts,
            "stop": cs1_stops})
        
        if cs2_starts is not None:
            cs2 = pd.DataFrame({
                "type": "cs2",
                "type_trial": np.arange(len(cs2_starts), dtype=int),
                "start": cs2_starts,
                "stop": cs2_stops,
            })

            trials = pd.concat([cs1, cs2], ignore_index=True).sort_values("start").reset_index(drop=True)
            trials["trial"] = np.arange(len(trials), dtype=int)  # global trial number in sorted order
        else:
            trials = cs1.copy()
            trials["trial"] = np.arange(len(trials), dtype=int)  # global trial number in sorted order

        self.trials_table = trials
        return trials[["trial", "type", "type_trial", "start", "stop"]]
    
    def extract_trial_tensor(self, session='Recall1', signal='dff', window=(-2, 5),normalize =True,bin=False,bin_size=30):
        """
        Extract a trials x time x cells tensor of activity around tone onsets.

        Parameters
        ----------
        session : {'Recall1', 'Recall2'}, optional
            Session to extract from (default 'Recall1').
        signal : {'dff', 'events', 'model'}, optional
            Signal type to extract (default 'dff').
        window : tuple of float, optional
            (start_time, end_time) in seconds relative to each stimulus onset,
            defining the extraction window (default (-2, 5)).
        normalize : bool, optional
            If True, divide each cell's trace by its standard deviation before
            extracting trials (default True).
        bin : bool, optional
            Not implemented; currently has no effect (default False).
        bin_size : int, optional
            Not implemented; currently has no effect (default 30).

        Returns
        -------
        trial_tensor : ndarray, shape (n_trials, n_timepoints, n_cells)
            Trial-aligned activity tensor, if `session` is 'Recall1'.
        trial_tensor_1, trial_tensor_2 : ndarray, shape (n_trials, n_timepoints, n_cells)
            Trial-aligned activity tensors for CS1 and CS2, if `session` is
            'Recall2'.

        Notes
        -----
        Trials whose extraction window falls partially outside the available
        trace data are dropped, so `n_trials` may be less than the total
        number of tone presentations. Returns None for any other `session`
        value.
        """
        # Load data using built-in methods
        traces_df = self.load_traces(sessions=[session], signal=signal)
        timestamps = traces_df.iloc[:, 0].values  # First column is timestamps
        traces = traces_df.iloc[:, 1:].values    # Remaining columns are traces

        if normalize:
            traces = traces / traces.std(axis=0, keepdims=True)
        # if bin: not implemented 
        #     binned = bin_traces(traces.T,bin_size=bin_size)
        #     traces = binned.T
        sampling_rate = 1 / np.median(np.diff(timestamps))
        window_frames = (int(window[0] * sampling_rate), int(window[1] * sampling_rate))
        win_length = window_frames[1] - window_frames[0]
        n_cells = traces.shape[1]
        if session=='Recall1':
            on_times, _ = self.load_tone_times(session_name=session)
            
            trial_traces = []
            for onset in on_times:
                center_idx = np.searchsorted(timestamps, onset)
                start_idx = center_idx + window_frames[0]
                end_idx = center_idx + window_frames[1]

                snippet = traces[start_idx:end_idx, :]
                if snippet.shape[0] == win_length:
                    trial_traces.append(snippet)

            trial_tensor = np.stack(trial_traces, axis=0)
            return trial_tensor
    
        elif session=='Recall2':
            on_times_1, _,on_times_2,_ = self.load_tone_times(session_name=session)
            trial_traces_1 = []
            for onset in on_times_1:
                center_idx = np.searchsorted(timestamps, onset)
                start_idx = center_idx + window_frames[0]
                end_idx = center_idx + window_frames[1]

                snippet = traces[start_idx:end_idx, :]
                if snippet.shape[0] == win_length:
                    trial_traces_1.append(snippet)
            trial_tensor_1 = np.stack(trial_traces_1, axis=0)

            trial_traces_2=[]
            for onset in on_times_2:
                center_idx = np.searchsorted(timestamps, onset)
                start_idx = center_idx + window_frames[0]
                end_idx = center_idx + window_frames[1]

                snippet = traces[start_idx:end_idx, :]
                if snippet.shape[0] == win_length:
                    trial_traces_2.append(snippet)
            trial_tensor_2 = np.stack(trial_traces_2, axis=0)
            return trial_tensor_1,trial_tensor_2

        
    ###### Running functions ######
    def compute_velocity(self, sessions=None, window_size=30, threshold=1):
        """
        Compute velocity for one or more sessions from position data.

        Parameters
        ----------
        sessions : list of str, optional
            Sessions to compute velocity for; if None, uses `self.sessions`
            (default None).
        window_size : int, optional
            Window size, in frames, used for the velocity calculation (default
            30).
        threshold : float, optional
            Velocity threshold below which values are zeroed out as noise
            (default 1).

        Returns
        -------
        velocity : ndarray or dict of ndarray
            Velocity trace for the session, if a single session is requested;
            otherwise a dict mapping session name to velocity trace.

        Notes
        -----
        Sets `self.velocity` to the returned value.
        """
        if sessions is None:
            sessions = self.sessions

        velocity_dict = {}
        for sess in sessions:
            df_pos = self.load_position_df(sess)

            vt0 = calc_velocity(
                df_pos['position'].values,
                df_pos['frame_times'].values,
                frame_period=np.diff(df_pos['frame_times'].values).mean(),
                window_size=window_size
            )
            vt = thr_velocity(vt0, thr=threshold)
            velocity_dict[sess] = vt

        self.velocity = velocity_dict if len(sessions) > 1 else list(velocity_dict.values())[0]
        return self.velocity

    def load_position_df(self, session):
        """
        Load wheel position data for a session.

        Parameters
        ----------
        session : str
            Session name (e.g. 'Recall1').

        Returns
        -------
        pandas.DataFrame
            Wheel position dataframe with 'frame_times' and 'position' columns
            (plus any other columns from the wheel data CSV), with the
            'Unnamed: 0' index column dropped.
        """
        info = self.exp_data
        TSeries = info['TSeries_g'].loc[(info['Animal']==self.animal) &
                                        (info['FOV']==self.fov) &
                                        (info['Session']==session)].values[0]
        path = os.path.join(self.base_dir,'Behavior','Wheel_data',f'{TSeries}_wheeldata.csv')
        return pd.read_csv(path).drop(['Unnamed: 0'], axis=1)

    
    ##### Spatial ROI functions ##### 
    def load_cnmf_components(self,session=None):
        """
        Load CNMF output components (spatial/temporal footprints, SNR, etc.) for a session.

        Parameters
        ----------
        session : str, optional
            Session to load CNMF components for (default None).

        Returns
        -------
        cnmf_data : dict
            Dictionary with keys 'A' (spatial footprints, sparse matrix
            components), 'C' (temporal components), 'F_dff' (dF/F traces),
            'b' (spatial background), 'f' (temporal background), 'Cn'
            (correlation image), 'SNR_comp' (per-component SNR), and 'dims'
            (tuple, movie frame dimensions).
        """
        TSeries = self.exp_data['TSeries_g'].loc[(self.exp_data['Animal']==self.animal)&(self.exp_data['FOV']==self.fov)&(self.exp_data['Session']==session)].values[0]
        TSeries_g = [item for item in self.TSeries_names if TSeries in item][0]
        cnmf_path = os.path.join(self.paths['GCaMP'],'caiman_output','cnmf')
        hdf = [os.path.join(cnmf_path,h) for h in os.listdir(cnmf_path) if (TSeries_g in h) &(h.endswith('.hdf5'))][0]
        members_to_load = ['A', 'C', 'F_dff', 'b', 'f', 'Cn', 'SNR_comp',]
        # Open the HDF5 file 
        with h5py.File(hdf, 'r') as file:
            estimates = file['/estimates']
            cnmf_data={}
            for member in members_to_load:
                item = estimates[member]
                # Check if the item is a dataset
                if isinstance(item, h5py.Dataset):
                    cnmf_data[member] = item[()]  # load dataset into memory
                elif isinstance(item, h5py.Group):
                    # If the member is a group, you need to decide how to handle it.For example, load each dataset within the group.
                    group_data = {}
                    for key, subitem in item.items():
                        if isinstance(subitem, h5py.Dataset):
                            group_data[key] = subitem[()]  # load each dataset
                    cnmf_data[member] = group_data
            cnmf_data['dims']=tuple(file['dims'][()])            
        return cnmf_data
    
    def load_footprints(self,session='Recall1',roi_style='patch'):
        """
        Load and sort cell spatial footprints (patches or contours) by engram status.

        Parameters
        ----------
        session : str, optional
            Session to load footprints for (default 'Recall1').
        roi_style : {'patch', 'contours'}, optional
            Footprint representation to return: dense image patches or
            contour coordinates (default 'patch').

        Returns
        -------
        A_sorted : ndarray or list
            Footprints sorted with engram (tagged) cells first, followed by
            non-engram cells. An ndarray of image patches if `roi_style` is
            'patch', or a list of contour coordinate arrays if 'contours'.
        is_engram : list of bool
            True for engram cells, False for non-engram cells, aligned with
            `A_sorted`.
        """
        from spatial.get_footprints_CellReg import reshape_ROIs
        from plotting import get_contours
        
        cnmf_data = self.load_cnmf_components(session=session)
        A_dict = cnmf_data['A']
        A = sp.csc_matrix((A_dict['data'], A_dict['indices'], A_dict['indptr']), shape=A_dict['shape'])
        
        engram_dict = self.load_engram_idx() 
        engram_idxs = engram_dict[session]['tagged']
        non_idxs = engram_dict[session]['non-tagged']
        
        if roi_style=='patch':
            A_3d = reshape_ROIs(A, self.base_dir,self.animal,self.fov,session=session,save_mat=False,dims = cnmf_data['dims'])
            A_engram = A_3d[engram_idxs,:,:]
            A_nonengram = A_3d[non_idxs,:,:]
            A_sorted = np.concatenate((A_engram,A_nonengram),axis=0)
        elif roi_style=='contours':
            A_coord = get_contours(A,dims=cnmf_data['dims'])
            A_engram= [A_coord[d]['coordinates'] for d in engram_idxs]
            A_nonengram = [A_coord[d]['coordinates'] for d in non_idxs]
            A_sorted = A_engram + A_nonengram
        
        is_engram = [True for ei in engram_idxs]+[False for ni in non_idxs]
        return A_sorted,is_engram
    
  

