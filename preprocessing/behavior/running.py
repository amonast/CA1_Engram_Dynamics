import pandas as pd
import matplotlib.pyplot as plt
import os
import datetime as dt
from scipy.interpolate import interp1d
import numpy as np

def get_rest_epochs(timestamps, velocity_trace, min_duration=1.0):
    """
    Identify rest epochs (zero velocity) from a velocity trace using timestamps.
    Parameters
    ----------
    timestamps : array-like
        1D array of timestamps, same length as `velocity_trace`.
    velocity_trace : array-like
        1D array of velocity values over time.
    min_duration : float, optional
        Minimum duration (in the same units as `timestamps`) for a valid rest
        bout (default 1.0).
    Returns
    -------
    list of tuple of int
        List of (start_idx, stop_idx) tuples giving the index bounds of each
        rest epoch into `timestamps`/`velocity_trace`.
    Raises
    ------
    ValueError
        If `timestamps` and `velocity_trace` do not have the same shape.
    """
    timestamps = np.asarray(timestamps)
    velocity_trace = np.asarray(velocity_trace)

    if timestamps.shape != velocity_trace.shape:
        raise ValueError("timestamps and velocity_trace must be the same shape")

    is_rest = velocity_trace == 0 
    changes = np.diff(is_rest.astype(int))

    start_times = timestamps[np.where(changes == 1)[0] + 1]
    stop_times = timestamps[np.where(changes == -1)[0] + 1]

    # Handle edge cases
    if is_rest[0]:
        start_times = np.insert(start_times, 0, timestamps[0])
    if is_rest[-1]:
        stop_times = np.append(stop_times, timestamps[-1])

    # Filter by minimum duration
    valid_epochs = [(start, stop) for start, stop in zip(start_times, stop_times)
                    if (stop - start) >= min_duration]

    # Convert back to indices
    index_epochs = [(np.searchsorted(timestamps, start), np.searchsorted(timestamps, stop))
                    for start, stop in valid_epochs]

    return index_epochs

def get_running_epochs(timestamps, velocity_trace, threshold=2.0, min_duration=1.0,return_time=False):
    """
    Identify running epochs (velocity above threshold) from a velocity trace.
    Parameters
    ----------
    timestamps : array-like
        1D array of timestamps, same length as `velocity_trace`.
    velocity_trace : array-like
        1D array of velocity values over time.
    threshold : float, optional
        Velocity threshold above which the animal is considered running
        (default 2.0).
    min_duration : float, optional
        Minimum duration (in the same units as `timestamps`) for a valid
        running bout (default 1.0).
    return_time : bool, optional
        If True, return epochs as (start_time, stop_time) tuples instead of
        indices (default False).
    Returns
    -------
    list of tuple
        List of (start, stop) tuples representing running epochs, either as
        indices into `timestamps`/`velocity_trace` (default) or as timestamp
        values (if `return_time` is True).
    Raises
    ------
    ValueError
        If `timestamps` and `velocity_trace` do not have the same shape.
    """

    timestamps = np.asarray(timestamps)
    velocity_trace = np.asarray(velocity_trace)

    if timestamps.shape != velocity_trace.shape:
        raise ValueError("timestamps and velocity_trace must be the same shape")

    is_running = velocity_trace > threshold
    changes = np.diff(is_running.astype(int))

    start_times = timestamps[np.where(changes == 1)[0] + 1]
    stop_times = timestamps[np.where(changes == -1)[0] + 1]

    # Handle edge cases
    if is_running[0]:
        start_times = np.insert(start_times, 0, timestamps[0])
    if is_running[-1]:
        stop_times = np.append(stop_times, timestamps[-1])
    
    # Filter by minimum duration
    if return_time:
        valid_epochs = [(start, stop) for start, stop in zip(start_times, stop_times)
                    if (stop - start) >= min_duration]
        return valid_epochs

    else: # Convert back to indices
        index_epochs = [(np.searchsorted(timestamps, start), np.searchsorted(timestamps, stop))
                    for start, stop in valid_epochs]
        return index_epochs

def percent_run_all(animals,fov_lists,sessions,file_key,base_dir,tinds=None):
    """
    Build a dataframe of percent time spent running/resting across sessions.

    Parameters
    ----------
    animals : list of str
        List of animal identifiers.
    fov_lists : list of list of str
        List of FOV identifiers per animal; `fov_lists[i]` gives the FOVs for
        `animals[i]`.
    sessions : list of str
        List of session identifiers.
    file_key : str
        Path to the metadata CSV for the experiment.
    base_dir : str
        Base directory for processed data.
    tinds : tuple of int, optional
        Start/stop frame indices restricting the window used to compute
        percent time (default None, meaning the full trace is used).

    Returns
    -------
    pandas.DataFrame
        Concatenated dataframe with one row per animal/FOV/session containing
        percent time running and resting.
    """
    df = pd.DataFrame()
    for a,ani in enumerate(animals):
        for fov in fov_lists[a]:
            for session in sessions:
                df_ani = run_df_ani(ani,fov,session,file_key,base_dir,tinds=tinds)
                df = pd.concat([df,df_ani],ignore_index=True)
    return df

def run_df_ani(ani,fov,session,file_key,base_dir,tinds=None,window_size=30):
    """
    Compute percent time running/resting for a single animal, FOV, and session.

    Parameters
    ----------
    ani : str
        Animal identifier.
    fov : str
        FOV identifier.
    session : str
        Session identifier.
    file_key : str
        Path to the metadata CSV for the experiment.
    base_dir : str
        Base directory for processed data.
    tinds : tuple of int, optional
        Start/stop frame indices restricting the window used to compute
        percent time (default None, meaning the full trace is used).
    window_size : int, optional
        Number of frames over which velocity is calculated (default 30).

    Returns
    -------
    pandas.DataFrame
        Single-row dataframe with columns 'Animal', 'FOV', 'Session', 'Group',
        '% Run', and '% Rest'.
    """
    info = pd.read_csv(file_key)
    df=load_position(ani,fov,session,file_key,base_dir)
    vt0 = calc_velocity(df.position.values,df.frame_times.values,frame_period=df.frame_times.values[1]-df.frame_times.values[0],window_size=window_size)
    
    vt = thr_velocity(vt0)
    rest, run = get_rest_array(vt)
   
    df = pd.DataFrame()
    df['Animal']=[ani]
    df['FOV']=[fov]
    df['Session']=[session]
    df['Group'] = [info['Group'].loc[info['Animal']==ani].values[0]]
    df['% Run'] =  percent_time(rest,state='run',tinds=tinds)
    df['% Rest'] = percent_time(rest,state='rest',tinds=tinds)

    return df

def load_position(ani,fov,session,file_key,base_dir):
    """
    Load wheel position data for a given animal, FOV, and session.
    Parameters
    ----------
    ani : str
        Animal identifier.
    fov : str
        FOV identifier.
    session : str
        Session identifier.
    file_key : str
        Path to the metadata CSV for the experiment.
    base_dir : str
        Base directory containing the 'Behavior/Wheel_csvs' folder.
    Returns
    -------
    pandas.DataFrame
        Wheel position dataframe for the matching TSeries
    """
    info = pd.read_csv(file_key)
    TSeries = info['TSeries_g'].loc[(info['Animal']==ani)&(info['FOV']==fov)&(info['Session']==session)].values[0]
    path=os.path.join(base_dir,'Behavior','Wheel_csvs',TSeries+'_wheeldata.csv')
    return pd.read_csv(path).drop(['Unnamed: 0'],axis=1)

def thr_velocity(vt,thr=1.5):
    """
    Zero out low-magnitude velocity values below a minimum threshold.
    Parameters
    ----------
    vt : ndarray, shape (T,)
        Velocity trace.
    thr : float, optional
        Minimum velocity magnitude (cm/s) to be considered true running;
        values with smaller magnitude are set to zero. Default 1.5,
        chosen by eye.
    Returns
    -------
    ndarray, shape (T,)
        Velocity trace with sub-threshold values zeroed out.
    """
    vt[(vt>0)&(vt<thr)]=0
    vt[(vt<0)&(vt>-thr)]=0
    return vt

def get_rest_array(vt, bout=1, fr=30, threshold=1.0):
    """
    Compute binary rest and running arrays from a velocity trace.
    Parameters
    ----------
    vt : array-like, shape (T,)
        Velocity trace.
    bout : float, optional
        Time window before/after a rest epoch begins, in seconds. If 0,
        rest/running is determined by simple thresholding rather than
        windowed padding (default 1).
    fr : float, optional
        Frame rate (default 30).
    threshold : float, optional
        Velocity threshold used to define running when `bout == 0` (default 1.0).
    Returns
    -------
    rest : ndarray of int, shape (T,)
        Binary array; 1 indicates rest, 0 indicates running.
    run : ndarray of int, shape (T,)
        Binary array; 1 indicates running, 0 indicates rest.
    """
    vt = np.asarray(vt)
    run = np.zeros(len(vt))
    
    if bout == 0:
        run[np.where(np.abs(vt) > threshold)] = 1
        rest = 1 - run  # inverse of run
    else:
        run[np.where(np.abs(vt) > 0)] = 1
        fr_bout = int(bout * fr)
        rest_pad = np.zeros(len(vt) + fr_bout)
        for i in range(len(vt)):
            if (run[i - fr_bout:i + fr_bout] == 0).all():
                rest_pad[i + fr_bout] = 1
        rest = rest_pad[fr_bout:]
    
    return rest.astype(int), run.astype(int)
    
def percent_time(rest,state='rest',tinds=None):
    """
    Compute the percent of time spent in a given behavioral state.
    Parameters
    ----------
    rest : array-like
        Binary rest epoch array (1 = rest, 0 = running).
    state : {'rest', 'run'}, optional
        Behavioral state to compute percent time for (default 'rest').
    tinds : tuple of int, optional
        Start/stop frame indices restricting the window (default None,
        meaning the full array is used).
    Returns
    -------
    float
        Percent of time spent in `state` within the specified window.
    """
    if tinds is None:
        ti,tii = 0,len(rest)
    else:
        ti,tii = tinds[0],tinds[1]
    
    if state == 'rest':
        arr = rest[ti:tii]
    elif state == 'run':
        arr = 1-rest[ti:tii]
    percent = arr.sum()/len(arr) *100

    return percent

def calc_velocity(x, times, frame_period, window_size):
    """
    Compute velocity from position data using a sliding window.
    Parameters
    ----------
    x : array-like, shape (T,)
        Position data.
    times : array-like, shape (T,)
        Frame times, in seconds.
    frame_period : float
        Frame period from the TSeries XML metadata, in seconds.
    window_size : int
        Number of frames over which to calculate velocity; should be odd for
        symmetry.
    Returns
    -------
    ndarray, shape (T,)
        Velocity at each frame.
    """

    window_size = int(window_size)
    pad_size = window_size - 1

    # Pad time and position
    pre_app = np.linspace(times[0] - frame_period * pad_size,
                          times[0] - frame_period,
                          pad_size)
    post_app = np.linspace(times[-1] + frame_period,
                           times[-1] + frame_period * pad_size,
                           pad_size)
    t_padded = np.concatenate((pre_app, times, post_app))
    x_padded = np.pad(x, (pad_size, pad_size), mode='edge')

    dX = []
    dT = []

    for t in range(len(x)):
        ti = t_padded[t]
        tii = t_padded[t + window_size - 1]
        xi = x_padded[t]
        xii = x_padded[t + window_size - 1]
        dT.append(tii - ti)
        dX.append(xii - xi)

    dx = np.array(dX, dtype='float')
    dt = np.array(dT, dtype='float')
    return dx / dt

def get_frametimes(ani,fov,session,metadata_path,file_key):
    """
    Get frame times and frame period for a given animal, FOV, and session.
    Parameters
    ----------
    ani : str
        Animal identifier.
    fov : str
        FOV identifier.
    session : str
        Session identifier.
    metadata_path : str
        Folder containing per-animal subfolders with
        '{TSeries}_timestamps.npz' and '{TSeries}_metadata_settings.csv'.
    file_key : str
        Path to the metadata CSV for the experiment.
    Returns
    -------
    frame_times : ndarray
        Frame times relative to the first frame (t=0), in milliseconds.
    frame_period : float
        Frame period, in seconds.
    """
    info = pd.read_csv(file_key)
    path = os.path.join(metadata_path,ani)
    metadata = get_xml_metadata(ani,fov,session,file_key,path)
    TSeries = info['TSeries_g'].loc[(info['Animal']==ani)&(info['FOV']==fov)&(info['Session']==session)].values[0]
    timestamps = np.load(f"{path}/{TSeries}_timestamps.npz")

    frame_period = metadata.frame_period
    frame_times = timestamps['frame_times']
    return frame_times,frame_period

def get_positions(ani,fov,session,frame_times,wheel_path,metadata_path,file_key):
    """
    Interpolate wheel position at each imaging frame time.

    Parameters
    ----------
    ani : str
        Animal identifier.
    fov : str
        FOV identifier.
    session : str
        Session identifier, e.g. 'Baseline' or 'Post'.
    frame_times : array-like
        Imaging frame times, in milliseconds (e.g. 0, 33., 66., ...).
    wheel_path : str
        Path to the folder containing all wheel CSVs.
    metadata_path : str
        Path to the folder containing TSeries XML metadata exports.
    file_key : str
        Path to the metadata CSV for the experiment.
    Returns
    -------
    xnew : list
        Wheel position at each of the T requested frame times.
    """
    #get the wheel dataframe
    wheeldf = get_wheel_df(ani,fov,session,file_key,wheel_path)
    t0 = get_t0(ani,fov,session,file_key,metadata_path)
    #figure out times 
    dts=[] #to store the datetime objects
    deltas=[]#to store the time difference relative to first frame time 

    for i in range(len(wheeldf)):
        y = int(wheeldf[0].iloc[i][0:4])
        m = int(wheeldf[0].iloc[i][5:7])
        d = int(wheeldf[0].iloc[i][8:10])

        tprime=wheeldf[0].iloc[i][11:]
        dattime=dt.datetime(year=y,month=m,day=d,hour=int(tprime.split(':')[0]),minute=int(tprime.split(':')[1]),
                            second=int(tprime.split(':')[-1].split('.')[0]),microsecond=int(tprime.split(':')[-1].split('.')[-1])*1000)
        dts.append(dattime)
        deltas.append((dattime-t0).total_seconds()*1000)

    #make new dataframe for recorded arduino times 
    df_short = pd.DataFrame()
    df_short['datetime']=dts #datetime objects
    df_short['time'] = deltas #relative to the first frame time
    x0= extract_position_short(wheeldf)
    df_short['position'] =x0

    #make a long dataframe interpolated for the missing times, time relative to the first frame
    ms = np.arange(0,int(df_short['time'].iloc[-1])) #long array, starting from 0 first frame, 1ms steps 
    interp = interp1d(df_short['time'],x0,kind='previous') #interpolate function with previous value
    xlong = interp(ms).astype(int) #interpolated positions

    df_long = pd.DataFrame()
    df_long['image_time']=ms
    df_long['position'] = xlong 

    #get positions of only frame times
    xnew = [df_long.position.iloc[np.where(df_long.image_time==np.round(frame_times[i]))[0][0]] for i in range(len(frame_times))]
    
    return xnew

def extract_position_short(wheeldf):
    """
    Extract wheel position values from the raw wheel dataframe.
    Parameters
    ----------
    wheeldf : pandas.DataFrame
        Raw wheel dataframe as read from the wheel CSV.
    Returns
    -------
    list of int
        Position value extracted from each row.
    """    
    if wheeldf.shape[1]==4: 
            pos_col=3
    elif wheeldf.shape[1]==3:
        pos_col=2
        
    pos = [int(wheeldf[pos_col].iloc[i].split(':')[-1]) for i in range(len(wheeldf[2]))] #positions
    return pos

def get_t0(ani,fov,session,file_key,metadata_path):
    """
    Get the datetime of the first imaging frame for a session.

    Reads the TSeries XML metadata's acquisition start timestamp and adds the
    offset to the first frame to obtain the precise datetime of the first
    acquired image.

    Parameters
    ----------
    ani : str
        Animal identifier.
    fov : str
        FOV identifier.
    session : str
        Session identifier.
    file_key : str
        Path to the metadata CSV for the experiment.
    metadata_path : str
        Path to the folder containing per-animal TSeries XML metadata exports.

    Returns
    -------
    datetime.datetime
        Datetime of the first imaging frame, precise to microseconds.
    """
    info = pd.read_csv(file_key)
    TSeries = info['TSeries_g'].loc[(info['Animal']==ani)&(info['FOV']==fov)&(info['Session']==session)].values[0]
    metadata = get_xml_metadata(ani,fov,session,file_key,os.path.join(metadata_path,ani))

    t0_str=metadata.timestamp0.values[0] #string time of the TSeries start time in PC time
    offset = metadata.frame1_absTime.values[0] #delay between acquistion start + first frame 

    #make date time object for t0, being the date and time of the first frame, precise to microsec
    dstr=metadata.date[0][:10]
    m = int(dstr.split('/')[0])
    d = int(dstr.split('/')[1])
    y = int(dstr.split('/')[2])
    h = int(t0_str.split(':')[0])
    minute = int(t0_str.split(':')[1])
    s = int(t0_str.split(':')[2][:2])
    us =  int(t0_str.split(':')[-1].split('.')[-1][:6])

    time = dt.datetime(y,m,d,h,minute,s,us)
    t0 = time+dt.timedelta(seconds=offset)
    return t0

def get_wheel_df(ani,fov,session,file_key,wheel_path):
    """
    Load and clean the raw wheel encoder CSV for a session.
    Parameters
    ----------
    ani : str
        Animal identifier.
    fov : str
        FOV identifier.
    session : str
        Session identifier.
    file_key : str
        Path to the metadata CSV for the experiment.
    wheel_path : str
        Path to the folder containing wheel CSVs.
    Returns
    -------
    pandas.DataFrame
        Cleaned wheel dataframe, starting from the first row with index 0,
        with bad rows dropped and reset index.
    """
    exp_info = pd.read_csv(file_key)
    TSeries = exp_info['TSeries_g'].loc[(exp_info['Animal']==ani)&(exp_info['FOV']==fov)&(exp_info['Session']==session)].values[0]
    print(TSeries)
    #get wheel csv data from experiment
    try:
        csv =f"{wheel_path}/{TSeries}-wheel.csv"
        df = pd.read_csv(csv)
    except FileNotFoundError:
        csv =f"{wheel_path}/{TSeries}_wheel.csv"
        df = pd.read_csv(csv)

    #read in csv accounting for different artefacts
    try:
        wheeldf=pd.read_csv(csv,header=None,usecols = [0,1,2,3]).dropna().reset_index(drop=True)
    except:
        wheeldf=pd.read_csv(csv,header=None).dropna().reset_index(drop=True)
    
    #get shape of the dataframe 
    if wheeldf.shape[1]==4:
        col=3
    elif wheeldf.shape[1]==3:
        col=2

    #drop any nan rows from the csv
    bad=[]
    for i,row in wheeldf.iterrows():

        try:
            int(wheeldf[col].iloc[i].split(':')[-1])
            int(wheeldf[1].iloc[i])
        except:
            bad.append(i)
    wheeldf=wheeldf.drop(bad,axis=0)
    wheeldf[1] = wheeldf[1].astype(int)

    ii =wheeldf.loc[wheeldf[1]==0].index

    return wheeldf.iloc[ii[0]:].reset_index(drop=True)

def get_xml_metadata(ani,fov,session,file_key,metadata_folder):
    """
    Load exported TSeries XML metadata from CSV.
    Parameters
    ----------
    ani : str
        Animal identifier.
    fov : str
        FOV identifier.
    session : str
        Session identifier.
    file_key : str
        Path to the metadata CSV for the experiment.
    metadata_folder : str
        Path to the folder containing exported '{TSeries}_metadata_settings.csv'
        files.
    Returns
    -------
    pandas.DataFrame
        Metadata settings dataframe for the matching TSeries.
    """
    exp_info = pd.read_csv(file_key)
    TSeries = exp_info['TSeries_g'].loc[(exp_info['Animal']==ani)&(exp_info['FOV']==fov)&(exp_info['Session']==session)].values[0]
    print(TSeries)
    csvfile = os.path.join(metadata_folder,TSeries+'_metadata_settings.csv')
    metadata=pd.read_csv(csvfile)
    return metadata