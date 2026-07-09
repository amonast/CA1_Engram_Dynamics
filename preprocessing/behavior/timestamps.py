#%%
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
'''
Functions for generating, saving and retrieving timestamps for tone trials in retrieval session
'''

def get_tone_frame_vector(on_times,off_times,frame_times):
    """
    Build a binary tone on/off vector aligned to imaging frame times.

    Parameters
    ----------
    on_times : array-like
        Onset time for each tone presentation, across all trials.
    off_times : array-like
        Offset time for each tone presentation, across all trials.
    frame_times : array-like
        Imaging frame times, same units as `on_times`/`off_times`.

    Returns
    -------
    tone_vector : ndarray, shape (50000,)
        Binary vector, one entry per frame index (fixed length 50000), with
        1 during tone presentation and 0 otherwise.
    """
    tone_vector = np.zeros(50000)

    for trial in range(len(on_times)):
        time0= on_times[trial] 
        time1 = off_times[trial]
        times = frame_times[(time0<frame_times)&(time1>frame_times)]
        inds = np.argwhere(np.in1d(frame_times, times)).flatten()
        tone_vector[inds]=1

    return tone_vector

def get_trial_times(on_times,off_times,frame_times,trial,seconds_pre=1,seconds_post=25):
    """
    Get frame times and indices for a single trial, centered on tone onset.

    Parameters
    ----------
    on_times : array-like
        Onset time for each tone presentation, across all trials.
    off_times : array-like
        Offset time for each tone presentation, across all trials.
    frame_times : array-like
        Imaging frame times, same units as `on_times`/`off_times`.
    trial : int
        Trial number to extract (0-indexed).
    seconds_pre : float, optional
        Seconds before tone onset to include in the window (default 1).
    seconds_post : float, optional
        Seconds after tone onset to include in the window (default 25).

    Returns
    -------
    centered_times : ndarray
        Frame times within the trial window, re-centered so tone onset is 0.
    inds : ndarray of int
        Indices into `frame_times` (and any aligned trace array) for frames
        within the trial window.
    onset_offset : tuple of int
        (onset_c, offset_c), the fixed centered onset (0) and offset (20000)
        times, in the same units as `on_times`.
    """

    time0= on_times[trial] - seconds_pre*1000
    time1 = on_times[trial]+seconds_post*1000
    times = frame_times[(time0<frame_times)&(time1>frame_times)]
    inds = np.argwhere(np.in1d(frame_times, times)).flatten()

    centered_times = times - on_times[trial]
    onset_c = 0
    offset_c = 20000

    return centered_times,inds,(onset_c,offset_c)

def get_timestamps(ani,fov,session,file_key,base_dir,stimulus=False,recall2=False):
    """
    Load frame times, and tone onset/offset times if applicable, for a session.

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
        Base directory containing the 'xml_metadata/{ani}' folder with the
        saved timestamps `.npz` file.
    stimulus : bool, optional
        Whether this session includes tone stimuli (default False).
    recall2 : bool, optional
        Unused; session type is instead inferred from `session` being
        'Recall1' or 'Recall2' (default False).

    Returns
    -------
    frame_times : ndarray
        Imaging frame times, in milliseconds.
    on_times, off_times : ndarray
        CS1 tone onset/offset times, only returned if `stimulus` is True and
        `session` is 'Recall1'.
    on_times_1, off_times_1, on_times_2, off_times_2 : ndarray
        CS1 and CS2 tone onset/offset times, only returned if `stimulus` is
        True and `session` is 'Recall2'.
    """
    metadata =pd.read_csv(file_key)
    TSeries = metadata['TSeries_g'].loc[(metadata['Session']==session)&(metadata['Animal']==ani)&(metadata['FOV']==fov)].values[0]
    path = os.path.join(base_dir,'xml_metadata',ani,TSeries+'_timestamps.npz')
    timestamp_file = np.load(path)
    if stimulus & (session=='Recall1'):
        try:
            on_times = timestamp_file['tone_on_ts_1']
            off_times = timestamp_file['tone_off_ts_1']
        except:
            on_times = timestamp_file['tone_on_ts']
            off_times = timestamp_file['tone_off_ts']
        frame_times = timestamp_file['frame_times'] * 1000
        return frame_times,on_times,off_times
    elif stimulus & (session == 'Recall2'):
        on_times_1 = timestamp_file['tone_on_ts_1']
        off_times_1 = timestamp_file['tone_off_ts_1']
        on_times_2 = timestamp_file['tone_on_ts_2']
        off_times_2 = timestamp_file['tone_off_ts_2']
        frame_times = timestamp_file['frame_times'] * 1000
        return frame_times,on_times_1,off_times_1,on_times_2,off_times_2
    else:
        frame_times = timestamp_file['frame_times'] * 1000
        return frame_times
    
def write_timestamps(ani,fov,session,exp_info,raw_data_path,metadata_path,stimulus=True,recall2=False,TSeries=None):
    """
    Compute and save frame times and tone on/off timestamps for a session.

    Reads the voltage TTL recording (and XML acquisition metadata) for a
    TSeries, thresholds the voltage trace to identify tone onset/offset
    events, and saves frame times (and tone timestamps, if applicable) to an
    `.npz` file.

    Parameters
    ----------
    ani : str
        Animal identifier.
    fov : str
        FOV identifier.
    session : str
        Session identifier.
    exp_info : str or pandas.DataFrame
        Path to the metadata CSV for the experiment, or an already-loaded
        dataframe.
    raw_data_path : str
        Path to the raw TSeries data folder containing the voltage recording
        CSV.
    metadata_path : str
        Base folder for XML metadata; timestamps are saved under
        `metadata_path/{ani}/`.
    stimulus : bool, optional
        Whether this session includes tone stimuli (default True).
    recall2 : bool, optional
        If True, expects two tone channels (CS1 and CS2) in the voltage
        recording; if False, expects one (default False).
    TSeries : str, optional
        TSeries name; if None, inferred from the last path component of
        `raw_data_path` (default None).

    Returns
    -------
    None
        Writes a `.npz` file to
        `metadata_path/{ani}/{TSeries}_timestamps.npz` containing
        `frame_times` and, if `stimulus` is True, `tone_on_ts_1`/
        `tone_off_ts_1` (and `tone_on_ts_2`/`tone_off_ts_2` if `recall2` is
        True).
    """
    if stimulus & (recall2==False):
        voltage,voltage_times = get_voltage_ttl(raw_data_path,TSeries)
    if stimulus & (recall2==True):
        voltage,voltage2,voltage_times = get_voltage_ttl(raw_data_path,TSeries)
    metadata = get_xml_metadata(ani,fov,session,metadata_path,exp_info)
    frame_period = metadata['frame_period'].values[0]
    frames = np.arange(1,metadata['nFrames'].values[0]+1)
    frame_times = frame_period * frames
    if stimulus & recall2:
        tone_on_ts_1 = np.argwhere(np.diff(voltage > 4.5, prepend=False))[::2].flatten()
        tone_off_ts_1 = np.argwhere(np.diff(voltage > 4.5, prepend=False))[1::2].flatten()
        tone_on_ts_2 = np.argwhere(np.diff(voltage2 > 4.5, prepend=False))[::2].flatten()
        tone_off_ts_2 = np.argwhere(np.diff(voltage2 > 4.5, prepend=False))[1::2].flatten()
    elif stimulus & (recall2==False):
        tone_on_ts_1 = np.argwhere(np.diff(voltage > 4.5, prepend=False))[::2].flatten()
        tone_off_ts_1 = np.argwhere(np.diff(voltage > 4.5, prepend=False))[1::2].flatten()
    TSeries = exp_info['TSeries_g'].loc[(exp_info['Animal']==ani)&(exp_info['Session']==session)&(exp_info['FOV']==fov)].values[0]
    print('Saved timestamps as: '+os.path.join(metadata_path,TSeries+'_timestamps.npz'))
    if stimulus &(recall2==False):
        np.savez(os.path.join(metadata_path,ani,TSeries+'_timestamps.npz'),tone_on_ts_1=tone_on_ts_1,tone_off_ts_1=tone_off_ts_1,frame_times=frame_times)
    elif stimulus & (recall2==True):
        np.savez(os.path.join(metadata_path,ani,TSeries+'_timestamps.npz'),tone_on_ts_1=tone_on_ts_1,tone_off_ts_1=tone_off_ts_1,
                                                                            tone_on_ts_2=tone_on_ts_2,tone_off_ts_2=tone_off_ts_2,
                                                                            frame_times=frame_times)
    else:
        np.savez(os.path.join(metadata_path,ani,TSeries+'_timestamps.npz'),frame_times=frame_times)

    # plt.plot(voltage, 'r')
    # plt.vlines(frame_times * 1000, 0, 5)

def get_xml_metadata(ani,fov,session,metadata_folder,exp_info):
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
    metadata_folder : str
        Base folder containing per-animal metadata exports; the CSV is read
        from `metadata_folder/{ani}/GCaMP/{TSeries}_metadata_settings.csv`.
    exp_info : str or pandas.DataFrame
        Path to the metadata CSV for the experiment, or an already-loaded
        dataframe.

    Returns
    -------
    pandas.DataFrame
        Metadata settings dataframe for the matching TSeries.
    """
    if type(exp_info)==str:
        exp_info=pd.read_csv(exp_info)
    TSeries = exp_info['TSeries_g'].loc[(exp_info['Animal']==ani)&(exp_info['FOV']==fov)&(exp_info['Session']==session)].values[0]
    print(TSeries)
    csvfile = os.path.join(metadata_folder,ani,'GCaMP',TSeries+'_metadata_settings.csv')
    metadata=pd.read_csv(csvfile)
    return metadata

def get_voltage_ttl(raw_data_path,TSeries=None):
    """
    Load the voltage TTL recording for a TSeries.

    Parameters
    ----------
    raw_data_path : str
        TSeries path from Bruker raw data; must contain (in this folder or
        a subfolder) the voltage recording CSV named
        '{TSeries}_Cycle00001_VoltageRecording_001.csv'.
    TSeries : str, optional
        TSeries name; if None, inferred from the last path component of
        `raw_data_path` (default None).

    Returns
    -------
    voltage : ndarray
        Voltage signal from 'Input 1'.
    voltage2 : ndarray, optional
        Voltage signal from 'Input 2', only returned if present in the CSV.
    voltage_times : ndarray
        Timestamps (ms) for the voltage recording.
    """
    voltage_file=[]
    if TSeries is None:
        TSeries = os.path.split(raw_data_path)[-1]

    for root, dir, files in os.walk(raw_data_path):
        if TSeries + '_Cycle00001_VoltageRecording_001.csv' in files:
            voltage_file.append(os.path.join(root, TSeries + '_Cycle00001_VoltageRecording_001.csv'))

    stim_csv = voltage_file[0]
    stim_df = pd.read_csv(stim_csv)
    voltage_times = stim_df['Time(ms)'].values  # time of ttl in seconds
    voltage = stim_df[' Input 1'].values
    try:
        voltage2 = stim_df[' Input 2'].values
        return voltage, voltage2,voltage_times
    except:
        return voltage,voltage_times
