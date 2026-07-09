#%%
import numpy as np
import pandas as pd
import os
import pickle5
import matplotlib as mpl
import matplotlib.pyplot as plt
import ast
from scipy.stats import zscore
import holoviews as hv


def mean_rates(traces_df, session_name=None, time_ranges=None, 
               weighted=True, binarize=False, fr=30, threshold=.5):
    """
    Compute mean firing rates for each cell, with optional weighting, binarization, and multiple time windows.

    Parameters
    ----------
    traces_df : pandas.DataFrame
        Spike activity where rows are time points and the first column is
        timestamps.
    session_name : str, optional
        Name for this imaging session (default None).
    time_ranges : list of tuple of float, optional
        List of (start_time, end_time) tuples defining windows to average
        over; if None, uses the entire trace (default None).
    weighted : bool, optional
        Whether to use weighted event rate (default True).
    binarize : bool, optional
        Whether to binarize traces before computing event rate (default
        False).
    fr : int, optional
        Frame rate (default 30).
    threshold : float, optional
        Threshold used for binarizing traces (default 0.5).

    Returns
    -------
    result_df : pandas.DataFrame
        Dataframe with columns 'cell', 'mean_rate', 'is_engram', and,
        if `session_name` is given, 'session'.
    """

    timestamps = traces_df.iloc[:, 0]
    data_matrix = np.zeros(traces_df.shape[1] - 1)  # to accumulate rates
    total_frames = 0
    
    # If no specific time_ranges provided, use the entire session
    if time_ranges is None:
        time_ranges = [(timestamps.iloc[0], timestamps.iloc[-1])]
    
    for start_time, end_time in time_ranges:
        # Filter by time range
        filtered_df = traces_df[(timestamps >= start_time) & (timestamps <= end_time)]
        segment_matrix = filtered_df.iloc[:, 1:].T.to_numpy()

        if binarize:
            binarized_data = (segment_matrix > threshold).astype(int)
            rates = event_rate(binarized_data, fr=fr)
        elif weighted:
            rates = weight_event_rate(segment_matrix, fr=fr)
        else:
            rates = event_rate(segment_matrix, fr=fr)

        num_frames = segment_matrix.shape[1]
        data_matrix += rates * num_frames
        total_frames += num_frames

    mean_rates_across_epochs = data_matrix / total_frames

    # Extract and rename cells
    original_names = traces_df.columns[1:]
    renamed_cells = [f"E_{ast.literal_eval(name)[1]}" if eval(name)[0] else f"N_{ast.literal_eval(name)[1]}" for name in original_names]
    is_engram_list = [ast.literal_eval(col)[0] for col in original_names]

    # Create result DataFrame
    result_df = pd.DataFrame({'mean_rate': mean_rates_across_epochs, 'is_engram': is_engram_list}, index=renamed_cells).reset_index(names='cell')
    
    if session_name is not None:
        result_df['session'] = [session_name] * len(result_df)
    
    return result_df


def calculate_deciles_ani(ani_df):
    """
    Compute per-decile mean event rate and tagged-cell proportion for one animal.

    Splits cells into deciles based on 'Event Rate' and, for each decile,
    computes the mean event rate and the proportion of cells labeled
    'Tagged' in the 'Population' column.

    Parameters
    ----------
    ani_df : pandas.DataFrame
        Per-cell dataframe for a single animal, with columns 'Animal',
        'Event Rate', and 'Population'.

    Returns
    -------
    df : pandas.DataFrame
        Per-decile summary with columns 'Animal', 'Decile Mean Rates', and
        'P(Tagged)'.
    ani_df : pandas.DataFrame
        Input dataframe with an added 'Decile' column (0-9).
    """
    animal = ani_df['Animal'].values[0]
    ani_df['Decile']=pd.qcut(ani_df['Event Rate'], 10,labels=False)
    # ani_df['Decile']=ani_df['Decile']+1
    decile_ptagged = [ani_df.loc[(ani_df['Decile']==d)&(ani_df['Population']=='Tagged')].shape[0] 
                  / ani_df.loc[(ani_df['Decile']==d)].shape[0] for d in np.unique(ani_df['Decile'])]
    decile_means = [ani_df['Event Rate'].loc[ani_df['Decile']==d].mean() for d in np.unique(ani_df['Decile'])]
    df = pd.DataFrame()
    df['Animal']=[animal]*10
    df['Decile Mean Rates']=decile_means
    df['P(Tagged)']=decile_ptagged
    return df,ani_df


def ev_trace(events,times,T):
    """
    Convert event magnitudes and times into deconvolved activity traces.

    Parameters
    ----------
    events : array-like, length N
        Event magnitudes for each of N neurons; `events[i]` gives the
        event magnitudes for the i-th neuron.
    times : array-like, length N
        Event frame times for each of N neurons; `times[i]` gives the
        frame indices of each event for the i-th neuron.
    T : int
        Total number of frames.

    Returns
    -------
    ev : ndarray, shape (N, T)
        Deconvolved activity array.
    """
    N = len(events)
    ev = np.zeros([N,T])

    for i in np.arange(0,N):
        event_cell=events[i]
        times_cell = times[i]
        ev[i,times_cell]=event_cell
    return ev


def get_traces(animal, fov, session, file_key,base_dir):
    """
    Load deconvolution results for a given animal, FOV, and session.

    Parameters
    ----------
    animal : str
        Animal identifier.
    fov : str
        FOV identifier.
    session : str
        Session identifier.
    file_key : str
        Path to the metadata CSV for the experiment.
    base_dir : str
        Base directory containing the
        'deconvolution/deconvolution_results/deconv_results_min5' folder.

    Returns
    -------
    dff : ndarray
        dF/F traces.
    ev : ndarray or list
        Deconvolved event data.
    times : ndarray or list
        Event/frame times.
    est : ndarray
        Estimated (smoothed) calcium trace from the deconvolution model.
    """
    metadata = pd.read_csv(file_key)
    deconv_path = os.path.join(base_dir, 'deconvolution','deconvolution_results','deconv_results_min5')
    TSeries = metadata['TSeries_g'].loc[(metadata['Animal'] == animal) & (metadata['FOV'] == fov) & (metadata['Session'] == session)].values[0]
    dfile = [os.path.join(deconv_path, f) for f in os.listdir(deconv_path) if ((TSeries in f)&(f.endswith('l0deconv.pkl')))][0]
    print('deconvolution file: '+ dfile)
    with open(dfile, 'rb') as file:
        data = pickle5.load(file)
    dff = data['dff']
    times = data['times']
    ev = data['events']
    est = data['est']
    return dff, ev, times, est
 

def weight_event_rate(D,fr=30):
    """
    Compute weighted event rates for multiple cell traces over a session.

    Parameters
    ----------
    D : ndarray, shape (N, T)
        Deconvolved activity for N neurons over T frames. Data should not
        be downsampled in T.
    fr : int, optional
        Frame rate (default 30).

    Returns
    -------
    rates_wt : ndarray, shape (N,)
        Weighted event rate for each neuron, computed as the sum of
        positive event magnitudes divided by the session duration.
    """
    rates_wt = np.zeros(D.shape[0])
    T = D.shape[1]

    for i in np.arange(0, D.shape[0]):
        d = D[i]
        sum_mag = np.sum(d[d > 0])
        t = T / fr
        rates_wt[i] = sum_mag / t
    return rates_wt

def event_rate(D,fr=30):
    """
    Compute event rate for multiple cell traces over a session.

    All events are treated equally regardless of magnitude.

    Parameters
    ----------
    D : ndarray, shape (N, T)
        Deconvolved activity for N neurons over T frames, zero except at
        frames with calcium events. Data should not be downsampled in T.
    fr : int, optional
        Frame rate (default 30).

    Returns
    -------
    rates : ndarray, shape (N,)
        Event rate (events per second) for each neuron.
    """
    rates = np.zeros(D.shape[0])
    T = D.shape[1]
    for i in np.arange(0,D.shape[0]):
        d = D[i]
        n = d[d>0].shape[0]
        t = T/fr
        rates[i]=n/t
    return rates

def bin_traces_overlap(D,bin_size=4):
    """
    Bin traces into overlapping bins (50% overlap) using the sum.

    Parameters
    ----------
    D : ndarray, shape (N, T)
        Deconvolved calcium activity for N neurons over T frames.
    bin_size : int, optional
        Length of each bin, in frames (default 4).

    Returns
    -------
    d_bin : ndarray, shape (N, B)
        Binned activity array, where B = T / bin_size * 2, summing values
        within each overlapping bin.
    """
    T = D.shape[1]
    d_bin = np.empty([D.shape[0],int(T / bin_size * 2)])
    bin_starts = np.arange(0,T,bin_size/2,dtype=int)
    for n in np.arange(0,D.shape[0]):
        d = D[n,:]
        for i,b in enumerate(bin_starts):
            bin_end = b+bin_size
            try:
                d_bin[n,i] = np.sum(d[b:bin_end])
            except IndexError:
                pass
    return d_bin

def bin_traces(D, bin_size=4):
    """
    Bin traces into non-overlapping bins using the sum.

    Suitable for activity arrays.

    Parameters
    ----------
    D : ndarray, shape (N, T)
        Deconvolved calcium activity for N neurons over T frames.
    bin_size : int, optional
        Length of each bin, in frames (default 4).

    Returns
    -------
    d_bin : ndarray, shape (N, T // bin_size)
        Binned activity array, summing values within each bin.
    """
    T = D.shape[1]
    d_bin = np.empty([D.shape[0],int(T / bin_size)])
    t = np.arange(0, T, bin_size)
    for n in np.arange(0,D.shape[0]):
        d=D[n,:]
        for i in np.arange(0, t.shape[0]):
            try:
                d_bin[n,i] = np.sum(d[t[i]:t[i] + bin_size])
            except IndexError:
                pass
    return d_bin

def bin_traces_time(D, bin_size=4):
    """
    Bin a time-series array into non-overlapping bins using the mean.

    Suitable for downsampling time arrays.

    Parameters
    ----------
    D : ndarray, shape (M, T)
        Input time array, where M is trials (or cells) and T is
        timepoints. 1D input is treated as a single trial.
    bin_size : int, optional
        Size of each bin, in frames (default 4).

    Returns
    -------
    d_bin : ndarray, shape (M, T // bin_size)
        Binned activity array, using the mean within each bin. Any
        trailing frames that don't fill a full bin are dropped.
    """
    if D.ndim == 1:
        D = D[np.newaxis, :]  # Convert to 2D if 1D input

    M, T = D.shape
    num_bins = T // bin_size
    D = D[:, :num_bins * bin_size]  # Truncate to fit full bins only

    D_reshaped = D.reshape(M, num_bins, bin_size)
    d_bin = D_reshaped.mean(axis=2)

    return d_bin

def bin_overlap_time(D, bin_size=4, overlap=0.5, agg="mean"):
    """
    Bin a time-series array into overlapping bins.

    Parameters
    ----------
    D : ndarray, shape (M, T) or (T,)
        Input time array, where M is trials and T is timepoints; a 1D
        array of shape (T,) is treated as a single trial.
    bin_size : int, optional
        Size of each bin, in frames (default 4).
    overlap : float, optional
        Fractional overlap between consecutive bins, in [0, 1). An
        overlap of 0.5 means 50% overlap (step size = bin_size / 2)
        (default 0.5).
    agg : {"mean", "sum"}, optional
        Aggregation function applied within each bin (default "mean").

    Returns
    -------
    d_bin : ndarray, shape (M, B)
        Binned array, where B is the number of overlapping bins. Only
        full bins (windows entirely within [0, T)) are included.

    Raises
    ------
    ValueError
        If `overlap` is not in [0, 1), if `bin_size` is not positive, or
        if `agg` is not "mean" or "sum".
    """
    if D.ndim == 1:
        D = D[np.newaxis, :]

    if not (0 <= overlap < 1):
        raise ValueError("overlap must be in [0, 1).")
    if bin_size <= 0:
        raise ValueError("bin_size must be a positive integer.")

    M, T = D.shape
    step = max(1, int(round(bin_size * (1 - overlap))))

    # Start indices such that [start, start+bin_size) is fully within the array
    starts = np.arange(0, T - bin_size + 1, step, dtype=int)
    B = starts.size
    d_bin = np.empty((M, B), dtype=float)

    for i, s in enumerate(starts):
        window = D[:, s:s + bin_size]
        if agg == "mean":
            d_bin[:, i] = window.mean(axis=1)
        elif agg == "sum":
            d_bin[:, i] = window.sum(axis=1)
        else:
            raise ValueError('agg must be "mean" or "sum".')

    return d_bin

def plot_deconv(S, i, T, frame_rate=30):
    """
    Plot dF/F, deconvolution model fit, and events for one or more cells.

    Works best in a Jupyter notebook with holoviews' bokeh backend
    enabled (``import holoviews as hv; hv.extension('bokeh')``).

    Parameters
    ----------
    S : dict
        Deconvolution output with keys:

        - ``'dff'`` : ndarray, shape (N, T), dF/F traces for N neurons
          over T frames.
        - ``'estimated_calcium'`` : ndarray, shape (N, T), smoothed model
          estimate of calcium from deconvolution.
        - ``'event_mag'`` : list, length N, each item the event
          magnitudes for that cell.
        - ``'spiketimes'`` : list, length N, each item the event times
          (frames) for that cell.
    i : int or array-like of int
        Index (or indices) of the cell(s) to plot.
    T : int
        Total number of frames for all traces (should match the time
        dimension of ``S['dff']`` and ``S['estimated_calcium']``).
    frame_rate : int, optional
        Imaging frame rate, in fps (default 30).

    Returns
    -------
    plot : holoviews.Overlay
        Holoviews object with the raw dF/F, model fit, deconvolved
        activity, and spike traces overlaid.
    """
    est = S['estimated_calcium'][i]
    dff = S['dff'][i]
    event_mag_pos = np.squeeze(S['event_mag'][i])
    spikes = np.squeeze(S['spiketimes'][i])

    event_mag = np.zeros(T)
    event_mag[spikes] = event_mag_pos

    event_plot = spikes, -1 * np.ones(spikes.shape)
    t = T / frame_rate

    plot = hv.Curve(dff, label='Raw df/f').opts(width=900, height=300, ylabel='df/f', xlabel='Frame #') * \
           hv.Curve(est, label='Deconvolution model fit').opts(width=900, height=300, color='#e5ae38') * \
           hv.Curve(event_mag, label='Deconvolved activity').opts(width=900, height=300, color='red') * \
           hv.Scatter(event_plot, label='Spikes').opts(fill_color='k', line_color='k', size=3)
    plot.opts(legend_position='right')
    return plot

def zscore_traces(traces_df):
    """
    Z-score each cell's activity over time.

    Parameters
    ----------
    traces_df : pandas.DataFrame
        Dataframe where rows are time points and all columns except the
        first (timestamps) are cells.

    Returns
    -------
    zscored_df : pandas.DataFrame
        Dataframe with the timestamp column preserved and each cell's
        values Z-scored.
    """
    timestamps = traces_df.iloc[:, 0]  # Preserve the timestamp column
    data_matrix = traces_df.iloc[:, 1:]  # Exclude timestamps

    # Apply Z-score normalization for each cell (column-wise)
    zscored_data = data_matrix.apply(zscore, axis=0, nan_policy='omit')

    # Reconstruct DataFrame with timestamps
    zscored_df = pd.concat([timestamps, zscored_data], axis=1)
    
    return zscored_df

def ridge_plot(array,times,overlay=None,trace_spacing=5,ytick_spacing=1,title=None,color='black',text_color = None,
               alpha=0.5,line_width=1,ax = None,legend=False,eventtimes=None,eventoffset=3,event_linewidth=1,event_linelength=.5,event_alpha=1,event_color='k'):
    """
    Plot a ridge plot of multiple cells' activity traces.

    Adapted from E. Thomson, CaImAn Flatiron CCN Workshop, 06/2022.

    Parameters
    ----------
    array : ndarray, shape (N, T)
        Traces (e.g. dF/F) for N cells over T timepoints.
    times : ndarray, shape (T,)
        Time values corresponding to the columns of `array`.
    overlay : ndarray, shape (N, T), optional
        A second trace (e.g. a denoised trace) to plot on top of each
        cell's trace (default None).
    trace_spacing : float, optional
        Vertical distance between each cell's trace on the y-axis
        (default 5).
    ytick_spacing : int, optional
        Period, in number of traces, between y-tick labels (default 1).
    title : str, optional
        Plot title (default None).
    color : str, tuple, list, or matplotlib.colors.LinearSegmentedColormap, optional
        Line color: a matplotlib color string, an (r, g, b) tuple, a
        colormap (sampled evenly across traces), or a list of colors with
        one entry per trace (default 'black').
    text_color : str, optional
        Color for text and axes (default None, uses matplotlib defaults).
    alpha : float, optional
        Alpha (transparency) for each trace (default 0.5).
    line_width : float, optional
        Line width for traces (default 1).
    ax : matplotlib.axes.Axes, optional
        Axes to draw into; if None, a new figure and axes are created
        (default None).
    legend : bool, optional
        If True, label the raw and overlay traces for use in a legend
        (default False).
    eventtimes : array-like, shape (N,), optional
        Per-cell array of event times (same timescale as `times`) to plot
        as tick marks beneath each trace (default None).
    eventoffset : float, optional
        Fraction of `trace_spacing` used to offset event marks below each
        trace (default 3).
    event_linewidth : float, optional
        Line width for event marks (default 1).
    event_linelength : float, optional
        Line length for event marks (default 0.5).
    event_alpha : float, optional
        Alpha for event marks (default 1).
    event_color : str, optional
        Color for event marks, or 'match' to use each trace's own color
        (default 'k').

    Returns
    -------
    ax : matplotlib.axes.Axes
        Axes with the ridge plot drawn.
    """
    num_traces = array.shape[0]
    num_yticks = int(num_traces//ytick_spacing)

    # set y position of each trace
    y_position_traces = np.linspace(1, (num_traces)*trace_spacing, num=num_traces)

    # set y tick properties
    y_ticks = np.linspace(1, (num_traces)*trace_spacing, num=num_yticks)
    y_tick_labels = np.arange(1, (num_traces+1)+2*ytick_spacing, ytick_spacing, dtype=np.uint8) # +2*y_tick_spacing just for insurance
    y_tick_labels = y_tick_labels[:num_yticks]

    if ax is None:
        f, ax = plt.subplots()

    #set colors of lines
    for ind, line in enumerate(array):
        floats = np.linspace(0,1,array.shape[0])
        if type(color) is mpl.colors.LinearSegmentedColormap:
            c = color(floats[ind])
        elif (type(color)==np.ndarray) or (type(color)==list):
            c = color[ind]
        else:
            c=color
        dff,=ax.plot(times,line+y_position_traces[ind],color=c,alpha=alpha,linewidth=line_width)
        if eventtimes is not None:
            if event_color =='match':
                event_c=c
            else:
                event_c = event_color
            ax.eventplot(eventtimes[ind],lineoffsets=y_position_traces[ind]-(trace_spacing/eventoffset),color=event_c,alpha=event_alpha,linewidths=event_linewidth,linelengths=event_linelength)
        if overlay is not None: #can overlay another trace in a translucent lighter same color
            smooth,=ax.plot(times,overlay[ind,:]+y_position_traces[ind],
                    color=c,alpha=1,linewidth=line_width/2)
   # set text color
    if text_color is not None:
        print('text color')
        mpl.rcParams['text.color'] = text_color
        mpl.rcParams['axes.labelcolor'] = text_color
        mpl.rcParams['xtick.color'] = text_color
        mpl.rcParams['ytick.color'] = text_color
        mpl.rcParams['axes.edgecolor'] = text_color

    # only show left/bottom axis lines
    if legend:
        dff.set_label('Raw dF/F')
        if overlay is not None:
            smooth.set_label('Model Fit')
        #ax.legend(bbox_to_anchor = (2, 0.6))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ylims = ax.get_ylim()
    # set ylimits to make it pretty (this could use some tweaking probably)
    #ax.set_ylim(0.1*ylims[0], ylims[1]-0.05*ylims[1])
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_tick_labels)
    #plt.autoscale(enable=True, axis='x', tight=True)
    return ax

def normalize_eventtimes(times,tmin,tmax,frame_rate=30):
    """
    Restrict event times to a window and convert to seconds relative to window start.

    Parameters
    ----------
    times : array-like
        Event times, in frames.
    tmin : float
        Start of the window, in frames.
    tmax : float
        End of the window, in frames.
    frame_rate : int, optional
        Frame rate used to convert frames to seconds (default 30).

    Returns
    -------
    ndarray
        Event times within [tmin, tmax], shifted so that `tmin` is 0, and
        converted to seconds.
    """
    tprime = times[(times >= tmin) & (times <= tmax)]
    tprime = tprime-tmin
    return tprime/frame_rate