import argparse
from pairwise_corr_utils import *
import os
import time
import sys
sys.path.extend(['/projectnb/sramirezlab/amonast/Engram_2P/Engram_2P/utilities'])
from animal import animal
from traces import bin_traces_time,bin_overlap_time

def main(args=None):
    parser = argparse.ArgumentParser(
        description='Pairwise correlations: CS+trace vs baseline matched for running time.')
    parser.add_argument('ani',    type=str, help='Animal name (e.g. 992N)')
    parser.add_argument('binned', type=int, help='Bin size in frames (e.g. 10)')
    parser.add_argument('--no_correlations',action='store_true', default=False,
                        help='skip the correlations, just plot the selected epochs')
    parser.add_argument('--overlap', action='store_true', default=False, help='Use overlapping bins')
    parser.add_argument('--trace_period', type=int, default=25,
                        help='Seconds after CS offset to include in trace period (default 25)')
    parser.add_argument('--plot', action='store_true', default=False,
                        help='Save a diagnostic plot of epoch selections over velocity trace')
    args = parser.parse_args(args)

    t_start = time.time()
    ani = args.ani
    b   = args.binned
    overlap = args.overlap
    trace_period = args.trace_period
    correlate = not args.no_correlations
    do_plot = args.plot
    fov = 'FOV1'
    base_dir  = '/Users/amonast/Desktop/Tone2P'
    file_key  = base_dir + '/Data_info_TFC.csv'
    binned_path = base_dir + '/Traces/binned_traces'

    if overlap:
        data = np.load(os.path.join(binned_path, "binned_traces_overlap" + str(b),
                                    ani + "_" + fov + "_" + str(b) + "_binned_traces_split_all.npz"))
    else:
        data = np.load(os.path.join(binned_path, "binned_traces_" + str(b),
                                    ani + "_" + fov + "_" + str(b) + "_binned_traces_split_all.npz"))

    d_tag = data['d_post_tag']
    d_non = data['d_post_non']
    t     = data['t'].squeeze()

    mouse = animal(ani, 'FOV1', file_key, base_dir)
    v     = mouse.compute_velocity(sessions=['Recall1'], window_size=30, threshold=1)
    vt    = bin_overlap_time(v, bin_size=b) if overlap else bin_traces_time(v, bin_size=b)
    run_bool = (vt > 0).squeeze()

    cs1_start, cs1_stop = mouse.load_tone_times(session_name='Recall1')

    # ---- CS + trace epoch (full windows, no running filter) ----
    cs_trace_windows = [(float(s), float(e) + trace_period)
                        for s, e in zip(cs1_start, cs1_stop)]
    cs_trace_mask = mask_from_windows(t, cs_trace_windows)

    # ---- Baseline pool: before first CS onset (minus 20 s buffer) ----
    bl_mask = mask_from_windows(t, [(0.0, float(cs1_start[0]) - 20.0)])

    # ---- Match running composition of CS+trace in baseline ----
    n_run_cs   = int((cs_trace_mask & run_bool).sum())
    n_total_cs = int(cs_trace_mask.sum())
    n_rest_cs  = n_total_cs - n_run_cs

    try:
        bl_run_matched = sample_equal_frames(bl_mask & run_bool, n_run_cs)
    except ValueError:
        n_run_cs = int((bl_mask & run_bool).sum())
        bl_run_matched = bl_mask & run_bool
        print(f"  Warning: only {n_run_cs} running frames in baseline; will trim cs_trace.")

    try:
        bl_rest_matched = sample_equal_frames(bl_mask & ~run_bool, n_rest_cs)
    except ValueError:
        n_rest_cs = int((bl_mask & ~run_bool).sum())
        bl_rest_matched = bl_mask & ~run_bool
        print(f"  Warning: only {n_rest_cs} rest frames in baseline; will trim cs_trace.")

    bl_matched = bl_run_matched | bl_rest_matched

    # trim cs_trace to match the actual baseline frame count if either cap was hit
    n_bl_total = int(bl_matched.sum())
    if n_bl_total < n_total_cs:
        cs_trace_mask = sample_equal_frames(cs_trace_mask, n_bl_total)
        print(f"  cs_trace trimmed from {n_total_cs} → {n_bl_total} frames to match baseline.")
        n_total_cs = n_bl_total

    print(f"CS+trace frames: {n_total_cs}  (running={n_run_cs}, rest={n_rest_cs})")
    print(f"Baseline matched frames: {bl_matched.sum()}")

    epochs = {
        'cs_trace': cs_trace_mask,
        'baseline': bl_matched,
    }
    if overlap:
        savepath = f'{base_dir}/Analysis/correlation/pairwise_correlation_new/overlap/cstrace_total/{ani}_{b}_cstrace_beh_matched.csv'
    else:
        savepath = f'{base_dir}/Analysis/correlation/pairwise_correlation_new/no_overlap/cstrace_total/{ani}_{b}_cstrace_beh_matched.csv'
        
    if correlate:
        DF = pairwise_corr_epochs(d_tag, d_non,
                                  t=t,
                                  named_epochs=epochs,
                                  ani=ani, group=mouse.group, fov=fov,
                                  corr_method='spearman')

        os.makedirs(os.path.dirname(savepath), exist_ok=True)
        DF.to_csv(savepath)
        print(f'saved {savepath} ({time.time() - t_start:.1f}s)')

    if do_plot:
        ax = plot_epoch_selection(
            t, vt.squeeze(),
            named_epochs=epochs,
            cs_starts=cs1_start,
            cs_stops=cs1_stop,
            title=f'{ani}  bin={b}  trace_period={trace_period}',
        )
        plotpath = savepath.replace('.csv', '_epoch_selection.png')
        ax.figure.savefig(plotpath, dpi=150, bbox_inches='tight')
        plt.close(ax.figure)

if __name__ == '__main__':
    main()