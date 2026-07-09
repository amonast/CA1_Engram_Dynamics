import os
import numpy as np
import pandas as pd
import ast
import sys
sys.path.extend(['/Users/amonast/Documents/GitHub/Engram_2P/Engram_2P',
                 '/Users/amonast/Documents/GitHub/Engram_2P/Amy_2P/analysis/correlation/PairwiseCorrelations'])
from Networks.util_func import distRiemLE
from utilities.animal import animal
from Correlation import correlation
from pairwise_corr_utils import sample_equal_frames


def mask_valid_pair(mat_a, mat_b):
    """
    Restrict two adjacency matrices to the cell pairs valid in both.

    A cell is considered invalid in a matrix if its diagonal entry equals the
    dummy value 999 (as produced by `correlation.get_corr_matrix` for
    constant/zero traces). Both matrices are subset to the intersection of
    valid cells, preserving cell order.

    Parameters
    ----------
    mat_a : ndarray, shape (N, N)
        First adjacency/correlation matrix.
    mat_b : ndarray, shape (N, N)
        Second adjacency/correlation matrix, same shape as `mat_a`.

    Returns
    -------
    tuple of ndarray
        (mat_a, mat_b) restricted to rows/columns where both matrices have a
        valid (non-999) diagonal entry.
    """
    
    valid = (np.diag(mat_a) != 999) & (np.diag(mat_b) != 999)
    return mat_a[np.ix_(valid, valid)], mat_b[np.ix_(valid, valid)]


def main():
    """
    Compute rest-matched Riemannian network distances between Baseline and Post sessions.
    For each animal/FOV, loads dF/F traces and cell registration across
    Baseline and Post sessions, computes velocity to identify resting frames,
    and subsamples an equal number of resting frames from each session. For
    engram and non-engram cell populations separately, builds Spearman
    correlation (adjacency) matrices from the matched resting frames, masks
    out cells invalid in either session, and computes the Riemannian distance
    (log-Euclidean) between the Baseline and Post adjacency matrices,
    normalized by the square root of the number of cells. Results are
    compiled into a dataframe and written to CSV.

    Returns
    -------
    net_distance : dict
        Nested dictionary of normalized Riemannian distances, keyed as
        ``net_distance[label][f"{animal}_{fov}"]`` for
        ``label in {'engram', 'non_engram'}``.

    """
    data_dict = {'149L': ['FOV1', 'FOV2'],
                 '034R': ['FOV1'],
                 '146R': ['FOV2'],
                 '217N': ['FOV1','FOV2'],
                 '217R': ['FOV1'],
                 '218L': ['FOV1', 'FOV2'],
                 '989N': ['FOV1', 'FOV2'],
                 '9972R':['FOV1','FOV2'],
                 '994R': ['FOV1', 'FOV2'],
                 '160R': ['FOV1', 'FOV2'],
                 '493R': ['FOV1'],
                 '492N': ['FOV1', 'FOV2'],
                 '1912L': ['FOV1', 'FOV2']}

    file_key = '/Volumes/AM_SSD1/Spont2P/Data_info.csv'
    base_dir = '/Volumes/AM_SSD1/Spont2P'
    sessions = ['Baseline', 'Post']

    net_distance = {'engram': {}, 'non_engram': {}}

    for ani in data_dict.keys():
        fovs = data_dict[ani]
        print(f"Processing {ani}...")
        for fov in fovs:
            mouse = animal(ani, fov, file_key=file_key, base_dir=base_dir)
            traces_dict = mouse.load_traces(sessions=sessions, signal='dff')
            cellreg_df = mouse.load_cellreg(sessions=sessions)

            # Velocity and timestamps (outside label loop)
            v_base = mouse.compute_velocity(sessions=['Baseline'], window_size=30, threshold=1)
            v_post = mouse.compute_velocity(sessions=['Post'],     window_size=30, threshold=1)
            t_base = traces_dict['Baseline']['timestamps'].values
            t_post = traces_dict['Post']['timestamps'].values
            run_bool_base = (v_base > 0).squeeze()
            run_bool_post = (v_post > 0).squeeze()

            # Equal rest frames across full session
            base_rest_mask = ~run_bool_base
            post_rest_mask = ~run_bool_post
            n_frames = min(base_rest_mask.sum(), post_rest_mask.sum())
            base_mask = sample_equal_frames(base_rest_mask, n_frames)
            post_mask = sample_equal_frames(post_rest_mask, n_frames)
            print(f"  Rest frames — Baseline: {base_mask.sum()}, Post: {post_mask.sum()}")

            for label in ['engram', 'non_engram']:
                baseline_df = traces_dict['Baseline']
                post_df     = traces_dict['Post']
                baseline_cols = [ast.literal_eval(col) for col in baseline_df.columns[1:]]

                matched_baseline, matched_post = [], []
                for is_engram, base_id in baseline_cols:
                    if (label == 'engram' and is_engram) or (label == 'non_engram' and not is_engram):
                        match_row = cellreg_df[cellreg_df['Baseline'] == base_id]
                        if not match_row.empty:
                            post_id = int(match_row.iloc[0]['Post'])
                            if post_id != -1:
                                matched_baseline.append((is_engram, int(base_id)))
                                matched_post.append((is_engram, post_id))

                matched_baseline_str = [str(t) for t in matched_baseline]
                matched_post_str     = [str(t) for t in matched_post]

                df_base = baseline_df.loc[:, ['timestamps'] + matched_baseline_str]
                df_post = post_df.loc[:,     ['timestamps'] + matched_post_str]

                # Apply epoch masks (N x T)
                act_base = df_base.iloc[:, 1:].values.T[:, base_mask]
                act_post = df_post.iloc[:, 1:].values.T[:, post_mask]

                C_base = correlation(act_base)
                C_base.get_corr_matrix(method='spearman')
                adj_base = C_base.corr

                C_post = correlation(act_post)
                C_post.get_corr_matrix(method='spearman')
                adj_post = C_post.corr

                adj_base, adj_post = mask_valid_pair(adj_base, adj_post)
                dist = distRiemLE(adj_base, adj_post)
                net_distance[label][f"{ani}_{fov}"] = dist / np.sqrt(adj_base.shape[0])

    info = pd.read_csv(file_key)
    df_rows = []
    for label in ['engram', 'non_engram']:
        for id_fov, dist in net_distance[label].items():
            ani, fov = id_fov.split('_')
            group = info[info['Animal'] == ani]['Group'].values[0]
            df_rows.append({'animal': ani, 'fov': fov, 'group': group,
                            'population': label, 'riem_dist': dist})

    df_net = pd.DataFrame(df_rows)
    print(df_net.head())
    df_net.to_csv('/Volumes/AM_SSD1/Spont2P/Analysis/networks/rieman_distance_rest_matched.csv', index=False)
    return net_distance


if __name__ == '__main__':
    main()
