import numpy as np
import pandas as pd
import seaborn as sb
import pickle
import scipy.stats as stats
import os
import matplotlib.pyplot as plt
import scipy.stats as stats
import sys
import scipy.signal
from tqdm import tqdm
sys.path.extend(['/projectnb/sramirezlab/amonast/Engram_2P/Engram_2P',
                 '/Users/amonast/Documents/GitHub/Engram_2P/Engram_2P'])
from spatial.rois import *

class correlation:
    """
    Class for computing pairwise correlation matrices from activity traces.
    Parameters
    ----------
    dff : ndarray, shape (N, T)
        dF/F traces (or deconvolved activity) for N neurons over T frames.
    idx : array-like of int, optional
        Indices of cells to subset from `dff` (default None, meaning all
        cells are used).
    Attributes
    ----------
    dff : ndarray, shape (N, T)
        dF/F traces, optionally subset by `idx`.
    """
    def __init__(self, dff, idx=None):
        if idx is None:
            self.dff = dff
        else:
            self.dff = dff[idx, :]


    def get_corr_matrix(self,method='spearman'):
        """
        Compute the pairwise correlation matrix between all cells in `self.dff`.
        Only correlates a set of cells with itself. To correlate one group of
        cell traces against another, use the module-level `corr_matrix` function.

        Parameters
        ----------
        method : {'spearman', 'kendall', 'jaccard'}, optional
            Correlation method: Spearman's rho, Kendall's tau-b, or Jaccard
            similarity of binarized (>0) traces (default 'spearman'). Cell pairs
            with all-zero or zero-std traces are assigned a dummy value of 999.
        Returns
        -------
        corr_r : ndarray, shape (N, N)
            Correlation (or similarity) coefficient for each pair of cells.
        corr_p : ndarray, shape (N, N)
            P-values for each pair of cells. Only returned when `method` is not
            'jaccard'.
        """
        corr_r = np.zeros([self.dff.shape[0], self.dff.shape[0]])
        corr_p = np.zeros([self.dff.shape[0], self.dff.shape[0]])
        # if method=='spearman':
        #     print('using spearman rho')
        # elif method=='kendall':
        #     print('using kendall tau')

        for i in np.arange(0, self.dff.shape[0]):
            dff_1 = self.dff[i]
            for j in np.arange(0, self.dff.shape[0]):
                dff_2 = self.dff[j]
                if not ((dff_1.any()) & dff_2.any()) or np.std(dff_1) == 0 or np.std(dff_2) == 0:
                    corr_r[i, j] = 999
                    corr_p[i, j] = 999
                else:
                    if method=='spearman':
                        corr_r[i, j] = stats.spearmanr(dff_1, dff_2)[0]
                        corr_p[i, j] =  stats.spearmanr(dff_1,dff_2)[1]
                    elif method=='kendall':
                        corr_r[i,j] = stats.kendalltau(dff_1,dff_2)[0]
                        corr_p[i,j] =  stats.kendalltau(dff_1,dff_2)[1]
                    elif method=='jaccard':
                        # Compute intersection and union
                        X = (dff_1 > 0).astype(int)
                        Y = (dff_2 > 0).astype(int)
                        intersection = np.sum((X == 1) & (Y == 1))
                        union = np.sum((X == 1) | (Y == 1))
                        if union == 0:
                            corr_r[i,j]=999 #dummy value
                        else:
                            corr_r[i,j] =  intersection / union

        self.corr = corr_r
        self.pvals = None if np.all(corr_p == 0) else corr_p
        if method=='jaccard':
            return corr_r
        else:
            return corr_r,corr_p
            
    def cross_corr_with_array(self, target_array, max_lags=None, return_xcorr=False):
        """
        Compute cross-correlation of each cell's trace with a target array.

        Parameters
        ----------
        target_array : ndarray, shape (T,) or (M, T)
            1D array or 2D array (num_targets x timepoints) to correlate with
            each cell's trace.
        max_lags : int, optional
            Maximum number of lags to consider in the correlation computation
            (default None, meaning all lags are considered).
        return_xcorr : bool, optional
            If True, also return a dictionary of full cross-correlation traces
            for each cell-target pair (default False).

        Returns
        -------
        corr_r : ndarray, shape (N, M)
            Peak cross-correlation coefficient for each cell-target pair.
        lags_matrix : ndarray, shape (N, M)
            Lag at which the peak cross-correlation occurs, for each cell-target
            pair.
        xcorr_dict : dict, optional
            Dictionary mapping (cell_idx, target_idx) to (lags, xcorr) arrays.
            Only returned if `return_xcorr` is True.
        """
        num_cells = self.dff.shape[0]
        if target_array.ndim == 1:
            target_array = target_array[np.newaxis, :]
        num_targets = target_array.shape[0]

        corr_r = np.empty([num_cells, num_targets])
        corr_r.fill(np.nan)
        lags_matrix = np.empty([num_cells, num_targets])
        xcorr_dict = {} if return_xcorr else None

        for i in range(num_cells):
            dff_1 = self.dff[i]
            std_dff_1 = np.std(dff_1)

            for j in range(num_targets):
                signal2 = target_array[j]
                std_2 = np.std(signal2)

                if std_dff_1 == 0 or std_2 == 0:
                    corr_r[i, j] = np.nan
                    lags_matrix[i, j] = np.nan
                    if return_xcorr:
                        xcorr_dict[(i, j)] = ([], [])
                    continue

                xcorr = scipy.signal.correlate(dff_1, signal2, mode='full')
                lags = scipy.signal.correlation_lags(dff_1.size, signal2.size, mode='full')

                if max_lags is not None:
                    valid_lags = np.where((lags >= -max_lags) & (lags <= max_lags))
                    xcorr = xcorr[valid_lags]
                    lags = lags[valid_lags]

                max_idx = np.argmax(np.abs(xcorr))
                corr_r[i, j] = xcorr[max_idx] / (std_dff_1 * std_2 * len(dff_1))
                lags_matrix[i, j] = lags[max_idx]

                if return_xcorr:
                    xcorr_dict[(i, j)] = (lags, xcorr)

        if return_xcorr:
            return corr_r, lags_matrix, xcorr_dict
        
        return corr_r, lags_matrix

    def get_cross_corr_matrix_with_pvals(self, max_lags=None, n_perms=1000):
        """
        Compute pairwise cross-correlation with empirical p-values via permutation.
        For each pair of cells, finds the peak cross-correlation (and its lag),
        then estimates a p-value by comparing the observed peak correlation to a
        null distribution built from `n_perms` random permutations of one trace.

        Parameters
        ----------
        max_lags : int, optional
            Maximum number of lags to consider in the correlation computation
            (default None, meaning all lags are considered).
        n_perms : int, optional
            Number of permutations used to build the null distribution for each
            cell pair (default 1000).

        Returns
        -------
        corr_r : ndarray, shape (N, N)
            Peak cross-correlation value for each pair of cells.
        pvals : ndarray, shape (N, N)
            Empirical p-value for each pair of cells.
        lags_matrix : ndarray, shape (N, N)
            Lag at which the peak correlation occurs, for each pair of cells.
        """
        num_cells = self.dff.shape[0]
        corr_r = np.full((num_cells, num_cells), np.nan)
        pvals = np.full((num_cells, num_cells), np.nan)
        lags_matrix = np.full((num_cells, num_cells), np.nan)

        for i in tqdm(range(num_cells)):
            dff_1 = self.dff[i]
            std_1 = np.std(dff_1)
            if std_1 == 0:
                continue
            for j in range(num_cells):
                dff_2 = self.dff[j]
                std_2 = np.std(dff_2)
                if std_2 == 0:
                    continue
                xcorr = scipy.signal.correlate(dff_1, dff_2, mode='full')
                lags = scipy.signal.correlation_lags(len(dff_1), len(dff_2), mode='full')
                if max_lags is not None:
                    mask = (lags >= -max_lags) & (lags <= max_lags)
                    xcorr = xcorr[mask]
                    lags = lags[mask]

                max_idx = np.argmax((xcorr))
                N_overlap = len(dff_1) - abs(lags[max_idx])
                obs_corr = xcorr[max_idx] / (std_1 * std_2 * N_overlap)
                peak_lag = lags[max_idx]

                corr_r[i, j] = obs_corr
                lags_matrix[i, j] = peak_lag

                null_corrs = np.zeros(n_perms)
                for k in range(n_perms):
                    dff_2_perm = np.random.permutation(dff_2)
                    xcorr_perm = scipy.signal.correlate(dff_1, dff_2_perm, mode='full')
                    if max_lags is not None:
                        xcorr_perm = xcorr_perm[mask]
                    null_corrs[k] = np.max(xcorr_perm) / (std_1 * std_2 *N_overlap)

                pvals[i, j] = np.mean(null_corrs >= obs_corr)

        self.xcorr = corr_r
        self.xcorr_pval = pvals
        self.xcorr_lags = lags_matrix
        return corr_r, pvals, lags_matrix

    def get_cross_corr_matrix(self, max_lags=None, return_xcorr=False):
        """
        Compute the pairwise cross-correlation matrix over a window of lags.
        Only correlates a set of cells with itself. 

        Parameters
        ----------
        max_lags : int, optional
            Maximum number of lags to consider in the correlation computation
            (default None, meaning all lags are considered).
        return_xcorr : bool, optional
            If True, also return a dictionary of full cross-correlation traces
            for each cell pair (default False).

        Returns
        -------
        corr_r : ndarray, shape (N, N)
            Peak cross-correlation coefficient for each pair of cells.
        lags_matrix : ndarray, shape (N, N)
            Lag at which the peak cross-correlation occurs, for each pair of
            cells.
        xcorr_dict : dict, optional
            Dictionary mapping (i, j) cell index pairs to (lags, xcorr) arrays.
            Only returned if `return_xcorr` is True.
        """
        num_cells = self.dff.shape[0]
        corr_r = np.empty([num_cells, num_cells])
        corr_r.fill(np.nan)  # Initialize with NaNs to handle zero standard deviation cases
        lags_matrix = np.empty([num_cells, num_cells])
        xcorr_dict = {} if return_xcorr else None
        
        for i in range(num_cells):
            dff_1 = self.dff[i]
            std_dff_1 = np.std(dff_1)
            for j in range(num_cells):
                dff_2 = self.dff[j]
                std_dff_2 = np.std(dff_2)
                if std_dff_1 == 0 or std_dff_2 == 0:
                    corr_r[i, j] = np.nan  # Avoid division by zero if either signal is constant (zero std deviation)
                    lags_matrix[i, j] = np.nan
                    if return_xcorr:
                        xcorr_dict[(i, j)] = ([], [])  # Store empty arrays for xcorr
                    continue
                xcorr = scipy.signal.correlate(dff_1, dff_2, mode='full')
                lags = scipy.signal.correlation_lags(dff_1.size, dff_2.size, mode='full')
                if max_lags is not None:
                    valid_lags = np.where((lags >= -max_lags) & (lags <= max_lags))
                    xcorr = xcorr[valid_lags]
                    lags = lags[valid_lags]
                max_idx = np.argmax(xcorr)  # Get index of max correlation
                N_overlap = len(dff_1) - abs(lags[max_idx])
                corr_r[i, j] = xcorr[max_idx] / (std_dff_1 * std_dff_2 * N_overlap)
                lags_matrix[i, j] = lags[max_idx]
                if return_xcorr:
                    xcorr_dict[(i, j)] = (lags, xcorr)
        self.xcorr = corr_r
        self.lags = lags
        if return_xcorr:
            return corr_r, lags_matrix, xcorr_dict
        return corr_r, lags_matrix

    def get_lower_cross_corr_with_pvals(self):
        """
        Extract the lower-triangle values of the cross-correlation, p-value, and lag matrices.
        Returns
        -------
        corr_vals : ndarray
            Lower-triangle (excluding diagonal) values of `self.xcorr`.
        pval_vals : ndarray
            Lower-triangle (excluding diagonal) values of `self.xcorr_pval`.
        lag_vals : ndarray
            Lower-triangle (excluding diagonal) values of `self.xcorr_lags`.
        """
        if not hasattr(self, 'xcorr') or not hasattr(self, 'xcorr_pval') or not hasattr(self, 'xcorr_lags'):
            raise AttributeError("Missing one or more of: xcorr, xcorr_pval, xcorr_lags. "
                                "Run get_cross_corr_matrix_with_pvals first.")

        indl = np.tril_indices(self.xcorr.shape[0], k=-1)
        
        corr_vals = self.xcorr[indl]
        pval_vals = self.xcorr_pval[indl]
        lag_vals = self.xcorr_lags[indl]
        
        return corr_vals, pval_vals, lag_vals

    def get_lower_pvals(self):
        """
        Extract the lower-triangle values of the p-value matrix.
        Returns
        -------
        p_lower : ndarray
            Non-duplicate (lower-triangle, excluding diagonal) values of
            `self.pvals`.
        """
        # get the bottom triangle of the matrix to avoid duplicates
        indl = np.tril_indices(self.pvals.shape[0],k=-1)
        self.p_lower = self.pvals[indl]
        return self.p_lower

    def get_lower(self):
        """
        Extract the lower-triangle values of the correlation matrix.
        Returns
        -------
        corr_lower : ndarray
            Non-duplicate (lower-triangle, excluding diagonal) values of
            `self.corr`.
        """
        # get the bottom triangle of the matrix to avoid duplicates
        indl = np.tril_indices(self.corr.shape[0], k=-1)  # Get lower triangle indices excluding diagonal
        self.corr_lower = self.corr[indl]
        return self.corr_lower

    def plot_corr_matrix(self, vmin=-0.4, vmax=0.4, title: str = None, cmap="seismic"):
        """
        Plot the correlation matrix as a masked heatmap.

        Parameters
        ----------
        vmin : float
            Minimum r value for the colorbar.
        vmax : float
            Maximum r value for the colorbar.
        title : str, optional
            Plot title (default None).
        cmap : str, optional
            Colormap name (default "seismic").
        """
        mask = np.triu(np.ones_like(self.corr, dtype=bool))
        f, ax = plt.subplots()
        sb.heatmap(self.corr, mask=mask, cmap=cmap, vmin=vmin, vmax=vmax, center=0, square=True, cbar_kws={"shrink": .5})
        if title is not None:
            plt.title(title)
        plot = plt.show()
        return plot

    def get_clim(self):
        """
        Compute the min/max correlation values for setting a color scale.
        Returns
        -------
        tuple of float
            (vmin, vmax): minimum value of the full correlation matrix and
            maximum value of its lower triangle.
        """
        self.vmin = self.corr.min()
        lower = self.get_lower()
        self.vmax = lower.max()
        return (self.vmin, self.vmax)

def corr_matrix(dffs_1, dffs_2,method='spearman'):
    """
    Compute the pairwise correlation matrix between two sets of cell traces.

    Parameters
    ----------
    dffs_1 : ndarray, shape (N, T)
        First set of traces, cells by time.
    dffs_2 : ndarray, shape (M, T)
        Second set of traces, cells by time.
    method : {'spearman', 'kendall', 'jaccard'}, optional
        Correlation method: Spearman's rho, Kendall's tau-b, or Jaccard
        similarity of binarized (>0) traces (default 'spearman'). Cell pairs
        with all-zero or zero-std traces are assigned a dummy value of 999.

    Returns
    -------
    corr_r : ndarray, shape (N, M)
        Correlation (or similarity) coefficient for each pair of cells
        between `dffs_1` and `dffs_2`.
    corr_p : ndarray, shape (N, M)
        P-values for each pair of cells. Only returned when `method` is not
        'jaccard'.
    """
    # if method == 'spearman':
    #     print('using spearman rho')
    # elif method == 'kendall':
    #     print('using kendall tau')

    corr_r = np.empty([dffs_1.shape[0], dffs_2.shape[0]])
    corr_p = np.empty([dffs_1.shape[0], dffs_2.shape[0]])

    for i in np.arange(0, dffs_1.shape[0]):
        dff_1 = dffs_1[i]
        for j in np.arange(0, dffs_2.shape[0]):
            dff_2 = dffs_2[j]
            if not ((dff_1.any()) & dff_2.any()) or np.std(dff_1) == 0 or np.std(dff_2) == 0:
                corr_r[i, j] = 999
                corr_p[i, j] = 999
            else:
                if method=='spearman':
                    corr_r[i, j] = stats.spearmanr(dff_1, dff_2)[0]
                    corr_p[i, j] =  stats.spearmanr(dff_1,dff_2)[1]
                elif method=='kendall':
                    corr_r[i,j] = stats.kendalltau(dff_1,dff_2)[0]
                    corr_p[i,j] =  stats.kendalltau(dff_1,dff_2)[1]
                elif method=='jaccard':
                    # Compute intersection and union
                    X = (dff_1 > 0).astype(int)
                    Y = (dff_2 > 0).astype(int)
                    intersection = np.sum((X == 1) & (Y == 1))
                    union = np.sum((X == 1) | (Y == 1))
                    if union == 0:
                        corr_r[i,j]=999 #dummy value
                    else:
                        corr_r[i,j] =  intersection / union
    if method=='jaccard':
        return corr_r
    else:
        return corr_r,corr_p