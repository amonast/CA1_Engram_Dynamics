#%%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import pingouin as pg
import statsmodels.formula.api as smf
from glob import glob
import numpy as np
#%%
def load_pairwise_dfs(binsize):
    path = '/Users/amonast/Desktop/Tone2P/Analysis/correlation/pairwise_correlation_new/overlap/cstrace_total'
    animals = ['997B','639N','M1N','M2L','M5L','194L',
            'F5L','F7N','M8BL2','M9BR2','939L']
    pattern = f'{path}/*_{binsize}_cstrace_beh_matched.csv'
    files = glob(pattern)
    #for f in files: print(f)
    rows = []
    for f in files:
        ani_data =pd.read_csv(f,index_col=0)
        rows.append(ani_data)
    return pd.concat(rows)
#%%
palette1 = {'EN': '#FFC40C', 'NN': '#00ABC8', 'EE': '#F37243'}
#df_ = load_pairwise_dfs(8)
# %%# %%
def plot_means(binsize):
    df = load_pairwise_dfs(binsize)
    palette1 = {'EN': '#FFC40C', 'NN': '#00ABC8', 'EE': '#F37243'}

    df_cs_bl = df.loc[df['period'].isin(['baseline', 'cs1'])].copy()
    df_cs_bl = df_cs_bl[df_cs_bl != 999].dropna()
    df_cs_bl = df_cs_bl[df_cs_bl['pvals']<0.05]
    # Average cs1 across trials per animal × pair group
    mean_ani = (df_cs_bl
        .groupby(['animal','group','pair_type','period'])['corr']
        .mean()
        .reset_index())
    #
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 4))
    sb.pointplot(data=mean_ani,
        x='period', y='corr',
        hue='pair_type',
        palette=palette1,
        dodge=.1,
        capsize=0.05,
        errorbar='se',
        ax=axes[0])
    sb.barplot(data=mean_ani,
        x='period', y='corr',
        hue='pair_type',
        palette=palette1,
        dodge=True,
        capsize=0.05,
        errorbar='se',
        ax=axes[1])
    sb.stripplot(data=mean_ani,
        x='period', y='corr',
        hue='pair_type',
        palette=palette1,
        dodge=True,
        ax=axes[1],
        alpha=0.5,
        size=5,
        jitter=.1,
        edgecolor='k',
        linewidth=1,
        legend=False)
    axes[1].get_legend().remove()
    fig.suptitle('bin size:'+str(binsize))
    
    fig.tight_layout()
    
for b in [4,8,16,25,50,80,100,125]:
    plot_means(b)
#%%
## ---- Change in correlation ---- 
def delta_corr(binsize, pval=None):
    """
    Match valid (non-999) pairs across baseline and cs1, compute delta_corr and R_ratio.
    If pval is given, only keep pairs where at least one period has pvals < pval.
    Returns per-animal means and full matched long-form dataframe.
    """
    df_ = load_pairwise_dfs(binsize)
    df_ = df_[df_ != 999].dropna()

    df_baseline = df_.loc[df_['period'] == 'baseline'].copy()
    df_cs1 = df_.loc[df_['period'] == 'cs_trace'].copy()

    df_baseline['pair_idx'] = df_baseline.groupby(['animal', 'pair_type']).cumcount()
    df_cs1['pair_idx'] = df_cs1.groupby(['animal', 'pair_type']).cumcount()

    df_baseline['pair_key'] = (
        df_baseline['animal'] + '_' +
        df_baseline['pair_type'] + '_' +
        df_baseline['pair_idx'].astype(str))
    
    df_cs1['pair_key'] = (
        df_cs1['animal'] + '_' +
        df_cs1['pair_type'] + '_' +
        df_cs1['pair_idx'].astype(str))

    common_keys = set(df_baseline['pair_key']).intersection(df_cs1['pair_key'])
    df1 = df_baseline[df_baseline['pair_key'].isin(common_keys)].copy()
    df2 = df_cs1[df_cs1['pair_key'].isin(common_keys)].copy()

    if pval is not None:
        sig_keys = (
            set(df1.loc[df1['pvals'] < pval, 'pair_key']) |
            set(df2.loc[df2['pvals'] < pval, 'pair_key'])
        )
        df1 = df1[df1['pair_key'].isin(sig_keys)]
        df2 = df2[df2['pair_key'].isin(sig_keys)]

    df_merged = pd.merge(
        df1[['pair_key', 'corr', 'group', 'animal', 'pair_type']],
        df2[['pair_key', 'corr']],
        on='pair_key',
        suffixes=('_baseline', '_cs1')
    )

    df_merged['delta_corr'] = df_merged['corr_cs1'] - df_merged['corr_baseline']
    df_merged['R_ratio'] = df_merged['corr_cs1'] / df_merged['corr_baseline']
    df_merged['R_ratio_abs'] = np.abs(df_merged['R_ratio'])
    df_merged['delta_corr_abs'] = np.abs(df_merged['delta_corr'])
    df_merged['delta_corr_sign'] = np.sign(df_merged['delta_corr'])
    df_merged['bin_size'] = np.round(binsize / 30,2)

    df_grouped = (
        df_merged
        .groupby(['group', 'animal', 'pair_type'])[['delta_corr', 'R_ratio']]
        .mean()
        .reset_index()
    )
    df_grouped['bin_size'] = np.round(binsize / 30,2)
    df_grouped['n_pairs'] = (
        df_merged.groupby(['animal', 'pair_type'])['pair_key'].count().reset_index()['pair_key'].values
    )
    return df_grouped, df_merged

def delta_corr_z(binsize, pval=None):
    """
    Match valid (non-999) pairs across baseline and cs1, compute delta_corr and R_ratio.
    If pval is given, only keep pairs where at least one period has pvals < pval.
    Returns per-animal means and full matched long-form dataframe.
    """
    df_ = load_pairwise_dfs(binsize)
    df_ = df_[df_ != 999].dropna()

    df_baseline = df_.loc[df_['period'] == 'baseline'].copy()
    df_cs1 = df_.loc[df_['period'] == 'cs_trace'].copy()

    df_baseline['pair_idx'] = df_baseline.groupby(['animal', 'pair_type']).cumcount()
    df_cs1['pair_idx'] = df_cs1.groupby(['animal', 'pair_type']).cumcount()

    df_baseline['pair_key'] = (
        df_baseline['animal'] + '_' +
        df_baseline['pair_type'] + '_' +
        df_baseline['pair_idx'].astype(str))
    
    df_cs1['pair_key'] = (
        df_cs1['animal'] + '_' +
        df_cs1['pair_type'] + '_' +
        df_cs1['pair_idx'].astype(str))

    common_keys = set(df_baseline['pair_key']).intersection(df_cs1['pair_key'])
    df1 = df_baseline[df_baseline['pair_key'].isin(common_keys)].copy()
    df2 = df_cs1[df_cs1['pair_key'].isin(common_keys)].copy()

    if pval is not None:
        sig_keys = (
            set(df1.loc[df1['pvals'] < pval, 'pair_key']) |
            set(df2.loc[df2['pvals'] < pval, 'pair_key'])
        )
        df1 = df1[df1['pair_key'].isin(sig_keys)]
        df2 = df2[df2['pair_key'].isin(sig_keys)]

    df_merged = pd.merge(
        df1[['pair_key', 'corr', 'group', 'animal', 'pair_type']],
        df2[['pair_key', 'corr']],
        on='pair_key',
        suffixes=('_baseline', '_cs1')
    )

    # Z-score using pooled distribution (baseline + cs1) within animal x pair_type
    pooled = df_merged.melt(id_vars=['animal', 'pair_type'], value_vars=['corr_baseline', 'corr_cs1'], value_name='corr_pooled')
    pooled_stats = pooled.groupby(['animal', 'pair_type'])['corr_pooled'].agg(['mean', 'std']).reset_index()
    df_merged = df_merged.merge(pooled_stats, on=['animal', 'pair_type'])
    for col in ['corr_baseline', 'corr_cs1']:
        df_merged[col + '_z'] = (df_merged[col] - df_merged['mean']) / df_merged['std']
    df_merged = df_merged.drop(columns=['mean', 'std'])

    df_merged['delta_corr'] = df_merged['corr_cs1'] - df_merged['corr_baseline']
    df_merged['delta_corr_z'] = df_merged['corr_cs1_z'] - df_merged['corr_baseline_z']

    df_merged['R_ratio'] = df_merged['corr_cs1'] / df_merged['corr_baseline']
    df_merged['R_ratio_abs'] = np.abs(df_merged['R_ratio'])
    df_merged['delta_corr_abs'] = np.abs(df_merged['delta_corr'])
    df_merged['delta_corr_sign'] = np.sign(df_merged['delta_corr'])
    df_merged['bin_size'] = np.round(binsize / 30,2)

    df_grouped = (
        df_merged
        .groupby(['group', 'animal', 'pair_type'])[['delta_corr', 'R_ratio','delta_corr_z']]
        .mean()
        .reset_index()
    )
    df_grouped['bin_size'] = np.round(binsize / 30,2)
    df_grouped['n_pairs'] = (
        df_merged.groupby(['animal', 'pair_type'])['pair_key'].count().reset_index()['pair_key'].values
    )
    return df_grouped, df_merged


def delta_corr_across_bins(binsizes=[4, 8, 16, 25, 50, 80, 100, 125],
                           thr = None):
    '''
    Does not Z score the correlations before calculating the Change.
    '''

    ani_rows, long_rows = [], []
    for b in binsizes:
        ani, long = delta_corr(b) if thr is None else delta_corr(b,thr) 
        ani_rows.append(ani)
        long_rows.append(long)
    return pd.concat(ani_rows, ignore_index=True), pd.concat(long_rows, ignore_index=True)

def delta_corr_across_bins_Z_first(binsizes=[4, 8, 16, 25, 50, 80, 100, 125],
                           thr = None):
    '''
    Zscore the correlations before calculating the Change.
    '''

    ani_rows, long_rows = [], []
    for b in binsizes:
        ani, long = delta_corr_z(b) if thr is None else delta_corr_z(b,thr) 
        ani_rows.append(ani)
        long_rows.append(long)
    return pd.concat(ani_rows, ignore_index=True), pd.concat(long_rows, ignore_index=True)

def plot_delta_corr_bins(data, 
                    dv='delta_corr',
                    ani_mean=True,
                    title=None,
                    ylabel='Delta Corr'):
    if ani_mean:
        mean_df = (data
            .groupby(['group', 'animal', 'pair_type','bin_size'])[['delta_corr', 'R_ratio','R_ratio_abs',
                                                                   'delta_corr_abs','delta_corr_norm']]
            .mean()
            .reset_index())
        plot_data=mean_df
    else:
        plot_data=data
        
    g = sb.catplot(
        data=plot_data,
        hue='pair_type',
        x='bin_size',
        y=dv,
        kind='point',
        capsize=0.1,
        alpha=0.8,
        palette=palette1,
        markers='o',
        linewidth=1,
        markersize=6,
        join=True,
        height=3.5, aspect=1,
        dodge=0,
        legend=True,
        errorbar='ci',
        err_kws=dict(lw=2)
    )
    for ax in g.axes.flatten():
        # ax.axhline(0, color='k', linestyle='--', linewidth=1.5, alpha=0.5)
        ax.set_xlabel('Bin size (s)')
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        
#%%
def plot_delta_corr_by_sign(data,
                            dv='delta_corr',
                            ani_mean=True,
                            title=None,
                            ylabel='Delta Corr'):
    sign_labels = {1.0: 'Increased', -1.0: 'Decreased'}
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5), sharey=False)

    for ax, sign in zip(axes, [1.0, -1.0]):
        df_sign = data[data['delta_corr_sign'] == sign]
        if ani_mean:
            df_sign = (df_sign
                .groupby(['group', 'animal', 'pair_type', 'bin_size'])[[dv]]
                .mean()
                .reset_index())
        sb.pointplot(
            data=df_sign,
            hue='pair_type',
            x='bin_size',
            y=dv,
            capsize=0.1,
            alpha=0.8,
            palette=palette1,
            markers='o',
            linestyles='-',
            dodge=0,
            legend=(sign == 1.0),
            errorbar='se',
            err_kws=dict(lw=2),
            ax=ax)
        #ax.axhline(0, color='k', linestyle='--', linewidth=1.5, alpha=0.5)
        ax.set_xlabel('Bin size (s)')
        ax.set_ylabel(ylabel if sign == 1.0 else '')
        ax.set_title(sign_labels[sign])
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')

    if title:
        fig.suptitle(title)
    fig.tight_layout()

def fit_lme(data, formula, group_col='animal', animal_mean=False):
    """
    Fit a linear mixed effects model.

    Parameters
    ----------
    data : DataFrame
    formula : str
        e.g. 'delta_corr ~ C(pair_type, Treatment("NN")) * bin_size'
    group_col : str
        Column to use as random effect grouping variable (default 'animal')
    animal_mean : bool
        If True, average all rows within each animal x predictor combination
        before fitting (one datapoint per animal per condition)
    """
    df = data.copy()
    if animal_mean:
        # derive groupby cols from all columns except the dependent variable
        dv = formula.split('~')[0].strip()
        group_cols = [c for c in df.columns if c != dv]
        df = df.groupby(group_cols, observed=True)[dv].mean().reset_index()
    results = smf.mixedlm(formula, df, groups=df[group_col]).fit()
    print(results.summary())
    return results


def main():
    _, delta_all = delta_corr_across_bins(
                            binsizes=[4, 8, 16, 25, 50, 80, 100, 125],
                            thr=None)
    delta_all['delta_corr_norm'] = delta_all.groupby(['animal', 'bin_size'])['delta_corr'].transform(
        lambda x: (x - x.mean()) / x.std())
    
    delta_all.to_csv('/Users/amonast/Documents/GitHub/Amy_Reviews/data/delta_corr_recall1_cstrace_behmatch_thr_overlap.csv')
    
    
    _, delta_all_z = delta_corr_across_bins_Z_first(
                            binsizes=[4, 8, 16, 25, 50, 80, 100, 125],
                            thr=None)
    
    delta_all_z.to_csv('/Users/amonast/Documents/GitHub/Amy_Reviews/data/delta_corr_Z_recall1_cstrace_behmatch_thr_overlap.csv')
    
    return 

if __name__=='__main__':
    main()
# %%
# ## proportion of the cell pairs:
# # %%
# binsizes = delta_all['bin_size'].unique()
# fig, axes = plt.subplots(1, len(binsizes), figsize=(4*len(binsizes), 4), sharey=True)

# for ax, bs in zip(axes, sorted(binsizes)):
#     df_bs = delta_all[delta_all['bin_size'] == bs]
#     counts = df_bs.groupby(['animal', 'pair_type']).size().unstack(fill_value=0)
#     props = counts.div(counts.sum(axis=1), axis=0)
#     props = props[['EE', 'EN', 'NN']]  # order
#     props.plot(kind='bar', stacked=True, ax=ax,
#                color=[palette1[p] for p in props.columns],
#                legend=(ax == axes[-1]), width=0.8)
#     ax.set_title(f'{bs:.2f}s')
#     ax.set_xlabel('')
#     ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')

# axes[0].set_ylabel('Proportion of pairs')
# fig.tight_layout()