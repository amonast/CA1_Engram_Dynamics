#%%
import numpy as np
from scipy.stats import zscore
import statsmodels.formula.api as smf
import pingouin as pg
import sys
sys.path.extend(['/Users/amonast/Documents/GitHub/Engram_2P/Engram_2P'])
from behavior.running import get_running_epochs
from utilities.animal import animal
import ast
import matplotlib.pyplot as plt
plt.style.use('/Users/amonast/Documents/GitHub/CA1_Engram_Dynamics/figures/paper_style.mplstyle')

import pandas as pd

palette = {'engram':"#F37343", 'nonengram':"#06ABC8"}
base_dir = '/Volumes/AM_SSD1/Spont2P'
file_key = '/Volumes/AM_SSD1/Spont2P/Data_info.csv'
fov_data = {
    "994R": {"FOV1": {"Baseline": [(99, None)], "Post": [(106, None)]}, "FOV2": {"Baseline": [(0, 64)], "Post": [(0, 64)]}},
    "989N": {"FOV1": {"Baseline": [(131, None)], "Post": [(108, None)]}, "FOV2": {"Baseline": [(146, None)], "Post": [(0, 10), (79, None)]}},
    "9972R": {"FOV1": {"Baseline": [(0, 18), (52, 57), (70, 75)], "Post": [(72, None)]}, "FOV2": {"Baseline": [(116, None)], "Post": [(78, None)]}},
    "1912L": {"FOV1": {"Baseline": [(56, None)], "Post": [(0, 46)]}, "FOV2": {"Baseline": [(1, 75)], "Post": [(5, 84)]}},
    "160R": {"FOV1": {"Baseline": [(34, None)], "Post": [(0, 73)]}, "FOV2": {"Baseline": [(89, None)], "Post": [(65, None)]}},
    "492N": {"FOV1": {"Baseline": [(2, 18), (34, 124)], "Post": [(53, 161)]}, "FOV2": {"Baseline": [(0, 104), (121, 145)], "Post": [(7, 110)]}},
    "493R": {"FOV1": {"Baseline": [(4, 30)], "Post": [(41, None)]}},
    "218L": {"FOV1": {"Baseline": [(75, None)], "Post": [(138, None)]}, "FOV2": {"Baseline": [(127, None)], "Post": [(0, 72)]}},
    "217N": {"FOV1": {"Baseline": [(120, None)], "Post": [(0, 17), (98, 125)]}, "FOV2": {"Baseline": [(17, 55)], "Post": [(23, 47)]}},
    "217R": {"FOV1": {"Baseline": [(11, 46), (58, 97)], "Post": [(102, None)]}},
    "146R": {"FOV2": {"Baseline": [(119, 138)], "Post": [(3, 80)]}},
    "034R": {"FOV1": {"Baseline": [(0, 79)], "Post": [(23, 47), (110, None)]}}}
#%%

def load_isort(base_dir, params_dir, ani, fov_key, session):
    paths = [
        f"{base_dir}/Analysis/rastermap_batch/{params_dir}/{ani}_{fov_key}_{session}_isort_.npy",
        f"{base_dir}/Analysis/rastermap_batch/{params_dir}/{ani}_{session}_{fov_key}_isort_.npy",
        f"{base_dir}/Analysis/rastermap_batch/{params_dir}/{ani}_{session}_isort_.npy"
    ]
    
    for path in paths:
        try:
            isort = np.load(path)
            print(f"Loaded: {path}")
            return isort
        except FileNotFoundError:
            continue
#%%     
df_rows = []  # collect rows as dicts
for ani, fovs in fov_data.items():
    for fov_key in ["FOV1", "FOV2"]:
        if fov_key in fovs:
            for session in ["Baseline", "Post"]:
                index_ranges = fovs[fov_key][session]
                mouse = animal(ani,fov_key,file_key=file_key,base_dir = base_dir)
                traces_df = mouse.load_traces(signal='dff',sessions=[session])
                is_engram = [bool(ast.literal_eval(col)[0]) for col in traces_df.columns[1:]]
                
                ### get sorting from rastermap
                params_dir='nPCs10_lag60_loc0.1_bin15'
                isort = load_isort(base_dir,params_dir,ani,fov_key,session)
                sorted_engram = np.array(is_engram)[isort] #engram ids sorted by rastermap
                
                #get sequence participating cells
                in_seq = np.zeros(len(is_engram),dtype=bool)
                for start, end in index_ranges:
                    in_seq[start:end]=True #sequence participating cells from above
                seq_eng = in_seq & sorted_engram
                seq_non = in_seq & ~sorted_engram
                
                #prop of sequence that is engram or non-engram
                prop_engram = seq_eng.sum() / in_seq.sum()
                prop_non = seq_non.sum() / in_seq.sum()
                
                #proportion of engram or non-engram cells that are 'sequencey'
                prop_pop_seq_eng = seq_eng.sum() / sorted_engram.sum()
                prop_pop_seq_non = seq_non.sum() / (~sorted_engram).sum()
                
                df_rows.append({
                        "animal": ani,
                        "fov": fov_key,
                        "session": session,
                        "group":mouse.group,
                        "propSeq_engram": prop_engram,
                        "propSeq_nonengram": prop_non,
                        "propPop_engram":prop_pop_seq_eng,
                        "propPop_nonengram":prop_pop_seq_non
            })
#%%                
df = pd.DataFrame(df_rows)
df['session'] = df['session'].replace({'Baseline': 'D0', 'Post': 'D4'})
df["rep"] = df.groupby(["animal", "fov", "session"]).cumcount()
df_long = pd.wide_to_long(
                            df,
                            stubnames=["propSeq", "propPop"],
                            i=["animal", "fov", "session","rep"],
                            j="cell_type",
                            sep="_",
                            suffix=".*"
                        ).reset_index()  
save_path = '/Users/amonast/BOSTON UNIVERSITY Dropbox/Amy Monasterio/Manuscripts/Engram2P/Figures/RevisionFigures/Round2'
# %%
df_long['ani_fov']=df_long['animal']+'_'+df_long['fov']
fc = df_long.loc[df_long['group']=='FC']#.groupby(['animal','cell_type','session']).mean(['propSeq','propPop']).reset_index()
hc = df_long.loc[df_long['group']=='HC']#.groupby(['animal','cell_type','session']).mean(['propSeq','propPop']).reset_index()
#%%
import seaborn as sb
fig,ax=plt.subplots(figsize=(2,2))
sb.boxplot(data=fc,
           y='propSeq',
           x='session',
           hue_order=['nonengram','engram'],
           hue='cell_type',
           ax=ax,
           width=0.5,
           legend=False,
           palette=palette)

plt.hlines(1.01,.8,1.2,color='k',linewidth=.5)
plt.text(1,1.03,'*',ha='center',size=9)
plt.hlines(1.03,-.2,.2,color='k',linewidth=.5)
plt.text(0,1.05,'*',ha='center',size=9)
plt.ylabel('Proportion of \n  Sequence-Active Cells')
plt.legend(frameon=False,bbox_to_anchor=(1,1))
plt.title('FC')
sb.despine()
plt.savefig(f"{save_path}/Fig3E_propSequece_FC.svg",transparent=True)
#%%
model = smf.mixedlm("propSeq ~ cell_type * session", 
                    data=fc,  
                    groups="animal", 
                    vc_formula={"fov": "0 + C(ani_fov)"})  # random intercepts per fov               
                    
result = model.fit()
print(result.summary())
##%%
pg.pairwise_tests(data=fc,dv='propSeq',within=['session','cell_type'],
                  padjust='fdr_bh',
                  subject='ani_fov')
#%%
fig,ax=plt.subplots(figsize=(2,2))
sb.boxplot(data=hc,
           y='propSeq',
           x='session',
           hue='cell_type',
            hue_order=['nonengram','engram'],
            width=0.5,
           ax=ax,
           legend=False,
           palette=palette)
plt.hlines(1,.8,1.2,color='k',linewidth=.5)
plt.text(1,1,'**',ha='center',size=9)
plt.hlines(1,-.2,.2,color='k',linewidth=.5)
plt.text(0,1,'**',ha='center',size=9)
plt.ylabel('Proportion of \n  Sequence-Active Cells')
plt.title('HC')
sb.despine()
plt.savefig(f"{save_path}/Fig3E_propSequece_HC.svg",transparent=True)

#%%
# model = smf.mixedlm("propSeq ~ cell_type * session", 
#                     data=hc,  
#                     groups="animal", 
#                     vc_formula={"fov": "0 + C(ani_fov)"})  # random intercepts per fov               
                    
# result = model.fit()

# print(result.summary())
##%%
pg.pairwise_tests(data=hc,dv='propSeq',within=['session','cell_type'],
                  padjust='fdr_bh',
                  subject='ani_fov')
#%%
fig,ax=plt.subplots(figsize=(2,2))
sb.boxplot(data=fc,
           y='propPop',
           x='session',
           hue='cell_type',
           ax=ax,
           palette=palette)
plt.legend(frameon=False,bbox_to_anchor=(1,1))
plt.ylabel('Proportion of Population \n  Recruited to Sequences')
plt.tight_layout()
plt.title('FC')
sb.despine()
#plt.savefig()