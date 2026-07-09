from timestamps import write_timestamps
import pandas as pd 
import os
from glob import glob
####
'''
Saves timestamp files as npz
'''
####
def main():
    """
    Compute and save timestamps for a hardcoded animal/session's defined TSeries.

    Loads experiment metadata, locates the corresponding TSeries, and calls
    `write_timestamps` to compute and save frame times (no stimulus) as an
    `.npz` file.
    """
    base_dir = '/projectnb/sramirezlab/amonast/Tone2P'
    exp_info = pd.read_csv(glob(f"{base_dir}/*info*.csv")[0])
    metadata_path = os.path.join(base_dir,'xml_metadata')
    animals=['939L']
    for ani in animals:
        fov='FOV1'
        session='Baseline'
        TSeries =  exp_info['TSeries_g'].loc[(exp_info['Animal']==ani)&(exp_info['FOV']==fov)&(exp_info['Session']==session)].values[0]
        voltage_path = os.path.join(base_dir,'GCaMP_raw',ani)
        write_timestamps(ani,fov,session,exp_info,voltage_path,metadata_path,stimulus=False,recall2=False,TSeries=TSeries)

if __name__=='__main__':
    main()
    