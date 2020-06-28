from core.constants import accelerometer_data, pids_path
import pandas as pd
import glob
import os

def tac_readings(pid):
    fname = glob.glob(os.path.join('data', 'clean_tac', f'{pid}*.csv'))[0]
    df = pd.read_csv(fname)
    df['pid'] = pid
    return df

def pids_tacs_and_acc():
    # setup acc_df
    acc_df = pd.read_csv(accelerometer_data)
    # sort by time of recording the data
    acc_df.sort_values(by=['time'], inplace=True)
    # create column with timestamp in seconds
    acc_df['timestamp'] = acc_df['time']//1000
    # setup all_tacs
    pids = pd.read_csv(pids_path)
    all_tacs = pd.concat(pids.pid.apply(tac_readings).to_list())
    # sort by timestamp
    all_tacs.sort_values(by=['timestamp'], inplace=True)
    return pids, all_tacs, acc_df