from core.constants import segment_length
import librosa
from scipy.interpolate import interp1d
import numpy as np

def extract_segments_with_resample(pid, acc_df, all_tacs):
    # filter accelerometer readings for the pid
    acc_pid = acc_df[acc_df.pid == pid]
    # get TAC readings for the pid
    tacs_pid = all_tacs[all_tacs.pid == pid]
    # interpolation function for TAC readings of pid
    expected_tac = interp1d(tacs_pid.timestamp, tacs_pid.TAC_Reading, kind='linear')
    # get the start and end time for the TAC readings in seconds
    start, end = tacs_pid.timestamp.min(), tacs_pid.timestamp.max()
    # extract accelerator readings in the time frame TAC has been active
    acc_pid = acc_pid[(start <= acc_pid.timestamp) & (acc_pid.timestamp <= end)]
    acc_pid = acc_pid.sort_values(by=['time'])
    # 10 seconds grouping window
    acc_pid['ten_seconds_group'] = acc_pid.timestamp//10
    # select groups to resample
    samples_in_groups = acc_pid.ten_seconds_group.value_counts()
    valid_groups = samples_in_groups[(samples_in_groups > 375) & (samples_in_groups < 425)].index
    valid_groups_acc_pid = acc_pid[acc_pid['ten_seconds_group'].isin(valid_groups)]
    grouped = valid_groups_acc_pid.groupby('ten_seconds_group').agg(lambda s: s.to_list())
    grouped = grouped[['x', 'y', 'z', 'timestamp']]
    grouped['x'] = grouped['x'].apply(lambda x: librosa.resample(np.array(x), len(x), segment_length))
    grouped['y'] = grouped['y'].apply(lambda x: librosa.resample(np.array(x), len(x), segment_length))
    grouped['z'] = grouped['z'].apply(lambda x: librosa.resample(np.array(x), len(x), segment_length))
    grouped['timestamp'] = grouped.timestamp.map(lambda x: x[0])
    grouped['tac'] = grouped.timestamp.map(lambda x: expected_tac(x))
    grouped['pid'] = pid
    return grouped