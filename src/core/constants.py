import os

sample_rate = 40
frame_length = sample_rate
segment_length = 400

accelerometer_data = os.path.join('data','all_accelerometer_data_pids_13.csv')
pids_path = os.path.join('data', 'phone_types.csv')
output_path = os.path.join('cache', 'predictions.csv')
features_path = os.path.join('cache', 'X')
labels_path = os.path.join('cache', 'y')