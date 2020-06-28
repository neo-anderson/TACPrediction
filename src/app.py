from pipeline.import_data import pids_tacs_and_acc
import pandas as pd
import numpy as np
from pipeline.pipeline import feature_extraction_pipeline, transform_labels
import time
from pipeline.transform import extract_segments_with_resample
from model.neuralnetwork import get_predictions
import os
from core.constants import features_path, labels_path, output_path
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("-p", "--preprocess",
                    action="store_true", default=False,
                    help="Preprocess data instead of loading from cache")
args = parser.parse_args()
start = time.time()

if args.preprocess:
    print("Importing datasets")
    pids, all_tacs, acc_df = pids_tacs_and_acc()
    print("Segmenting and resampling signals")
    features = pd.concat(pids.pid.apply(extract_segments_with_resample, args=(acc_df, all_tacs)).to_list())
    print("Extracting features")
    pipeline = feature_extraction_pipeline()
    X = pipeline.fit_transform(features.copy())
    y = transform_labels(features.copy())
    print("Caching transformed features and labels")
    np.save(features_path, X)
    np.save(labels_path, y)
else:
    X = np.load(features_path+'.npy')
    y = np.load(labels_path+'.npy')

print("Training the neural network")
y_pred = get_predictions(X, y)
pd.DataFrame(y_pred).to_csv(output_path)
print(f"Saved predictions to {output_path}")
print(f"Total time taken: {np.round(time.time() - start, 3)} seconds")
