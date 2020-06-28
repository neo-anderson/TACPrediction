import numpy as np
import pandas as pd
import librosa
from core.constants import frame_length
from sklearn.pipeline import Pipeline

class NumpyArrayConvertor():
    def transform(self, X, **fit_params):
        X['x'] = X['x'].apply(np.array)
        X['y'] = X['y'].apply(np.array)
        X['z'] = X['z'].apply(np.array)
        return X

    def fit(self, X, y=None, **fit_params):
        return self

class RMSCalculator():
    def rms(self, x):
        return librosa.feature.rms(x, frame_length=frame_length, hop_length=frame_length, center=False)[0]

    def transform(self, X, **fit_params):
        X['rms_x'] = X['x'].apply(self.rms)
        X['rms_y'] = X['y'].apply(self.rms)
        X['rms_z'] = X['z'].apply(self.rms)
        return X
    
    def fit(self, X, y=None, **fit_params):
        return self

class STFTCalculator():
    def avg_stft_per_frame(self, x):
        stft = librosa.amplitude_to_db(np.abs(librosa.stft(x, n_fft=frame_length, hop_length=frame_length, center=False)), ref=np.max)
        return stft.mean(axis=1)

    def transform(self, X, **fit_params):
        X['stft_x'] = X['x'].map(self.avg_stft_per_frame)
        X['stft_y'] = X['y'].map(self.avg_stft_per_frame)
        X['stft_z'] = X['z'].map(self.avg_stft_per_frame)
        return X

    def fit(self, X, y=None, **fit_params):
        return self

class FeaturesFlattener():
    def __init__(self, feature_cols):
        self.feature_cols = feature_cols

    def transform(self, X, **fit_params):
        return pd.concat([pd.DataFrame(X[c].tolist()) for c in self.feature_cols], axis=1).values

    def fit(self, X, y=None, **fit_params):
        return self

def feature_extraction_pipeline(feature_to_flatten=['rms_x', 'rms_y', 'rms_z', 'stft_x', 'stft_y', 'stft_z']):
    return Pipeline([("Convert signals to numpy arrays", NumpyArrayConvertor()),
                     ("Calculate RMS values for signals", RMSCalculator()),
                     ("Calculate average STFT vectors", STFTCalculator()),
                     ("Flatten features into a 2D numpy array", FeaturesFlattener(feature_to_flatten))
                    ])

def transform_labels(X):
    # y == True if TAC level > 0.8
    return (X.tac > 0.08).values