import glob
import os
import argparse
import shutil

import mne
import numpy as np
from mne.datasets import sample
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, KFold, train_test_split
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from joblib import Parallel, delayed

from mne_features.feature_extraction import extract_features, FeatureExtractor

SELECTED_CH_NAMES = np.array([
    'A1', 'A2', 'C3', 'C4', 'CZ', 'F3', 'F4', 'F7', 'F8', 'FP1', 'FP2', 'FZ',
    'O1', 'O2', 'P3', 'P4', 'PZ', 'T3', 'T4', 'T5', 'T6'
])

SFREQ = 100


def preprocess_one_file(path):
  f = mne.io.read_raw_edf(path)
  cleaned_ch_names = np.array([
      c.replace('EEG ', '').replace('-REF', '') for c in f.ch_names
  ])
  ch_idxs = np.array([
      np.where(cleaned_ch_names == ch)[0][0] for ch in SELECTED_CH_NAMES
  ])
  f = f.load_data().copy().resample(SFREQ, npad='auto')
  x = f.get_data()[ch_idxs, :]
  rnd_start_idx = np.random.randint(
      int(60 * SFREQ), int(x.shape[1] - (60 * SFREQ))
  )
  x = x[:, rnd_start_idx:int(rnd_start_idx + 60 * SFREQ)]
  x = x.clip(-800, 800)
  y = np.array([['abnormal' in path]])
  return (path, x, y)


def preprocess_dataset(output_dir):
  data_paths = glob.glob(
      '/storage/inria/viovene/tuh_data/**/*.edf', recursive=True
  )
  np.random.shuffle(data_paths)
  # data = [preprocess_one_file(path) for path in data_paths]
  data = Parallel(n_jobs=30
                  )(delayed(preprocess_one_file)(path) for path in data_paths)
  train, test = train_test_split(data)
  x_train = np.vstack([x[np.newaxis, :, :] for (_, x, _) in train])
  x_test = np.vstack([x[np.newaxis, :, :] for (_, x, _) in test])
  y_train = np.vstack([y for (_, _, y) in train])
  y_test = np.vstack([y for (_, _, y) in test])
  np.save(os.path.join(output_dir, 'x_train_raw.npy'), x_train)
  np.save(os.path.join(output_dir, 'x_test_raw.npy'), x_test)
  funcs = {
      'skewness',
      'kurtosis',
      'mean',
      'variance',
      'std',
      'ptp_amp',
      'hurst_exp',
      'app_entropy',
      'pow_freq_bands',
      'hjorth_complexity',
  }
  params = {
      'pow_freq_bands__freq_bands': np.array([0.5, 4, 8, 13, 30, 49]),
      'pow_freq_bands__ratios': 'all',
      'pow_freq_bands__log': True
  }
  fe = FeatureExtractor(sfreq=SFREQ, selected_funcs=funcs, params=params)
  x_train = fe.fit_transform(x_train)
  x_test = fe.transform(x_test)
  print(x_train.shape, x_test.shape)
  np.save(os.path.join(output_dir, 'x_train.npy'), x_train)
  np.save(os.path.join(output_dir, 'y_train.npy'), y_train)
  np.save(os.path.join(output_dir, 'x_test.npy'), x_test)
  np.save(os.path.join(output_dir, 'y_test.npy'), y_test)


output_dir = '/storage/inria/viovene/preprocessed_data'
preprocess_dataset(output_dir)
