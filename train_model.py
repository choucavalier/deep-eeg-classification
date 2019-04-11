import glob
import os

import mne
import numpy as np
from mne.datasets import sample
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, KFold, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from mne_features.feature_extraction import extract_features


def preprocess_dataset(output_dir):
    ch_names = np.array([
        'A1', 'A2', 'C3', 'C4', 'CZ', 'F3', 'F4', 'F7', 'F8', 'FP1', 'FP2',
        'FZ', 'O1', 'O2', 'P3', 'P4', 'PZ', 'T3', 'T4', 'T5', 'T6'
    ])
    data_paths = glob.glob('/storage/inria/viovene/tuh_data/**/*.edf',
                           recursive=True)
    np.random.shuffle(data_paths)
    train_paths, test_paths = train_test_split(data_paths)
    data = {}
    for dataset_name, data_paths in [('train', train_paths),
                                     ('test', test_paths)]:
        sfreqs = []
        xs = []
        ys = []
        for path in data_paths:
            f = mne.io.read_raw_edf(path)
            cleaned_ch_names = np.array([
                c.replace('EEG ', '').replace('-REF', '') for c in f.ch_names
            ])
            ch_idxs = np.array(
                [np.where(cleaned_ch_names == ch)[0][0] for ch in ch_names])
            sfreq = f.info['sfreq']
            if sfreq != 250.0:
                continue
            sfreqs.append(sfreq)
            x = f.get_data()
            x = x[ch_idxs, :]
            rnd_start_idx = np.random.randint(
                int(2 * 60 * sfreq), int(x.shape[1] - (2 * 60 * sfreq)))
            x = x[:, rnd_start_idx:int(rnd_start_idx + 60 * sfreq)]
            xs.append(x[np.newaxis, :, :])
            label = 'abnormal' in path
            ys.append(label)
        x = np.concatenate(xs, axis=0)
        y = np.array(ys)
        selected_funcs = {'mean', 'ptp_amp', 'std'}
        x = extract_features(x, sfreqs[0], selected_funcs)
        data[dataset_name] = {'x': x, 'y': y}

    for dataset_name in data:
        for k in ['x', 'y']:
            path = os.path.join(output_dir, f'{k}_{dataset_name}.npy')
            np.save(path, data[dataset_name][k])

output_dir = '/storage/inria/viovene/preprocessed_data'

if not os.path.isdir(output_dir):
    preprocess_data(output_dir)

x_train = np.load(os.path.join(output_dir, 'x_train.npy'))
y_train = np.load(os.path.join(output_dir, 'y_train.npy'))
x_test = np.load(os.path.join(output_dir, 'x_test.npy'))
x_test = np.load(os.path.join(output_dir, 'x_test.npy'))

pipe = Pipeline([('scaler', StandardScaler()),
                 ('rf', RandomForestClassifier(n_estimators=100, random_state=42))])
kf = KFold(n_splits=10, random_state=42)
scores = cross_val_score(pipe, x_train, y_train, scoring='accuracy', cv=kf)
print(scores)
