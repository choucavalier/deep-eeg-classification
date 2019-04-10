import glob

import mne
import numpy as np
from mne.datasets import sample
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from mne_features.feature_extraction import extract_features

ch_names = np.array([
    'A1', 'A2', 'C3', 'C4', 'CZ', 'F3', 'F4', 'F7', 'F8', 'FP1', 'FP2', 'FZ',
    'O1', 'O2', 'P3', 'P4', 'PZ', 'T3', 'T4', 'T5', 'T6'
])
data_paths = glob.glob('data/**/*.edf', recursive=True)
sfreqs = []
xs = []
ys = []
for path in data_paths:
    f = mne.io.read_raw_edf(path)
    cleaned_ch_names = np.array(
        [c.replace('EEG ', '').replace('-REF', '') for c in f.ch_names])
    ch_idxs = np.array(
        [np.where(cleaned_ch_names == ch)[0][0] for ch in ch_names])
    print('\t{}'.format(ch_idxs.shape))
    sfreq = f.info['sfreq']
    if sfreq != 250.0:
        continue
    sfreqs.append(sfreq)
    x = f.get_data()
    x = x[ch_idxs, :]
    print('\t{}'.format(sfreq))
    print('\t{}'.format(x.shape))
    rnd_start_idx = np.random.randint(
        int(2 * 60 * sfreq), int(x.shape[1] - (2 * 60 * sfreq)))
    x = x[:, rnd_start_idx:int(rnd_start_idx + 60 * sfreq)]
    print('\t{}'.format(x.shape))
    xs.append(x[np.newaxis, :, :])
    label = 'abnormal' in path
    ys.append(label)
x = np.concatenate(xs, axis=0)
y = np.array(ys)

selected_funcs = {'mean', 'ptp_amp', 'std'}
x = extract_features(x, sfreqs[0], selected_funcs)

pipe = Pipeline([('scaler', StandardScaler()),
                 ('lr', LogisticRegression(random_state=42))])

kf = KFold(n_splits=3, random_state=42)
scores = cross_val_score(pipe, x, y, scoring='accuracy', cv=kf)
