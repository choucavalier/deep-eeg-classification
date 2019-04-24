import os

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, KFold, train_test_split
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline

if __name__ == '__main__':
  data_dir = '/storage/inria/viovene/preprocessed_data'
  x_train = np.load(os.path.join(data_dir, 'x_train.npy'))
  y_train = np.load(os.path.join(data_dir, 'y_train.npy'))
  x_test = np.load(os.path.join(data_dir, 'x_test.npy'))
  y_test = np.load(os.path.join(data_dir, 'y_test.npy'))

  model = RandomForestClassifier(
      n_estimators=500, max_depth=4, random_state=42
  )
  pipe = Pipeline([('rf', model)])
  kf = KFold(n_splits=10, random_state=42)
  scores = cross_val_score(pipe, x_train, y_train, scoring='accuracy', cv=kf)
  print('cross_val_scores', scores)
  print(
      'test score',
      accuracy_score(y_test,
                     model.fit(x_train, y_train).predict(x_test))
  )
