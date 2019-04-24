import os
import numpy as np
import torch
import torch.cuda
import torch.nn
import torch.optim
import torch.utils.data
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split

torch.set_default_tensor_type('torch.cuda.FloatTensor')

data_dir = '/storage/inria/viovene/preprocessed_data'
mkpath = lambda path: os.path.join(data_dir, path)
x_train = np.load(mkpath('x_train.npy')).astype(np.float32)
x_test = np.load(mkpath('x_test.npy')).astype(np.float32)
y_train = np.load(mkpath('y_train.npy')).astype(np.int64).flatten()
y_test = np.load(mkpath('y_test.npy')).astype(np.int64).flatten()
x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train)
x_scaler = StandardScaler().fit(x_train)
x_train = x_scaler.transform(x_train)
x_valid = x_scaler.transform(x_valid)
x_test = x_scaler.transform(x_test)

input_shape = x_train.shape[1]
h1_shape = 12
h2_shape = 64
model = torch.nn.Sequential(
  torch.nn.Linear(input_shape, h1_shape),
  torch.nn.ReLU(),
  torch.nn.Linear(h1_shape, h2_shape),
  torch.nn.ReLU(),
  torch.nn.Linear(h2_shape, 2),
  torch.nn.Softmax(dim=1),
)

learning_rate = 1e-4
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


def get_batch():
  batch_size = 64
  idxs = np.random.randint(0, x_train.shape[0], size=batch_size)
  return (
    torch.from_numpy(x_train[idxs]).cuda(),
    torch.from_numpy(y_train[idxs]).cuda(),
  )


def get_model_accuracy(x, y):
  yh = torch.max(model(torch.from_numpy(x).cuda()), 1)[1].cpu().numpy()
  return accuracy_score(y, yh)


converged = False
last_test_error = None
count_increasing_test_error = 0
loss = []
while not converged:
  model.train()
  for i in range(100):
    x_batch, y_batch = get_batch()
    yh_batch = model(x_batch)
    loss = loss_fn(yh_batch, y_batch)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
  model.eval()
  train_accuracy = get_model_accuracy(x_train, y_train)
  valid_accuracy = get_model_accuracy(x_valid, y_valid)
  test_accuracy = get_model_accuracy(x_test, y_test)
  print('accuracy report')
  print(f'\ttrain\t{train_accuracy}')
  print(f'\tvalid\t{valid_accuracy}')
  print(f'\ttest\t{test_accuracy}')
