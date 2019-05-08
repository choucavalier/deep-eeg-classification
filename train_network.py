import os
import numpy as np
import torch
import torch.cuda
import torch.nn
import torch.optim
import torch.utils.data
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.base import TransformerMixin
from sklearn.model_selection import train_test_split

torch.set_default_tensor_type('torch.cuda.FloatTensor')
torch.cuda.set_device(1)

data_dir = '/storage/inria/viovene/preprocessed_data'
mkpath = lambda path: os.path.join(data_dir, path)
x_train = np.load(mkpath('x_train_raw.npy')).astype(np.float32)
x_test = np.load(mkpath('x_test_raw.npy')).astype(np.float32)
y_train = np.load(mkpath('y_train.npy')).astype(np.int64).flatten()
y_test = np.load(mkpath('y_test.npy')).astype(np.int64).flatten()
x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train)
x_train = x_train * 1e6
x_valid = x_valid * 1e6
x_test = x_test * 1e6
x_train = x_train[:, :, :, np.newaxis]
x_valid = x_valid[:, :, :368, np.newaxis]
x_test = x_test[:, :, :368, np.newaxis]


class Expression(torch.nn.Module):
  def __init__(self, fun, *args):
    super().__init__(*args)
    self.fun = fun

  def forward(self, x):
    return self.fun(x)


model = torch.nn.Sequential(
  Expression(lambda x: x.permute(0, 3, 2, 1)),
  torch.nn.Conv2d(1, 24, (56, 1), stride=1),
  torch.nn.Conv2d(24, 73, (1, 21), stride=1, bias=False),
  torch.nn.BatchNorm2d(73, momentum=0.1, affine=True),
  torch.nn.MaxPool2d(kernel_size=(84, 1), stride=(3, 1)),
  torch.nn.Dropout(0.328794),
  torch.nn.Conv2d(73, 2, (22, 1), bias=True),
  torch.nn.Softmax(dim=1),
  Expression(lambda x: x[:, :, 0, 0]),
)
model.cuda()

learning_rate = 3e-4
loss_fn = torch.nn.NLLLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


def get_batch(batch_size=512, seq_len=368):
  idxs = np.random.randint(0, x_train.shape[0], size=batch_size)
  sidxs = np.random.randint(0, x_train.shape[2] - seq_len, size=batch_size)
  x_batch = torch.from_numpy(
    np.concatenate([
      x_train[idxs[i], :, sidxs[i]:(sidxs[i] + seq_len)][np.newaxis]
      for i in range(batch_size)
    ],
                   axis=0)
  ).cuda()
  return (x_batch, torch.from_numpy(y_train[idxs]).cuda())


def get_model_accuracy(x, y):
  yh = torch.max(model(torch.from_numpy(x).cuda()), 1)[1].cpu().numpy()
  return accuracy_score(y, yh)


while True:
  model.train()
  for i in range(100):
    x_batch, y_batch = get_batch()
    yh_batch = model(x_batch)
    loss = loss_fn(yh_batch, y_batch)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
  model.eval()
  train_accuracy = get_model_accuracy(x_train[:, :, :368, :], y_train)
  valid_accuracy = get_model_accuracy(x_valid, y_valid)
  test_accuracy = get_model_accuracy(x_test, y_test)
  print('accuracy report')
  print(f'\ttrain\t{train_accuracy}')
  print(f'\tvalid\t{valid_accuracy}')
  print(f'\ttest\t{test_accuracy}')
