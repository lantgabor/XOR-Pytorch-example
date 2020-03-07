# -*- coding: utf-8 -*-

from __future__ import print_function
import torch
torch.manual_seed(2)

import numpy as np

nb_samples = 200
valid_split = 0.2
test_split = 0.1

# Commented out IPython magic to ensure Python compatibility.
samples = np.zeros(nb_samples, dtype=[('input', float, 2), ('output', float, 1)])

for i in range(0, nb_samples, 4):
  noise = np.random.normal(0,1,8)
  samples[i] = (-2+noise[0],-2+noise[1]), 0
  samples[i+1] = (2+noise[2],-2+noise[3]), 1
  samples[i+2] = (-2+noise[4],2+noise[5]), 1
  samples[i+3] = (2+noise[6],2+noise[7]), 0

import matplotlib.pyplot as plt
# %matplotlib inline

fig1 = plt.figure()
plt.scatter(samples['input'][:,0],samples['input'][:,1],c=samples['output'][:], cmap=plt.cm.cool)

samples_train = samples[0:int(nb_samples*(1-valid_split-test_split))]
samples_valid = samples[int(nb_samples*(1-valid_split-test_split)):int(nb_samples*(1-test_split))]
samples_test  = samples[int(nb_samples*(1-test_split)):]
  
from sklearn import preprocessing
# standardizálás
scaler = preprocessing.StandardScaler().fit(samples_train['input'])
samples_train['input'] = scaler.transform(samples_train['input'])
samples_valid['input'] = scaler.transform(samples_valid['input'])
samples_test['input'] = scaler.transform(samples_test['input'])

print(samples_test)

from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class XOR(nn.Module):
  def __init__(self, input_dim = 2, output_dim = 1):
    super(XOR, self).__init__()
    self.lin1 = nn.Linear(input_dim, 10)
    self.lin2 = nn.Linear(10, output_dim)

  def forward(self, x):
    x = self.lin1(x)
    x = F.tanh(x)
    x = self.lin2(x)
    x = F.tanh(x)
    return x

model = XOR()
print(model)

def weights_init(model):
  for m in model.modules():
    if isinstance(m, nn.Linear):
      m.weight.data.normal_(0,1)


weights_init(model)

loss_func = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.02, momentum = 0.9)

epochs = 2000

X = torch.from_numpy(samples_train['input']).float().cuda()
Y = torch.from_numpy(samples_train['output']).view(-1,1).float().cuda()

model.cuda()

steps = X.size(0)
for i in range(epochs):
    for j in range(steps):
        data_point = np.random.randint(X.size(0))
        x_var = Variable(X[data_point], requires_grad=False)
        y_var = Variable(Y[data_point], requires_grad=False)
        
        optimizer.zero_grad()
        y_hat = model(x_var)
        loss = loss_func.forward(y_hat, y_var)
        loss.backward()
        optimizer.step()
        
    if i % 50 == 0:
        print("Epoch: {0}, Loss: {1}, ".format(i, loss.data.cpu().detach().numpy()))

model = model.eval()

preds = model(torch.from_numpy(samples_test['input']).float().cuda())

preds = preds.cpu().detach().numpy()

fig1=plt.figure()
plt.scatter(samples_test['input'][:,0], \
            samples_test['input'][:,1], \
            c=np.round(preds[:,0]), cmap=plt.cm.cool)

from sklearn.metrics import mean_squared_error
test_mse = mean_squared_error(samples_test['output'],preds)
print("Test MSE: %f" % (test_mse))

