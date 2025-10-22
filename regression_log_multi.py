import pickle
import numpy as np
import torch as th
from tqdm import tqdm
import torch.optim as optim
from sklearn import preprocessing
from torch.nn import functional as F


with open("dataset_images_train",'rb') as fo:
    dict=pickle.load(fo,encoding='bytes')

data = dict['data']

X = data
Y = dict['target']
indices = np.random.permutation(X.shape[0])

training_idx = indices[:int(X.shape[0]*0.6)]
valid_idx = indices[int(X.shape[0]*0.6):]

X_train = X[training_idx,:]
Y_train = Y[training_idx]

X_valid = X[valid_idx,:]
Y_valid = Y[valid_idx]

X_train=preprocessing.normalize(X_train)
X_valid=preprocessing.normalize(X_valid)

np.random.seed(0)
th.manual_seed(0)
th.cuda.manual_seed(0)

d = X.shape[1]
k = 10

def prediction(f):
    return th.argmax(f, 1)

def error_rate(y_pred,y):
    return ((y_pred != y).sum().float())/y_pred.size()[0]

###### Régression logistique multivariée ######

class Reg_log_multi(th.nn.Module):

    def __init__(self,d,k):
        super(Reg_log_multi, self).__init__()

        self.layer = th.nn.Linear(d,k)
        self.layer.reset_parameters()

    def forward(self, x):
        out = self.layer(x)
        return F.softmax(out,1)


model = Reg_log_multi(d,k)
device = "cpu"
model = model.to(device)

X_train = th.from_numpy(X_train).float().to(device)
y_train = th.from_numpy(Y_train).long().to(device)

X_test = th.from_numpy(X_valid).float().to(device)
y_test = th.from_numpy(Y_valid).long().to(device)

eta = 0.001

criterion = th.nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(), lr=eta)

nb_epochs = 10000
pbar = tqdm(range(nb_epochs))

for i in pbar:
    optimizer.zero_grad()

    f_train = model(X_train)
    loss = criterion(f_train,y_train)

    loss.backward()
    optimizer.step()

    if (i % 1000 == 0):

        y_pred_train = prediction(f_train)

        error_train = error_rate(y_pred_train,y_train)
        loss = criterion(f_train,y_train)

        f_test = model(X_test)
        y_pred_test = prediction(f_test)

        error_test = error_rate(y_pred_test, y_test)

        pbar.set_postfix(iter=i, loss = loss.item(), error_train=error_train.item(), error_test=error_test.item())


with open("data_images_test", 'rb') as fo:
    dict_test = pickle.load(fo, encoding='bytes')

X_test = th.from_numpy(dict_test['data']).float()
X_test=preprocessing.normalize(X_test)
X_test = th.from_numpy(X_test).float().to(device)

f_test = model(X_test)
y_pred = prediction(f_test)

np.savetxt("images_test_predictions.csv", y_pred)

