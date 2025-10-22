import pickle
import numpy as np
import torch as th
from tqdm import tqdm
import torch.optim as optim
from torch.nn import functional as F
from sklearn import preprocessing
import torch.nn as nn
from torchvision import transforms

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

X_train = th.from_numpy(X_train).float()
X_valid = th.from_numpy(X_valid).float()
Y_train = th.from_numpy(Y_train).long()
Y_valid = th.from_numpy(Y_valid).long()

X_train = X_train.reshape(X_train.shape[0],3,32,32)
X_valid = X_valid.reshape(X_valid.shape[0], 3, 32, 32)

np.random.seed(0)
th.manual_seed(0)
th.cuda.manual_seed(0)

d = X.shape[1]
k = 10

transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

trainloader = th.utils.data.DataLoader(th.utils.data.TensorDataset(X_train, Y_train), batch_size=64, shuffle=True)
valloader = th.utils.data.DataLoader(th.utils.data.TensorDataset(X_valid, Y_valid), batch_size=64, shuffle=True)

def prediction(f):
    return th.argmax(f, 1)

def error_rate(y_pred,y):
    return ((y_pred != y).sum().float())/y_pred.size()[0]

class CNN(th.nn.Module):

    def _get_flattened_size(self, input_shape):
        size_input = th.zeros(1, *input_shape)
        output = self.pool(F.relu(self.conv3(self.pool(F.relu(self.conv2(self.pool(F.relu(self.conv1(size_input)))))))))
        return output.view(1, -1).size(1)

    # Constructeur qui initialise le modèle
    def __init__(self,d,k,h1,h2,dropout):
        super(CNN, self).__init__()

        self.dropout = dropout

        self.conv1 = nn.Conv2d(3, 32, 3)
        self.conv2 = nn.Conv2d(32, 32, 3)
        self.conv3 = nn.Conv2d(32, 32, 3)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        flattened_size = self._get_flattened_size((3, 32, 32))

        self.layer1 = th.nn.Linear(flattened_size, h1)
        self.layer2 = th.nn.Linear(h1, h2)
        self.layer3 = th.nn.Linear(h2, k)

        self.layer1.reset_parameters()
        self.layer2.reset_parameters()
        self.layer3.reset_parameters()

    # Implémentation de la passe forward du modèle
    def forward(self, x):

        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        

        x = F.relu(self.layer1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = F.relu(self.layer2(x))
        x = F.dropout(x, p=self.dropout, training=self.training)

        return th.softmax(self.layer3(x),1)

d = 3072
k = 10

nnet = CNN(d,k,200,100,0.5)
device = "cpu"
nnet = nnet.to(device)
criterion = th.nn.CrossEntropyLoss()
eta = 0.001
optimizer = optim.Adam(nnet.parameters(), lr=eta)

error_test = np.nan

nb_epochs = 50
pbar = tqdm(range(nb_epochs))
for epoch in pbar:
    cpt_batch = 0
    for images, labels in trainloader:

        # envoi ds données sur le device
        images = images.to(device)
        labels = labels.to(device)
        
        # Remise à zéro des gradients
        optimizer.zero_grad()

        f_train = nnet(images)

        loss = criterion(f_train,labels)

        ###Ajout d'une régularization L2
        l2_lambda = 0.1
        l2_reg = 0

        for param in nnet.parameters():
            l2_reg += th.norm(param)

        loss += l2_lambda * l2_reg

        # Calculs des gradients
        loss.backward()

        # Mise à jour des poids du modèle avec l'optimiseur choisi et en fonction des gradients calculés
        optimizer.step()

        cpt_batch += 1

        y_pred_train = prediction(f_train)
        error_train = error_rate(y_pred_train, labels)


        pbar.set_postfix(iter=epoch, idx_batch = cpt_batch, loss=loss.item(), error_train=error_train.item(), error_test = error_test, l2_reg=l2_reg)


    #test sur l'ensemble de validation à la fin de chaque epoch
    error_avg = 0
    all_count   = 0

    for images, labels in valloader:
        
        images = images.to(device)
        labels = labels.to(device)
        
        f_test = nnet(images)

        # Affichage des 5 premières images avec les prédictions associées
        if(all_count == 0):
            probas = f_test.cpu().detach().numpy()

        y_pred_test = prediction(f_test)
        error_avg += error_rate(y_pred_test, labels)
        all_count += 1

    error_test = (error_avg/all_count).item()

    pbar.set_postfix(iter=epoch, idx_batch=cpt_batch, loss=loss.item(), error_train=error_train, error_test = error_test)

with open("data_images_test", 'rb') as fo:
    dict_test = pickle.load(fo, encoding='bytes')

X_test=th.from_numpy(dict_test["data"]).float()
X_test = X_test.view(-1, 3, 32, 32)
f_test=nnet(X_test)
y_pred=prediction(f_test)
np.savetxt("images_test_predictions.csv", y_pred)