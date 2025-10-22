import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing
from tqdm import tqdm


with open("dataset_images_train",'rb') as fo:
    dict=pickle.load(fo,encoding='bytes')

data = dict['data']
#3
#3.1
image1 = dict['data'][0].reshape(3,32,32)
#plt.imshow(image1.transpose((1,2,0)))
# plt.show()

#3.1_2
def afficher(numclasse):
    compt = 0
    for i in range(data.shape[0]):
        if(dict['target'][i]==numclasse):
            plt.imshow(data[i].reshape(3,32,32).transpose((1,2,0)))
            plt.show()
            compt+=1
            if(compt==10):
                return
# afficher(3)

#3.1_3
data_embedded = TSNE(n_components=2).fit_transform(data[:3000])
data_embedded.shape
#for i in range(10):
#    plt.scatter(data_embedded[dict['target'][:3000]==i,0],data_embedded[dict['target'][:3000]==i,1],label=i)

#plt.legend(loc="lower right")
# plt.show()

#3.2
#3.2_1 et 3.2_2 ##### k plus proches voisins ######
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

list_accuracy = []
for k in tqdm(range(1,50,2)):
    knn=KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, Y_train)
    y_pred=knn.predict(X_valid)
    list_accuracy.append((y_pred==Y_valid).sum()/Y_valid.shape[0])

#3.2_3
plt.plot(range(1,50,2),list_accuracy)
plt.xlabel("k")
plt.ylabel("accurary")
plt.show()

#3.2_4
with open("data_images_test",'rb') as fo:
    dict_test = pickle.load(fo,encoding='bytes')

knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(dict['data'],dict['target'])
y_pred = knn.predict(dict_test['data'])

np.savetxt("images_test_predictions.csv", y_pred)