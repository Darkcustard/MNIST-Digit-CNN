from keras.models import Sequential, load_model
from keras.layers import *
from keras import metrics
from keras.datasets import mnist

import numpy as np

def encodeNum(x):
    
    template = [0,0,0,0,0,0,0,0,0]
    template[x-1] = 1
    return template
    
def decodeNum(x):
    
    return x.index(max(x))+1
    
def loadDataset():
    print('loading dataset...')
    (X, Y), (X_test, Y_test) = mnist.load_data()
    
    Y_encoded = []
    
    #encoding 
    for unencoded_label in Y:
        encodedNum = encodeNum(unencoded_label)
        Y_encoded.append(encodedNum)
        
    return X, np.array(Y_encoded)

def buildModel():
    #define model
    print('building model...')
    
    classifier = Sequential()

    #input
    classifier.add(Input(shape=(28,28,1)))

    #hidden
    classifier.add(Conv2D(64,(6,6),padding="same"))
    classifier.add(MaxPooling2D(pool_size=(3,3),padding="same"))

    classifier.add(Conv2D(48,(5,5),padding="same"))
    classifier.add(MaxPooling2D(pool_size=(2,2),padding="same"))

    classifier.add(Conv2D(32,(4,4),padding="same"))
    classifier.add(MaxPooling2D(pool_size=(2,2),padding="same"))

    classifier.add(Flatten())

    classifier.add(Dense(200,activation="relu"))
    classifier.add(Dense(150,activation="relu"))
    classifier.add(Dense(100,activation="relu"))
    classifier.add(Dense(68,activation="relu"))
    classifier.add(Dense(32,activation="relu"))
    classifier.add(Dense(15,activation="relu"))

    #output
    classifier.add(Dense(9, activation="softmax"))
    classifier.summary()
    
    #compile
    classifier.compile(loss = 'binary_crossentropy', run_eagerly=True ,optimizer = 'adam', metrics=[metrics.CategoricalAccuracy()])
    
    return classifier

images, labels  = loadDataset()
classifier = buildModel()
classifier.fit(x=images,y=labels,batch_size = 100,epochs=10,verbose="auto",shuffle=True)
classifier.save('classifier.ai')