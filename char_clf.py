# neural net                         
 

from lasagne import layers                                                                                                                                  
from lasagne.nonlinearities import softmax
from lasagne.updates import nesterov_momentum, sgd
from nolearn.lasagne import NeuralNet

# sklearn models                                                                                                                                                             
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import classification_report
from sklearn.learning_curve import learning_curve
from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import LabelEncoder

# graphs                                                                                                                                                                                             
import matplotlib.pyplot as plt
import seaborn as sns

# images                                                                                                                                                                                             
from scipy.ndimage import convolve, rotate
from skimage.feature import hog
from skimage import draw, data, io, segmentation, color, exposure
from skimage.measure import regionprops
from skimage.filters import threshold_otsu
from skimage.transform import resize
from skimage.transform import warp
from PIL import Image

# basics                                    
import cv2                                                                                                                             
import random
import pickle
import pandas as pd
import numpy as np
import math
import glob

def load_data(size):
    char_files = glob.glob("/users/derekjanni/pyocr/chars/*")
    non_files = glob.glob("/users/derekjanni/pyocr/notChars/*")
    # pre-process character data
    X_char = []
    for i in char_files:
        img = Image.open(i)
        img = img.convert("L")
        img = img.resize((size,size))
        image = np.asarray(img).astype('int32')
        image.setflags(write=True)
        thresh = threshold_otsu(image)
        binary = image > thresh
        X_char.append(binary)
        
    Y_char = [1 for i in range(len(X_char))]
    # pre-process non-char data
    X_non = []
    for i in char_files:
        img = Image.open(i)
        img = img.convert("L")
        img = img.resize((size,size))
        image = np.asarray(img).astype('int32')
        image.setflags(write=True)
        thresh = threshold_otsu(image)
        binary = image > thresh
        X_non.append(binary)
        
    for i in range(1000):
        radius = np.random.randint(10,15)+np.random.randint(5)
        offset = np.random.randint(-5,5)
        img = np.ones((size, size), dtype=np.int32)
        cv2.circle(img,((offset+size)/2,(offset+size)/2), radius, (0,0,0), -1)
        X_non.append(img)

    Y_non = [0 for i in range(len(X_non))]
    X = np.asarray(X_char + X_non).reshape(-1,1,50,50).astype(np.float32)
    Y = np.asarray(Y_char + Y_non).astype(np.int32)
    # some trickery to emulate a train-test split
    indices = list(zip(X, Y)) #name to indicate what's going on here, not the actual data struct 
    random.shuffle(indices)
    X, Y = zip(*indices)
    X = np.asarray(X)
    Y = np.asarray(Y)
    return X, Y
    
X, Y = load_data(50)

# select ~ 4/5 of data for training, 1/5 for test
X_train, X_test = X[:18000], X[18000:]
Y_train, Y_test = Y[:18000], Y[18000:]

print 'Shape'
print X.shape, Y.shape

# define and evaluate model
from lasagne.layers import DenseLayer
from lasagne.layers import InputLayer
from lasagne.layers import DropoutLayer
from lasagne.nonlinearities import softmax, sigmoid
from lasagne.updates import nesterov_momentum, sgd
from nolearn.lasagne import NeuralNet
from lasagne import layers

model = NeuralNet(
    layers=[
        ('input', layers.InputLayer),
        ('conv1', layers.Conv2DLayer),
        ('pool1', layers.MaxPool2DLayer),
#        ('conv2', layers.Conv2DLayer),
#        ('pool2', layers.MaxPool2DLayer),
#        ('conv3', layers.Conv2DLayer),
#        ('pool3', layers.MaxPool2DLayer),
        ('hidden4', layers.DenseLayer),
        ('hidden5', layers.DenseLayer),
        ('output', layers.DenseLayer),
        ],

    input_shape = (None, 1, 50, 50),
    conv1_num_filters=32, conv1_filter_size=(4, 4), pool1_pool_size=(4, 4),
#    conv2_num_filters=64, conv2_filter_size=(4, 4), pool2_pool_size=(4, 4),
#    conv3_num_filters=128, conv3_filter_size=(2, 2), pool3_pool_size=(2, 2),
    hidden4_num_units=250,
    hidden5_num_units=250,
    output_num_units=2, output_nonlinearity=softmax,

    update=nesterov_momentum,
    update_learning_rate=0.01,
    update_momentum=0.9,

    regression=False,
    max_epochs=500,
    verbose=1,
    )

print "Fitting Model..."
model.fit(X_train, Y_train)

Y_pred = model.predict(X_test)
Y_pred = [1 - i for i in Y_pred]  

# print clf report to see how well we did
print classification_report(Y_test, Y_pred)

with open('charclf.pkl', 'w') as outfile:
    pickle.dump(model, outfile)
