# neural net 
from sklearn.preprocessing import StandardScaler, LabelEncoder

# sklearn metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

# images
from skimage.feature import hog
from skimage import draw, data, io, segmentation, color, exposure
from skimage.measure import regionprops
from skimage.filters import threshold_otsu
from skimage.transform import resize 
from skimage.transform import warp 
from PIL import Image
from scipy.ndimage import convolve, rotate

# basics
import pickle
import pandas as pd
import numpy as np
import math
import glob 

def get_test_img(i, size):
    """
    Returns image from my file directory with corresponding index i
    """
    img = Image.open('/users/derekjanni/pyocr/test/'+ str(i) + '.Bmp')
    img = img.convert("L")
    img = img.resize((50,50))
    image = np.asarray(img)
    image.setflags(write=True)
    thresh = threshold_otsu(image)
    binary = image > thresh
    return binary

def nudge_dataset(X, Y):
    """                                                                                                                                                                        
    This produces a dataset 5 times bigger than the original one,                                                                                                              
    by moving the (size x size) images around by 1px to left, right, down, up                                                                                                  
    """

    direction_vectors = [
        [[0, 1, 0],
         [0, 0, 0],
         [0, 0, 0]],

        [[0, 0, 0],
         [1, 0, 0],
         [0, 0, 0]],

        [[0, 0, 0],
         [0, 0, 1],
         [0, 0, 0]],

        [[0, 0, 0],
         [0, 0, 0],
         [0, 1, 0]]]

    shift = lambda x, w: convolve(x.reshape(50,50), mode='constant',
                                  weights=w)
    #add nudged data                                                                                                                                                           
    X = np.concatenate([X] +
                       [np.apply_along_axis(shift, 1, X, vector)
                        for vector in direction_vectors])
    Y = np.concatenate([Y for _ in range(5)], axis=0)
    return X, Y    

size = 50
LE = LabelEncoder()
X_train, Y_train, X_test, Y_test = None, None, None, None

with open('classifier_data.pkl', 'r') as infile:
    data = pickle.load(infile)
    X_train, Y_train, X_test, Y_test = data[0], data[1], data[2], data[3]

# make sure data is in right format, & properly label-encoded
X_train, X_test = X_train.astype(np.float32), X_test.astype(np.float32)
Y_train = LE.fit_transform(Y_train).astype(np.int32)
Y_test = LE.fit_transform(Y_test).astype(np.int32)

X_train, Y_train = nudge_dataset(X_train, Y_train)


#neural net & params
from lasagne.layers import DenseLayer
from lasagne.layers import InputLayer
from lasagne.layers import DropoutLayer
from lasagne.nonlinearities import softmax, sigmoid, tanh
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet
from lasagne import layers

model = NeuralNet(
    layers=[
        ('input', layers.InputLayer),
        ('conv1', layers.Conv2DLayer),
        ('pool1', layers.MaxPool2DLayer),
        ('conv2', layers.Conv2DLayer),
        ('pool2', layers.MaxPool2DLayer),
        ('conv3', layers.Conv2DLayer),
        ('pool3', layers.MaxPool2DLayer),
        ('hidden4', layers.DenseLayer),
        ('hidden5', layers.DenseLayer),
        ('output', layers.DenseLayer),
        ],

    input_shape=(None, 1, 50, 50),
    conv1_num_filters=32, conv1_filter_size=(3, 3), pool1_pool_size=(2, 2),
    conv2_num_filters=64, conv2_filter_size=(2, 2), pool2_pool_size=(2, 2),
    conv3_num_filters=128, conv3_filter_size=(2, 2), pool3_pool_size=(2, 2),
    hidden4_num_units=1000,
    hidden5_num_units=1000,
    output_num_units=62, 
    output_nonlinearity=softmax,

    update_learning_rate=0.01,
    update_momentum=0.9,

    regression=False,
    max_epochs=250,
    verbose=1,
    )

print 'Fitting...'
model.fit(X_train, Y_train)

print "Predicting..."
Y_pred = model.predict(X_test)
Y_pred = LE.inverse_transform(Y_pred)
Y_test = LE.inverse_transform(Y_test)

y1 = LE.inverse_transform(model.predict(X_train))
y2 = LE.inverse_transform(Y_train)
print classification_report(y2, y1)

print classification_report(Y_test, Y_pred)

with open('character_model.pkl', 'w') as outfile:
    pickle.dump([model, LE], outfile)

'''
X_sub = np.asarray([get_test_img(i, size) for i in range(6284, 12504)])
Y_sub = model.predict(X_sub)

print LE.inverse_transform(Y_sub[:20])
# create submission output
with open('submission.csv', 'w') as outfile:
    outfile.write('ID,Class\n')
    for i in range(len(Y_sub)):
        outfile.write(str(i+6284) + ','+ Y_sub[i] +'\n')'''
print "-------DONE-------"
