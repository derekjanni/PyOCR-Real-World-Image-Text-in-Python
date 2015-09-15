import sys
import numpy as np
import cv2
from PIL import Image
from skimage.filters import threshold_otsu
import pickle
from collections import OrderedDict 
from didYouMean import didYouMean
from ggplot import *

def process_img(img):
    """ Helper function that processes sub-image from source"""
    img = Image.fromarray(img)
    img = img.convert("L")
    img = img.resize((50,50))
    gray = np.asarray(img)
    gray.setflags(write=True)
    thresh = threshold_otsu(gray)
    binary = gray > thresh
    binary = np.asarray(binary.astype(np.float32))
    binary = binary.reshape(-1, 1, 50, 50)
    return binary

char2string_data = None
model = None
LE = None
print 'Loading Models'

# Character Model
with open('character_model.pkl', 'r') as infile:
    model, LE = pickle.load(infile)

fname = sys.argv[1]
im = cv2.imread(fname)
ret,im = cv2.threshold(im,127,255,cv2.THRESH_BINARY)

gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY) # grayscale                            
_,thresh = cv2.threshold(gray,150,255,cv2.THRESH_BINARY_INV) # threshold                                                                                                     
kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
dilated = cv2.dilate(thresh,kernel,iterations = 5) # dilate     

#################      Now finding Contours         ###################                                                                                                                          
contours,hierarchy = cv2.findContours(dilated,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE) # see docs, CV_RETR_LIST

print 'Processing Images'
images_to_predict ={}
hierarchy = [tuple(i) for i in hierarchy[0]]
dictionary = dict(zip(hierarchy, contours))
for key in dictionary:
    if cv2.contourArea(dictionary[key])> 100:
        [x,y,w,h] = cv2.boundingRect(dictionary[key])
        if  w > 20 and h > 20 and key[0] > 0:
            # made to block out small contours which could be mistaken for images
            try:
                subimg = process_img(im[y-15:y+h+15, x-15:x+w+15])
                #if char_clf_model.predict(subimg)[0]==0:
                images_to_predict[x,y]= subimg
            except: 
                # The only exception that happens is when a character is too close (within 5px) of edge
                # We'll assume these characters are no good anyway
                pass
            
images_to_predict = OrderedDict(sorted(images_to_predict.items(), key=lambda x: x[0][0], reverse=False))

for image in images_to_predict.values():
    cv2.imshow('image',image[0][0])
    key = cv2.waitKey(0)

print "Generating Output:"
rawstring = ''
for image in images_to_predict.values():
    rawstring += str(LE.inverse_transform(model.predict(image)[0])) # predict each image

print rawstring

print "Improved Guess:"

print didYouMean.didYouMean(rawstring)
