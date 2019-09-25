import numpy as np # linear algebra
import pandas as pd 
%matplotlib inline 
import matplotlib.pyplot as plt
import os
from keras import layers
from keras.models import Model
from keras.models import load_model
from keras import callbacks
import os
import cv2
import string
import numpy as np

#Init main values
symbols = string.ascii_lowercase + "0123456789" # All symbols captcha can contain
num_symbols = len(symbols)
img_shape = (50, 200, 1)

n_samples = len(os.listdir('../training_data'))
X = np.zeros((n_samples, 50, 200, 1)) #1070*50*200
y = np.zeros((5, n_samples, num_symbols)) #5*1070*36

for i, pic in enumerate(os.listdir('../training_data')):
    # Read image as grayscale
    img = cv2.imread(os.path.join('../training_data', pic), cv2.IMREAD_GRAYSCALE)
    pic_target = pic[:-4]
    if len(pic_target) < 6:
        # Scale and reshape image
        img = img / 255.0
        img = np.reshape(img, (50, 200, 1))
        # Define targets and code them using OneHotEncoding
        targs = np.zeros((5, num_symbols))
        for j, l in enumerate(pic_target):
            ind = symbols.find(l)
            targs[j, ind] = 1
        X[i] = img
        y[:, i] = targs
