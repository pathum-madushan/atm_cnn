from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
from keras.callbacks import TensorBoard
from keras.utils import plot_model
import tensorflow as tf
import keras
import json
import time
import sys
import os

from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import glob
import cv2

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# dimensions of our images
img_width, img_height = 150, 150

if K.image_data_format() == 'channels_first':
    input_shape = (1, img_width, img_height)
else:
    input_shape = (img_width, img_height, 1)

names = {
    0 : 'allowed',
    1 : 'rejected',
    
}


#_dir =  sys.argv[1]
_dir = "C:/final_project/testing/allowed/"


# load the model we saved
# model = load_model('./models/1516228873.3_fixedepos_model.h5')
# infile = open('./models/1516262902.05_last-1_convnet_model.json')
# model = keras.models.model_from_json(json.load(infile))
#input
model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3), input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.load_weights('./models/save_model7.h5')

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])


dir_files = glob.glob(_dir+'*.png')

for _file in dir_files:
    file_path = _file
    

    img = image.load_img(file_path, target_size=(img_width, img_height), grayscale=True)
    
    x = image.img_to_array(img)
    #x = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)
    x = np.expand_dims(x, axis=0)

    # score = model.evaluate(x, x, batch_size=32)

    images = np.vstack([x])

    classes = model.predict(images)
    
    p_classes = model.predict_classes(images)
    print (names[p_classes[0][0]])
    #print (_file+" : "+names[p_classes[0]])


# # predicting images








# # predicting multiple images at once
# img = image.load_img('./dataset/testing/angelina_jolie/AAJ142_cropped_ori resized.jpg', target_size=(img_width, img_height))
# y = image.img_to_array(img)
# y = np.expand_dims(y, axis=0)

# # pass the list of multiple images np.vstack()
# images = np.vstack([x, y])
# classes = model.predict_classes(images, batch_size=32)
# print(classes)

# # print the classes, the images belong to
# print classes
# print classes[0]
# print classes[0][0]
