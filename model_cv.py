# import required packages and global variables
import sys
import os.path as osp
import numpy as np
import math
import glob
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
import numpy as np
import argparse
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.layers import Dense, Activation, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam, SGD
from keras.models import load_model,Model,Sequential
from keras.applications.vgg16 import VGG16
from sklearn.metrics import accuracy_score,precision_score,roc_auc_score,recall_score

img_height = 256  # frame height in pixel
img_width = 256  # frame width in pixel
batch_size = 32

epochs = 5
train_data_dir = 'train'
test_data_dir = 'test'

nb_batch_sample = 1260

def parse_args():
    parser = argparse.ArgumentParser(description='Run all videos')
    parser.add_argument(
        '--f', '-f', dest='f', required=True,
        help='1-5')
    return parser.parse_args()
    

def CNNModel():
    classification_model = Sequential()
    input_shape = (256,256,3)
    classification_model.add(Conv2D(64, (3, 3), input_shape=input_shape))
    classification_model.add(Activation('relu'))
    classification_model.add(MaxPooling2D(pool_size=(2, 2)))

    classification_model.add(Flatten())
    classification_model.add(Dense(64))
    classification_model.add(Activation('relu'))
    classification_model.add(Dense(1))
    classification_model.add(Activation('sigmoid'))
    return classification_model



if __name__=='__main__':
    args = parse_args()
    fold=int(args.f)
    dirs = [('k1', 'k2', 'k3', 'k4', 'k5'), ('k2', 'k3', 'k4', 'k5', 'k1'), ('k1', 'k3', 'k4', 'k5', 'k2'),('k1','k2','k3','k5', 'k4'), ('k1', 'k2','k4','k5', 'k3')]
    fold_datagen = ImageDataGenerator(rescale=1./255,horizontal_flip=True)
    model = CNNModel()
    opt = Adam()
    model.compile(loss='binary_crossentropy',optimizer=opt,metrics=['acc'])
    for i in range(epochs):
        directory = dirs[fold-1]
        print(directory)
        generators = []
        for j in range(5):
            generators.append(fold_datagen.flow_from_directory(
                directory[j],
                target_size=(img_height,img_width),
                batch_size=batch_size,
                class_mode='binary',
                color_mode='rgb',
                classes=['Typical','Atypical']))
        for j in range(4):
            model.fit_generator(
                    generators[j],
                    steps_per_epoch=nb_batch_sample//batch_size,
                    epochs=1,
                    validation_data=generators[-1],
                    validation_steps=nb_batch_sample//batch_size)     
    
        model.save('epoch%d/model%d.h5'%((i+1),fold))
    print('Finished Training')