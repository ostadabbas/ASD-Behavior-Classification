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
nb_train_samples = 6300
nb_test_samples = 1574

train_data_dir = 'train'
test_data_dir = 'test'

test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
        test_data_dir,
        target_size=(img_height,img_width),
        batch_size=batch_size,
        class_mode='binary',
        color_mode='rgb',
        classes=['Typical','Atypical'],
        shuffle = False)   


def parse_args():
    parser = argparse.ArgumentParser(description='Get Metrics')
    parser.add_argument(
        '--e', '-e', dest='e', required=True,
        help='1-5')
    return parser.parse_args()
    

def getMetrics(model1,model2,model3,model4,model5):
    y_true = test_generator.classes        
    filenames = test_generator.filenames
    nb_samples = len(filenames)
    y_pred_all = np.zeros(len(y_true))
    for model in [model1,model2,model3,model4,model5]:
        y_pred = model.predict_generator(test_generator,steps = nb_samples//32+1)
        y_pred_t=np.zeros(nb_samples)
        for i in range(nb_samples):
            y_pred_t[i]=int(np.round(y_pred[i][0]))
            
        y_pred_all += y_pred_t
        
    for i in range(len(y_pred_all)):
        y_pred_all[i]=int(np.round(y_pred_all[i]/5))
        
    print('Test Accuracy: %.3f'%accuracy_score(y_true,y_pred_all))
    print('Test Precision: %.3f'%precision_score(y_true,y_pred_all))
    print('Test Recall: %.3f'%recall_score(y_true,y_pred_all))
    print('Test AUC: %.3f'%roc_auc_score(y_true,y_pred_all))
    
if __name__=='__main__':
    args = parse_args()
    e=int(args.e)
    print('Modeling metrics after epoch %d'%e)
    model1 = load_model('epoch%d/model1.h5'%e)
    model2 = load_model('epoch%d/model2.h5'%e)
    model3 = load_model('epoch%d/model3.h5'%e)
    model4 = load_model('epoch%d/model4.h5'%e)
    model5 = load_model('epoch%d/model5.h5'%e)
    getMetrics(model1,model2,model3,model4,model5)