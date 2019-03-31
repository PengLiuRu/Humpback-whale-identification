import keras
import os
import numpy as np

import operator
import tensorflow as tf
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from keras import Input, initializers
from keras.applications import Xception, InceptionV3, ResNet50
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.layers import Dense, Dropout, Flatten, Conv2D, Lambda, GlobalAveragePooling2D, Activation, Reshape, multiply, AveragePooling2D, BatchNormalization, Permute
from keras.models import Model, load_model
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.vis_utils import plot_model
import keras.backend as K
import random
import pandas as pd
from sklearn.model_selection import train_test_split
from tripletimage import triplet_train_gen,triplet_valid_gen1
class RocAuc(keras.callbacks.Callback):
    def __init__(self,validation_positive,interval=1):
        self.interval=interval
        # self.validation_generate=validation_generate
        self.validation_positive = validation_positive
    def on_epoch_end(self,epoch, logs={}):
    # 每次epoch,读取一批生成的数据
    #     x_val,y_val=next(self.validation_positive)

        x_val,y_val2 = next(self.validation_positive)
        # x_val2, y_val2 = next(self.validation_positive)
        # print('y_val:',y_val)
        if epoch % self.interval == 0:
            try:
                y_pred1,y_pred=self.model.predict(x_val,verbose=0)
                # print('y_pred1:', y_pred1.shape)
                # print('y_pred:',y_pred)
                # y_pred2 = self.model.predict(x_val2, verbose=0)
                best_t, map5score=mapk(y_val2,y_pred)
                # best_t1, map5score1 = mapk(y_val2, y2)
                # best_t2, map5score2 = mapk(y_val2, y_pred2)
                # best_t = (best_t+best_t2)/2
                # map5score = (map5score+map5score2)/2
                print('\n ROC_AUC - epoch:%d - best_t:%.6f -map5score:%.6f\n' % (epoch + 1,best_t , map5score))
            except:
                print('\n  epoch:%d  only one class!!\n' % (epoch + 1))


def triplet_loss(y_true, y_pred):
    # y_pred = K.l2_normalize(y_pred,axis=1)
    # batch = batch_size

    #print(batch)
    ref1 = y_pred[0:7,:]
    pos1 = y_pred[7:14,:]
    neg1 = y_pred[14:21,:]
    dis_pos = K.sum(K.square(ref1 - pos1), axis=-1, keepdims=True)
    dis_neg = K.sum(K.square(ref1 - neg1), axis=-1, keepdims=True)
    # dis_pos = K.sqrt(dis_pos)
    # dis_neg = K.sqrt(dis_neg)
    # a1 = 17
    a1 = K.constant(0.5)
    d1 = K.maximum(0.0, dis_pos - dis_neg + a1)
    # loss=dis_pos-dis_neg
    # loss = K.log(1 + K.exp(loss))
    return K.mean(d1)
def true_label(b):
    batch_size = b.shape[0]
    predicted = []
    for i in range(batch_size):
        bx = b[i]
        sorted_inds = np.argmax(bx)
        predicted.append(sorted_inds)
    predicted = np.array(predicted)
    # print('true_label:',predicted)
    return predicted
def pred_label(b,K=5):
    predicted = []
    batch_size = b.shape[0]
    predicted = []
    for i in range(batch_size):
        bx = b[i]
        sorted_inds = [i[0] for i in sorted(enumerate(-bx), key=lambda x: x[1])]

        predicted.append(sorted_inds[0:K])
    predicted = np.array(predicted)
    # print('pred_label:',predicted)
    return predicted
def apk(actual, predicted, k=5):
    actual = [int(actual)]
    if len(predicted)>k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i,p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i+1.0)

    if not actual:
        return 0.0
    return score / min(len(actual), k)

def mapk(y_true, y_pred):    #predicted = tensor([[ 0.5029,  0.4975,  0.5001,  ...,  0.5062,  0.5025,  0.5015]])   predicted shape: torch.Size([1, 5005])
    map5s = []
    ts = np.linspace(0.1, 0.9, 9)  # array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    for t in ts:
        y_pred[::1, 0] = t
        # print('t:',t)
        # print('y_true:', y_true)
        # print('y_pred:',y_pred)
        y_true1 = true_label(y_true)
        y_pred1 = pred_label(y_pred, K=5)
        map5_ = np.mean([apk(a,p,5) for a,p in zip(y_true1, y_pred1)])
        map5s.append(map5_)
    map5 = max(map5s)
    print(map5s)
    i_max = map5s.index(map5)
    best_t = ts[i_max]
    return best_t,map5

def cat(out3):
    results = (out3[::2] + out3[1::2])/2
    return results
def sigmoid(input):
    # input = K.dot(input,16)
    # results = input*16
    results = input
    # sigmoid = K.sigmoid(results)
    ones_like = K.ones_like(results[:, :1])*0.5
    out = K.concatenate([ones_like,results],1)
    return out

def l2_norm1(input,axis=1):
    norm = K.l2_normalize(input, axis=1)

    return norm

def l2_norm2(input,axis=-1):
    norm = K.l2_normalize(input, axis=-1)

    return norm

def Mean(input, axis=1):
    mean = K.mean(input,axis=2,keepdims=True)

    return mean
def squeeze(input):
    squeeze = K.squeeze(input,axis=-2)
    return squeeze
def sigmoid_cat(input):
    # input = K.dot(input,16)
    # results = input*16
    # results = input
    sigmoid = K.sigmoid(input)
    # ones_like = K.ones_like(sigmoid[:, :1])*0.5
    ones_like = K.zeros_like(sigmoid[:, :1])*0.0
    out = K.concatenate([ones_like,sigmoid],1)
    return out
def get_results(input):
    results = input*16
    return results

def build_SEXception(input_shape=(256, 512, 3)):
    inputs_dim = Input(input_shape)
    x = Lambda(lambda x: x / 255.0)(inputs_dim)  # 在模型里进行归一化预处理

    x = Xception(include_top=False, weights='imagenet', input_tensor=None, input_shape=None)(x)

    squeeze = GlobalAveragePooling2D()(x)

    excitation = Dense(units=2048 // 16)(squeeze)
    excitation = Activation('relu')(excitation)
    excitation = Dense(units=2048)(excitation)
    excitation = Activation('sigmoid')(excitation)
    excitation = Reshape((1, 1, 2048))(excitation)

    scale = multiply([x, excitation])

    x = AveragePooling2D(pool_size=(8, 16))(scale)
    Reshape1 = Reshape((-1,))(x)
    # dp_1 = Dropout(0.6)(x)
    # fc2 = Dense(out_dims)(dp_1)
    # fc2 = Activation('sigmoid')(fc2)  # 此处注意，为sigmoid函数

    model = Model(inputs=inputs_dim, outputs=Reshape1,name='SE-xception')
    return model