import os
import pandas as pd
import keras
import numpy as np
from tool import *
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from keras import Input
from keras.applications import Xception, InceptionV3
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.layers import Dense, Dropout, concatenate, maximum, Lambda, Flatten, BatchNormalization
from keras.models import Model, load_model
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import keras.backend as K
from tripletimage import triplet_train_gen,triplet_valid_gen,triplet_valid_gen1
# from keras.utils import plot_model
from keras.utils.vis_utils import plot_model
import random

BASE_DIR='./data/train'
IMG_ROW=512
IMG_COL=256
labelpath='./data/5005.csv'
traindata=pd.read_csv(labelpath)
modelfn=InceptionV3(include_top=False,
                    weights=None)
pathdata=traindata['Image']
labeldata=traindata['Id']
train_pathdata,valid_pathdata,train_labeldata,valid_labeldata=train_test_split(pathdata,labeldata,test_size=0.1)

from keras.utils.vis_utils import plot_model
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        rotation_range=25,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

batch_size = 7






validdata = triplet_valid_gen(BASE_DIR,IMG_ROW,IMG_COL,valid_pathdata,valid_labeldata,batch_size)
early_stopping = EarlyStopping(monitor='val_loss', patience=3)
# auto_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=0, mode='auto', epsilon=0.0001,
#                             cooldown=0, min_lr=0)
auto_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=0, mode='auto', min_delta=0.0001,
                            cooldown=0, min_lr=0)
save_model = ModelCheckpoint('roc_0.5marix_sigmoid_xception{epoch:02d}-{val_out_acc:.2f}.h5', period=1)
roc = RocAuc(validation_positive=triplet_valid_gen1(BASE_DIR,IMG_ROW,IMG_COL,valid_pathdata,valid_labeldata,batch_size=batch_size))
if os.path.exists('roc_0.5marix_sigmoid_xception07-0.92.h5'):
    model = load_model('roc_0.5marix_sigmoid_xception07-0.92.h5', custom_objects={'triplet_loss': triplet_loss})
else:
    # create the base pre-trained model
    input_tensor = Input(shape=(256, 512, 3))   
    base_model1 = Xception(include_top=True, weights='imagenet', input_tensor=None, input_shape=None)  
    base_model1 = Model(inputs=[base_model1.input], outputs=[base_model1.get_layer('avg_pool').output], name='xception')

    base_model2 = InceptionV3(include_top=True, weights='imagenet', input_tensor=None, input_shape=None)
    base_model2 = Model(inputs=[base_model2.input], outputs=[base_model2.get_layer('avg_pool').output],
                        name='inceptionv3')

    img1 = Input(shape=(256, 512, 3), name='img_1')
    # SEXception = build_SEXception()
    # feature = SEXception(img1)
    feature1 = base_model1(img1)
    # feature2 = base_model2(img1)

    BatchNor = BatchNormalization(name='BatchNormalization')(feature1)
    # let's add a fully-connected layer
    category_predict1 = Dense(5004, activation=None, name='ctg_out')(
        Dropout(0.5)(
            BatchNor
        )
    )
    category_predict11 = Lambda(sigmoid_cat, name='out')(category_predict1)



    # concat = concatenate([feature1, feature2])
    # concat = Dropout(0.5)(concat)
    # BatchNor1 = BatchNormalization(name='BatchNormalization1')(concat)
    # category_predict = Dense(500, activation=None, name='ctg_out1')(
    #     BatchNor1   #keras.layers.merge.Concatenate(axis=-1),Concatenate该层接收一个列表的同shape张量，并返回它们的按照给定轴相接构成的向量。
    # )
    # category_predict33 = Lambda(sigmoid_cat, name='out1')(category_predict)

    model = Model(inputs=[img1], outputs=[BatchNor,category_predict11])

    for layer in model.layers:
        print(layer.output_shape)
    model.summary()
    # model.save('dog_xception.h5')
    # plot_model(model, to_file='single_model.png')
    # first: train only the top layers (which were randomly initialized)
    # i.e. freeze all convolutional InceptionV3 layers
    for layer in base_model1.layers:
        layer.trainable = False

    for layer in base_model2.layers:
        layer.trainable = False

    # compile the model (should be done *after* setting layers to non-trainable)
    model.compile(optimizer='nadam',
                  loss={
                        'BatchNormalization': triplet_loss,
                        'out': 'categorical_crossentropy'


                       },
                  metrics=['accuracy'])

    model.fit_generator(triplet_train_gen(BASE_DIR,IMG_ROW,IMG_COL,train_pathdata,train_labeldata,batch_size=batch_size),
                        steps_per_epoch=69897 // batch_size + 1,
                        epochs=30,
                        validation_data=validdata,
                        validation_steps=7766// batch_size + 1,
                        callbacks=[early_stopping, auto_lr,save_model,roc])

    model.save('5005_xception.h5')


# for i, layer in enumerate(model.layers):
#     print(i, layer.name)
#
# # we chose to train the top 2 inception blocks, i.e. we will freeze
# # the first 172 layers and unfreeze the rest:
# cur_base_model = model.layers[1]
# for layer in cur_base_model.layers[:105]:
#     layer.trainable = False
# for layer in cur_base_model.layers[105:]:
#     layer.trainable = True
#
# # we need to recompile the model for these modifications to take effect
# # we use SGD with a low learning rate
#
# from keras.optimizers import SGD
# model.compile(optimizer=SGD(lr=0.0001, momentum=0.9),
#               loss={
#                   'BatchNormalization': triplet_loss,
#                   'out': 'categorical_crossentropy'
#
#               },
#               metrics=['accuracy'])
#
# # we train our model again (this time fine-tuning the top 2 inception blocks
# # alongside the top Dense layers
# save_model = ModelCheckpoint('roc_0.5marix_softmax_xception-tuned-{epoch:02d}-{val_out_acc:.2f}.h5', period=1)
# model.fit_generator(triplet_train_gen(BASE_DIR,IMG_ROW,IMG_COL,train_pathdata,train_labeldata,batch_size=batch_size),
#                         steps_per_epoch=69897 // batch_size + 1,
#                         epochs=30,
#                         validation_data=validdata,
#                         validation_steps=7766// batch_size + 1,
#                         callbacks=[early_stopping, auto_lr,save_model,roc])
# model.save('roc_0.5marix_softmax__xception_tuned.h5')