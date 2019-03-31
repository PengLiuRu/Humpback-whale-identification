import numpy as np
import random
import cv2
from keras.preprocessing.image import ImageDataGenerator
from fectchimage import imgarr,siamese_fetch_img
import os
import pandas as pd
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        rotation_range=25,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)
batch_size = 7
train_negative2 = train_datagen.flow_from_directory(
    './data/keras_train',

    target_size=(256, 512),
    # batch_size=1,
    shuffle=True,
    batch_size=batch_size,
    class_mode='categorical')
validation_negative2 = test_datagen.flow_from_directory(
    './data/keras_valid',

    target_size=(256, 512),
    shuffle=True,
    # batch_size=1,
    batch_size=batch_size,
    class_mode='categorical')

path = './data/5005labels.csv'  #标签
ha = pd.read_csv(open(path))
id = 0
# print(random.randint(0,3))
def triplet_train_gen(BASE_DIR,IMG_ROW,IMG_COL,pathdata,labeldata,batch_size=batch_size):  
    while(True):                                                                  
        anchorlist=[]
        positivelist=[]
        negativelist=[]
        anchorlabel=[]
        positivelabel=[]
        negativelabel=[]
        for i in range(batch_size):
            rndid=random.randint(0,len(pathdata)-1)
            imgpath=BASE_DIR+labeldata[labeldata.index[rndid]]+'\\'+pathdata[pathdata.index[rndid]]
            id = ha[ha['labels'] == labeldata[labeldata.index[rndid]]]['Id'].iloc[0]
            anchor_label = np.zeros((1, 5005))
            anchor_label[0, id] = 1
            anchorlabel.append(anchor_label[0])
            # print('anchor_label:', id)
            # print(imgpath)
            # print(os.path.exists(imgpath))
            anchor=imgarr(imgpath)
            anchor=cv2.resize(anchor,(IMG_ROW,IMG_COL))
            # print('anchor shape:',anchor.shape)
            anchorlist.append(anchor)
            positive,positive_label=siamese_fetch_img(BASE_DIR,labeldata[labeldata.index[rndid]],
                                            labeldata,pathdata,kind=1)
            negative,negative_label=siamese_fetch_img(BASE_DIR,labeldata[labeldata.index[rndid]],
                                   labeldata,pathdata,kind=0)
            positive=cv2.resize(positive,(IMG_ROW,IMG_COL))
            negative=cv2.resize(negative,(IMG_ROW,IMG_COL))
            positivelist.append(positive)
            positivelabel.append(positive_label[0])
            negativelist.append(negative)
            negativelabel.append(negative_label[0])
        # yield ([np.asarray(anchorlist),np.asarray(positivelist),np.asarray(negativelist)],
        #        None)
        x, y = train_negative2.next()
        y = np.zeros((batch_size, 5005))
        y[:, 0] = 1
        anchorlist.extend(positivelist)
        if random.random() < 0.5 and x.shape[0]==batch_size:
            anchorlist.extend(x)
            anchorlabel.extend(positivelabel)
            anchorlabel.extend(y)
        else:
            anchorlist.extend(negativelist)
            anchorlabel.extend(positivelabel)
            anchorlabel.extend(negativelabel)
        # print(np.asarray(anchorlist).shape)
        # print(np.argmax(np.asarray(anchorlabel[0])))
        # print(np.argmax(np.asarray(anchorlabel[1])))
        # print(np.argmax(np.asarray(anchorlabel[2])))
        yield ([np.asarray(anchorlist)],[np.asarray(anchorlabel),np.asarray(anchorlabel)])

def triplet_valid_gen(BASE_DIR,IMG_ROW,IMG_COL,pathdata,labeldata,batch_size=batch_size):
    while (True):
        anchorlist=[]
        positivelist=[]
        negativelist=[]
        anchorlabel = []
        positivelabel = []
        negativelabel = []
        for i in range(batch_size):
            rndid = random.randint(0, len(pathdata) - 1)
            imgpath=BASE_DIR+labeldata[labeldata.index[rndid]]+'\\'+pathdata[pathdata.index[rndid]]
            id = ha[ha['labels'] == labeldata[labeldata.index[rndid]]]['Id'].iloc[0]
            anchor_label = np.zeros((1, 5005))
            anchor_label[0, id] = 1
            anchorlabel.append(anchor_label[0])
            # print('anchor_label:', id)
            # print(imgpath)
            # print(os.path.exists(imgpath))
            anchor=imgarr(imgpath)
            anchor=cv2.resize(anchor,(IMG_ROW,IMG_COL))
            positive,positive_label=siamese_fetch_img(BASE_DIR,labeldata[labeldata.index[rndid]],
                                   labeldata,pathdata,kind=1)
            negative,negative_label=siamese_fetch_img(BASE_DIR,labeldata[labeldata.index[rndid]],
                                       labeldata,pathdata,kind=0)
            positive=cv2.resize(positive,(IMG_ROW,IMG_COL))
            negative=cv2.resize(negative,(IMG_ROW,IMG_COL))
            anchorlist.append(anchor)
            positivelist.append(positive)
            positivelabel.append(positive_label[0])
            negativelist.append(negative)
            negativelabel.append(negative_label[0])
        # x, y = validation_negative2.next()
        # y = np.zeros((1, 501))
        # y[0, 0] = 1
        anchorlist.extend(positivelist)
        # if random.random() < 0.5:
        #     anchorlist.extend(x)
        # else:
        anchorlist.extend(negativelist)
        anchorlabel.extend(positivelabel)
        # if random.random() < 0.5:
        #     anchorlabel.extend(y)
        # else:
        anchorlabel.extend(negativelabel)
        yield ([np.asarray(anchorlist)],[np.asarray(anchorlabel),np.asarray(anchorlabel)])

def triplet_train_gen1(BASE_DIR,IMG_ROW,IMG_COL,pathdata,labeldata,batch_size=batch_size):  
    while(True):                                                                    
        imglist=[]
        for i in range(batch_size):
            rndid=random.randint(0,len(pathdata)-1)
            imgpath=BASE_DIR+pathdata[pathdata.index[rndid]]
            anchor=imgarr(imgpath)
            anchor=cv2.resize(anchor,(IMG_ROW,IMG_COL))
            positive=siamese_fetch_img(BASE_DIR,labeldata[labeldata.index[rndid]],
                                       labeldata,pathdata,kind=1)
            negative=siamese_fetch_img(BASE_DIR,labeldata[labeldata.index[rndid]],
                                       labeldata,pathdata,kind=0)
            positive=cv2.resize(positive,(IMG_ROW,IMG_COL))
            negative=cv2.resize(negative,(IMG_ROW,IMG_COL))
            imglist.append([anchor,positive,negative])
        imglist=np.asarray(imglist)
        yield ([imglist[:,0],imglist[:,1],imglist[:,2]])

def triplet_valid_gen1(BASE_DIR,IMG_ROW,IMG_COL,pathdata,labeldata,batch_size=batch_size):
    while (True):
        anchorlist = []
        positivelist = []
        negativelist = []
        anchorlabel = []
        positivelabel = []
        negativelabel = []
        for i in range(batch_size):
            rndid = random.randint(0, len(pathdata) - 1)
            imgpath = BASE_DIR + labeldata[labeldata.index[rndid]] + '\\' + pathdata[pathdata.index[rndid]]
            id = ha[ha['labels'] == labeldata[labeldata.index[rndid]]]['Id'].iloc[0]
            anchor_label = np.zeros((1, 5005))
            anchor_label[0, id] = 1
            anchorlabel.append(anchor_label[0])
            # print(imgpath)
            # print(os.path.exists(imgpath))
            anchor = imgarr(imgpath)
            anchor = cv2.resize(anchor, (IMG_ROW, IMG_COL))
            positive, positive_label = siamese_fetch_img(BASE_DIR, labeldata[labeldata.index[rndid]],
                                                         labeldata, pathdata, kind=1)
            negative, negative_label = siamese_fetch_img(BASE_DIR, labeldata[labeldata.index[rndid]],
                                                         labeldata, pathdata, kind=0)
            positive = cv2.resize(positive, (IMG_ROW, IMG_COL))
            negative = cv2.resize(negative, (IMG_ROW, IMG_COL))
            anchorlist.append(anchor)
            positivelist.append(positive)
            positivelabel.append(positive_label[0])
            negativelist.append(negative)
            negativelabel.append(negative_label[0])
        x, y = validation_negative2.next()
        y = np.zeros((batch_size, 5005))
        y[0, 0] = 1
        anchorlist.extend(positivelist)

        # anchorlist.extend(x)
        # anchorlabel.extend(positivelabel)
        # anchorlabel.extend(y)
        if x.shape[0] == batch_size:
            anchorlist.extend(x)
            anchorlabel.extend(positivelabel)
            anchorlabel.extend(y)
        else:
            anchorlist.extend(negativelist)
            anchorlabel.extend(positivelabel)
            anchorlabel.extend(negativelabel)

        yield np.asarray(anchorlist), np.asarray(anchorlabel)

