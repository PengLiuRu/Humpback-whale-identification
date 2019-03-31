import numpy as np
import pandas as pd
import cv2
import random
import os
import matplotlib.pyplot as plt
path = './data/5005labels.csv'
ha = pd.read_csv(open(path))
id = 0
def imgarr(imgpath):
    img=cv2.imread(imgpath)
    return img

def siamese_fetch_img(BASE_DIR,label,labeldata,pathdata,kind):
    if(kind==0):
        pathdata1=pathdata[labeldata!=label]
        # print('pathdata1:',pathdata1)

    else:
        pathdata1=pathdata[labeldata==label]
        # print('pathdata1:', pathdata1)
    rndid=random.randint(0,len(pathdata1)-1)
    path=BASE_DIR+labeldata[pathdata1.index[rndid]]+'\\'+pathdata1[pathdata1.index[rndid]]
    id = ha[ha['labels'] == labeldata[pathdata1.index[rndid]]]['Id'].iloc[0]
    label = np.zeros((1, 5005))
    label[0, id] = 1
    # print('po ne label:',id)
    # print(path)
    # print(os.path.exists(path))
    img=imgarr(path)
    return img,label

