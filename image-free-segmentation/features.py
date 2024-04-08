import scipy.io as io
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import cv2
import os
import torch

'''卷积过程'''
def conv(pattern,data):
    n,m = data.shape
    k = pattern.shape[1]
    line = []
    for i in range(n//k):
        for j in range(m//k):
            s1 = i*k
            s2 = j*k
            a = data[s1:s1+k,s2:s2+k]
            line.append((np.multiply(pattern,a)))
    return np.array(line)



'''生成图片与pattern卷积的结果'''
def featureimgs(pattern_path,image_path,feature_img_save):

    # 加载pattern
    matr = io.loadmat(pattern_path)
    data = matr['array']
    
    # 归一化工具
    mm = MinMaxScaler()

    # 加载测试图片 
    path_list = os.listdir(image_path)
    for item in path_list:
        name = os.path.splitext(item)[0]
        imgx = cv2.imread(image_path + item,0)
        # imgx = cv2.cvtColor(imgx,cv2.COLOR_BGR2RGB)
        imgx = cv2.resize(imgx, [256, 256])
        imgx = imgx/255.0



        # 卷积过程
        for i in range(len(data)):
            pattern = data[i, :, :, :]
            imgrlt = conv(pattern,imgx)
            for j in range(len(imgrlt)):
                imgout = imgrlt[j, :, :]
                imgout = imgout[0]
                # 卷积后的图片数值限定在0~255.0
                imgout = mm.fit_transform(imgout)
                imgout = imgout * 255.0

                # 保存原图与pattern卷积后的结果
                cv2.imwrite('%s/%s-%s-%s.png' % (feature_img_save, name,i,j), imgout)




'''生成测量值'''
def feature(pattern_path,image_path):

    featurey = []
    featurex = []
    features = []
    # 加载pattern
    matr = io.loadmat(pattern_path)
    
    data = matr['array']
    print(data.shape)
    # 归一化工具
    mm = MinMaxScaler()

    # 加载测试图片 
    imgx = cv2.imread(image_path,0)
    # imgx = cv2.cvtColor(imgx,cv2.COLOR_BGR2GRAY)
    imgx = cv2.resize(imgx, [256, 256])
    imgx = imgx/255.0

    # 卷积过程
    for i in range(len(data)):
        pattern = data[i, :, :, :]
        imgrlt = conv(pattern, imgx)

        for j in range(len(imgrlt)):

            featurex.append(np.sum(imgrlt[j]))
            if len(featurex) == 8:
                featurey.append(featurex)
                featurex = []
                if len(featurey) == 8:
                    features.append(featurey)
                    featurey = []

        #io.savemat(feature_mat_save, {'data':np.array(features)})
    return features
