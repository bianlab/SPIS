from torchvision.utils import save_image
from nets.yolo_simulation import YoloBody
from SPIS_simulation import SPIS_detect
import cv2
import time
import scipy.io
import numpy as np
from PIL import Image
import os
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
from utils.utils import *


#设置默认数据类型，设置显卡序号，设置torch数据类型
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
dtype = 'float32'
torch.set_default_tensor_type(torch.FloatTensor)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


"""重建相关参数"""
# 测试图片
data_path = './test/'

# 模型权重
model_path = 'logs/ep632-loss4.382-val_loss5.624.pth'
classes_path='model_data/voc_classes.txt'
#---------------------------------------------------#
#   获得种类和先验框的数量
#---------------------------------------------------#
class_names, num_classes  = get_classes(classes_path)

#pattern
pattern_path = "./pattern_005.mat"

#measurements save path
featuremap="./features/features.mat"

#Visualized 2D measurements
featureimgs="./featureimgs/"

saveflag=False

# 载入网络
net    = YoloBody(num_classes, phi="s",patchSize=8)
model = net.cuda()
model.load_state_dict(torch.load(model_path, map_location=device))
print('SPIS model, and classes loaded.')





"""重建图像"""
def feature(pattern_path,image_path,featuremap_img,featuremap_mat,saveflag):

    featurey = []
    featurex = []
    featuremat = []

    # 加载pattern
    matr = scipy.io.loadmat(pattern_path)
    data = matr['data']
    #print(data.shape)


    
    # 归一化工具
    mm = MinMaxScaler()

    # 加载测试图片 
    path_list = os.listdir(image_path)
    for item in path_list:
        features = []
        name = os.path.splitext(item)[0]
        imgx = cv2.imread(image_path + item)
        imgx=cv2.cvtColor(imgx,cv2.COLOR_BGR2RGB)
        imgx = cv2.resize(imgx, [256, 256])
        imgx = np.array(imgx/255.)
        imgx = imgx.transpose(2,0,1)
        #print(imgx.shape)

        # 卷积过程
        for i in range(len(data)):
            pattern = data[i, :, :, :]
            #print(pattern.shape)

            imgrlt = conv(pattern,imgx)
            print(imgrlt.shape)
            for j in range(len(imgrlt)):
                imgout = imgrlt[j, :, :]
                imgout = imgout[0]
                featurex.append(np.sum(imgrlt[j]))
                if len(featurex) == 8:
                    featurey.append(featurex)
                    featurex = []
                    if len(featurey) == 8:
                        features.append(featurey)
                        featurey = []
                if saveflag:
                    # 卷积后的图片数值限定在0~255.0
                    imgout = mm.fit_transform(imgout)
                    imgout = imgout * 255.0
                    # 保存原图与pattern卷积后的结果
                    if not os.path.exists(featuremap_img+'/'+name):
                        os.makedirs(featuremap_img+'/'+name)
                    cv2.imwrite('%s/%s/%s-%s.png' % (featuremap_img, name,i,j), imgout)
        featuremat.append(features)
    featuremat=np.array(featuremat)
    featuremat =np.expand_dims(featuremat, axis=1)
    print(featuremat.shape)
    scipy.io.savemat(featuremap_mat, {'data':featuremat})   



def simulate_features(model, image_path):
    SPIS=model
    SPIS.eval()
    i=0
    path_list = os.listdir(data_path)

    for item in path_list:
        img_path=data_path + item
        # 提取featuremap并输入网络进行重建
        feature_map = np.array(feature(pattern_path,img_path))
        print(feature_map.shape)
        scipy.io.savemat("./features/%d.mat"%(i), {'data':feature_map})
        i=i+1
        # 保存重建图片

def extract(model, pattern_path):

    SPIS=model
    SPIS.eval()
    s1 = SPIS.FeatureMap[0].weight.cpu().detach().numpy()
    scipy.io.savemat(pattern_path, {"data": s1})


# '''提取pattern'''
#extract(model, pattern_path)

# '''生成卷积后的结果图'''
# featureimgs(pattern_path,image_path,feature_img_save)

'''生成测量值'''
feature(pattern_path,data_path,featureimgs,featuremap,saveflag)
print('All FeatureMap are saved successfully, the outputs are saved in the featuremap folder.')

# '''仿真'''
#simulate_features(model, data_path)