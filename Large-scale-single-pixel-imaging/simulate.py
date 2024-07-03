from torchvision.utils import save_image
from net.UDLSSPI1k_step2 import *
from utils import *
import cv2
from torch.autograd import Variable
import scipy.io
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import cv2

"""设置默认数据类型,设置torch数据类型"""
dtype = 'float32'
torch.set_default_tensor_type(torch.FloatTensor)
device = 'cuda'

"""重建图像"""
def featuremap(pattern_path,image_path,featuremap_img,featuremap_mat,saveflag):

    featurey = []
    featurex = []
    featuremat = []

    # 加载pattern
    matr = scipy.io.loadmat(pattern_path)
    data = matr['data']


    
    # 归一化工具
    mm = MinMaxScaler()

    # 加载测试图片 
    path_list = os.listdir(image_path)
    for item in path_list:
        features = []
        name = os.path.splitext(item)[0]
        imgx = cv2.imread(image_path + item)
        imgx = cv2.cvtColor(imgx,cv2.COLOR_BGR2RGB)
        imgx = cv2.resize(imgx, [1024, 1024])
        #imgx = imgx/255.0

        # 拆分为RGB三个通道
        r, g, b = cv2.split(imgx)

        # 卷积过程
        for i in range(len(data)):
            pattern = data[i, :, :, :]
            patternr = pattern[0, :,:]
            patterng = pattern[1, :,:]
            patternb = pattern[2, :,:]
            b1 = conv(patternb, b)
            g1 = conv(patterng, g)
            r1 = conv(patternr, r)
            imgrlt = (r1 + g1 + b1)/255.0
            for j in range(len(imgrlt)):
                imgout = imgrlt[j, :, :]
                #imgout = imgout[0]
                featurex.append(np.sum(imgrlt[j]))
                if len(featurex) == 32:
                    featurey.append(featurex)
                    featurex = []
                    if len(featurey) == 32:
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
    print(np.array(featuremat).shape)
    scipy.io.savemat(featuremap_mat, {'data':np.array(featuremat)})   

"""提取特征图相关参数"""

# 测试图片
image_path = './test/'

# FeatureMap_img
featuremap_img = './featuremap'

# FeatureMap_mat
featuremap_mat = './features/features.mat'

# pattern路径
pattern_path = "./pattern/pattern_1k.mat"

# 模型权重
model_path = './weights/UDLSSPI1k_step2.pth'

# step1权重
path_step1 = './weights/UDLSSPI1k_step1.pth'

# 是否保存featuremap
saveflag = False


# 载入网络
net = LSSPI_two(path=path_step1)
model = net.cuda()
model.load_state_dict(torch.load(model_path, map_location = device))
print('load succesful')
print('Sampling Rate: 3%; Reconstruction Resolution: 1024*1024.')

    
# 重建
featuremap(pattern_path,image_path,featuremap_img,featuremap_mat,saveflag) 
print('All FeatureMap are saved successfully, the outputs are saved in the featuremap folder.')
