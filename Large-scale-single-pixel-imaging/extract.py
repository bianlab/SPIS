from torchvision.utils import save_image
from net.UDLSSPI1k_step2_train import *
from utils import *
import cv2
from torch.autograd import Variable
import scipy.io
import numpy as np

"""设置默认数据类型,设置torch数据类型"""
dtype = 'float32'
torch.set_default_tensor_type(torch.FloatTensor)
device = 'cuda'

"""提取pattern"""
def extract(model, pattern_path):
    model.eval()
    s1 = model.LSSPI_U.FeatureMap[0].weight.cpu().detach().numpy()
    scipy.io.savemat(pattern_path, {"data": s1})

"""提取pattern相关参数"""
# pattern路径
pattern_path = "./pattern/pattern_1k.mat"

# 模型权重
model_path = './weights/UDLSSPI1k_step2.pth'

# step1权重
path_step1 = './weights/UDLSSPI1k_step1.pth'


# 载入网络
net = LSSPI_two(path=path_step1)
model = net.cuda()
model.load_state_dict(torch.load(model_path, map_location = device))
print('load succesful')
print('Sampling Rate: 3%; Reconstruction Resolution: 1024*1024.')

    
# 重建
extract(model,pattern_path) 
print('Patterns are saved successfully, the outputs are saved in the pattern folder.')
