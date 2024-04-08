from torchvision.utils import save_image
from net.UDLSSPI005_step2 import *
from utils import *
import cv2
from torch.autograd import Variable
import scipy.io
import numpy as np

"""设置默认数据类型,设置torch数据类型"""
dtype = 'float32'
torch.set_default_tensor_type(torch.FloatTensor)
device = 'cuda'

"""重建图像"""
def recon_images(model,feature_path):
    SPIS=model
    SPIS.eval()
    dict_ = scipy.io.loadmat(feature_path)
    features = dict_['data']
    #print(features.dtype())
    #print(features.shape)
    for i in range(len(features)):
        feature = torch.from_numpy(features[i].astype(np.float32)).unsqueeze(0).cuda()
        output = SPIS(feature)
        img = output[0].data
        save_image(img, "%s/%s.jpg" % ('results', i), nrow=5, normalize=True)
        print("The %s-th scene was reconstructed successfully"% (i))

"""重建相关参数"""
# 测试集
feature_path = './features/features.mat'

# 模型权重
model_path = './weights/UDLSSPI_STEP2.pth'

# step1权重
path_step1 = './weights/UDLSSPI_STEP1.pth'


# 载入网络
net = LSSPI_two(path=path_step1)
model = net.cuda()
model.load_state_dict(torch.load(model_path, map_location = device))
print('load succesful')
print('Sampling Rate: 5%; Reconstruction Resolution: 512*512.')

    
# 重建
recon_images(model,feature_path) 
print('All images are reconstructed successfully, the outputs are saved in the results folder.')
