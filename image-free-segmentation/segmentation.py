from torchvision.utils import save_image
from net.UDLSSPI005_step2 import *
from utils import *
import cv2
import scipy.io as io
import scipy.io
from sklearn.preprocessing import MinMaxScaler
from features import *

#设置默认数据类型，设置显卡序号，设置torch数据类型
dtype = 'float32'
torch.set_default_tensor_type(torch.FloatTensor)
device = 'cuda'
"""重建相关参数"""

# 模型权重
model_path = './weights/saved_models_005/SPIS_779_psnr23.pth'

#pattern
pattern_path = "./pattern_005.mat"

#卷积后的featuremat保存位置
feature_path = './features/'

#重建保存
recon_path = './results'

# 载入网络
net = LSSPI_two(path=model_path)
model = net.cuda()
print('load succesful')

def image_free_segement(model, feature_path,item):
    SPIS=model
    SPIS.eval()
    dict_ = scipy.io.loadmat(feature_path)
    features = dict_['data']
    feature=torch.FloatTensor(features).unsqueeze(0).cuda()
    print(feature.shape)
    output = SPIS(feature)
    img = F.sigmoid(output[0].data)
    save_image(img, "%s/%s.jpg" % ('results', item[:-4]), nrow=5, normalize=True)
    print("The %s-th scene was segemented successfully"% (item[:-4]))




# '''仿真重建'''
feature_list = os.listdir(feature_path)
for item in feature_list:
    path=feature_path+item
    image_free_segement(model, path,item)