from torchvision.utils import save_image
from net.UDLSSPI005_step2 import *
from utils import *
import cv2
import scipy.io
from sklearn.preprocessing import MinMaxScaler
from features import *

#设置默认数据类型，设置显卡序号，设置torch数据类型
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
dtype = 'float32'
torch.set_default_tensor_type(torch.FloatTensor)
device = 'cuda'


"""重建相关参数"""
# 测试图片
data_path = './test/'

# 模型权重
model_path = './weights/saved_models_005/SPIS_779_psnr23.pth'

#pattern
pattern_path = "./pattern_005.mat"


#重建保存
recon_path = './results'

# 载入网络
net = LSSPI_two(path=model_path)
model = net.cuda()
#model.load_state_dict(torch.load(model_path, map_location = device))
print('load succesful')


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

    swinir=model
    swinir.eval()
    s1 = swinir.LSSPI_U.FeatureMap[0].weight.cpu().detach().numpy()
    io.savemat(pattern_path, {"data": s1})


# '''提取pattern'''
# extract(model, pattern_path)

# '''生成卷积后的结果图'''
# featureimgs(pattern_path,image_path,feature_img_save)

'''生成测量值'''
#feature(pattern_path,image_path,feature_mat_save)

# '''仿真'''
simulate_features(model, data_path)