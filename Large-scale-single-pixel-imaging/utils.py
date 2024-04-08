import math
import torch
import torch.nn as nn
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from torchvision import models
from skimage.metrics import structural_similarity as compare_ssim

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        nn.init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm') != -1:
        # nn.init.uniform(m.weight.data, 1.0, 0.02)
        m.weight.data.normal_(mean=0, std=math.sqrt(2./9./64.)).clamp_(-0.025,0.025)
        nn.init.constant(m.bias.data, 0.0)

class VGG19_PercepLoss(nn.Module):
    """ Calculates perceptual loss in vgg19 space
    """
    def __init__(self, _pretrained_=True):
        super(VGG19_PercepLoss, self).__init__()
        self.vgg = models.vgg19(pretrained=_pretrained_).features
        for param in self.vgg.parameters():
            param.requires_grad_(False)

    def get_features(self, image, layers=None):
        if layers is None: 
            layers = {'30': 'conv5_2'} # may add other layers
        features = {}
        x = image
        for name, layer in self.vgg._modules.items():
            x = layer(x)
            if name in layers:
                features[layers[name]] = x
        return features

    def forward(self, pred, true, layer='conv5_2'):
        true_f = self.get_features(true)
        pred_f = self.get_features(pred)
        return torch.mean((true_f[layer]-pred_f[layer])**2)

# batch平均PSNR
def batch_PSNR(img, imclean, data_range):
    Img = img.data.cpu().numpy().astype(np.float32)
    Iclean = imclean.data.cpu().numpy().astype(np.float32)
    PSNR = 0
    for i in range(Img.shape[0]):
        PSNR += compare_psnr(Iclean[i,:,:,:], Img[i,:,:,:], data_range=data_range)
    return (PSNR/Img.shape[0])

# 图片PSNR
def PSNR(imgori,img,data_range):
    imgori = imgori.data.cpu().numpy().astype(np.float32)
    img = img.data.cpu().numpy().astype(np.float32)
    PSNR = compare_psnr(imgori, img, data_range=data_range)
    return (PSNR)

# 图片SSIM
def SSIM(imgori,img,data_range):
    imgori = imgori.data.cpu().numpy().astype(np.float32).reshape(256,256,3)
    img = img.data.cpu().numpy().astype(np.float32).reshape(256,256,3)
    ssim = compare_ssim(imgori, img, multichannel = True,data_range=data_range)
    return (ssim)


def data_augmentation(image, mode):
    out = np.transpose(image, (1,2,0))
    #out = image
    if mode == 0:
        # original
        out = out
    elif mode == 1:
        # flip up and down
        out = np.flipud(out)
    elif mode == 2:
        # rotate counterwise 90 degree
        out = np.rot90(out)
    elif mode == 3:
        # rotate 90 degree and flip up and down
        out = np.rot90(out)
        out = np.flipud(out)
    elif mode == 4:
        # rotate 180 degree
        out = np.rot90(out, k=2)
    elif mode == 5:
        # rotate 180 degree and flip
        out = np.rot90(out, k=2)
        out = np.flipud(out)
    elif mode == 6:
        # rotate 270 degree
        out = np.rot90(out, k=3)
    elif mode == 7:
        # rotate 270 degree and flip
        out = np.rot90(out, k=3)
        out = np.flipud(out)
    return np.transpose(out, (2,0,1))
    #return out

def normalize(matrix):
    # 计算每列的最小值和最大值
    min_vals = np.min(matrix, axis=0)
    max_vals = np.max(matrix, axis=0)
	
    # 计算归一化后的矩阵
    normalized_matrix = (matrix - min_vals) / (max_vals - min_vals)
    
    return normalized_matrix

def conv(pattern,data):
    n,m = data.shape
    k = pattern.shape[1] #k=32
    line = []
    for i in range(n//k): #i=8
        for j in range(m//k): #j=8
            s1 = i*k
            s2 = j*k
            a = data[s1:s1+k,s2:s2+k]
            line.append((np.multiply(pattern,a)))
    return np.array(line)

