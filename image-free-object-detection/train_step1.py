import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as utils
import  time 
from torch.autograd import Variable
from torch.utils.data import DataLoader

from torch.nn.modules.loss import _Loss 
from nets.SPOD import *
from utils import *
from utils.utils import batch_PSNR
import cv2
import random
import torch.utils.data as dataf
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
import time as time
import datetime
import sys
from torchvision.utils import save_image
import csv


"""设置默认数据类型,设置显卡序号,设置torch数据类型"""
dtype = 'float32'
os.environ["CUDA_VISIBLE_DEVICES"] = '3'
torch.set_default_tensor_type(torch.FloatTensor)


"""训练及网络参数设置"""
batch_size = 16

num_workers = 0

# 图片大小
picSize = 256 

# pattern大小
patchSize = 8 

# 是否加载预训练权重
pre_weights = False 

# 预训练权重
pre_weights_path = './saved_models_005/saved_models_005step0/SPOD_350_psnr18.pth'

# 学习率
LR = 0.0001

# 保存权重间隔多少epoch
checkpoint_interval=10

# 总训练epoch数
n_epochs=600


def transfer_model(pretrained_file, model):
    '''
    只导入pretrained_file部分模型参数
    tensor([-0.7119,  0.0688, -1.7247, -1.7182, -1.2161, -0.7323, -2.1065, -0.5433,-1.5893, -0.5562]
    update:
        D.update([E, ]**F) -> None.  Update D from dict/iterable E and F.
        If E is present and has a .keys() method, then does:  for k in E: D[k] = E[k]
        If E is present and lacks a .keys() method, then does:  for k, v in E: D[k] = v
        In either case, this is followed by: for k in F:  D[k] = F[k]
    :param pretrained_file:
    :param model:
    :return:
    '''
    pretrained_dict = torch.load(pretrained_file)  # get pretrained dict
    model_dict = model.state_dict()  # get model dict
    # 在合并前(update),需要去除pretrained_dict一些不需要的参数
    pretrained_dict = transfer_state_dict(pretrained_dict, model_dict)
    model_dict.update(pretrained_dict)  # 更新(合并)模型的参数
    model.load_state_dict(model_dict)
    return model

def transfer_state_dict(pretrained_dict, model_dict):
    '''
    根据model_dict,去除pretrained_dict一些不需要的参数,以便迁移到新的网络
    url: https://blog.csdn.net/qq_34914551/article/details/87871134
    :param pretrained_dict:
    :param model_dict:
    :return:
    '''
    # state_dict2 = {k: v for k, v in save_model.items() if k in model_dict.keys()}
    state_dict = {}
    for k, v in pretrained_dict.items():
        if k in model_dict.keys():
            # state_dict.setdefault(k, v)
            state_dict[k] = v
        else:
            print("Missing key(s) in state_dict :{}".format(k))
    return state_dict


"""测试集中输出重建测试样例"""
def sample_images(batches_done,model,X_test,y_test): 
    swinir=model
    swinir.eval()

    # 训练集中随机挑选一张图片
    i=random.randrange(1,425) 

    real_A = Variable(X_test[i,:,:,:]).cuda()
    real_B = Variable(y_test[i,:,:,:]).cuda()
    real_A=real_A.unsqueeze(0)
    real_B=real_B.unsqueeze(0)
    
    # 重建图片并且和GT拼接
    fake_B = swinir(real_A) 
    imgx=fake_B.data
    imgy=real_B.data
    x=imgx[:,:,:,:]
    y=imgy[:,:,:,:]
    img_sample = torch.cat((x,y), -2)

    # 重建图像保存路径
    save_image(img_sample, "%s/%s.png" % ('results/recon', batches_done), nrow=5, normalize=True)


"""主要训练过程"""
def main():

    # 训练数据集路径以及载入训练数据集
    path='./image_train/train/'
    training_x=[]
    path_list = os.listdir(path)
    path_list.sort(key=lambda x:int(x.split('.')[0]))
    for item in path_list:
        imgx= cv2.imread(path+item)
        imgx=cv2.cvtColor(imgx,cv2.COLOR_BGR2RGB)
        imgx=cv2.resize(imgx,(256,256))
        imgx=imgx/255.0
        training_x.append(imgx)
    X_train = []

    for features in training_x:
        X_train.append(features)
    X_train = np.array(X_train)
    X_train=X_train.astype(dtype)
    X_train= torch.from_numpy(X_train)
    X_train=X_train.permute(0,3,1,2)
    y_train=X_train


    # 测试数据集路径以及载入测试数据集
    path='./image_train/test/'
    test_x=[]
    path_list = os.listdir(path)
    path_list.sort(key=lambda x:int(x.split('.')[0]))

    for item in path_list:
        imgx= cv2.imread(path+item)
        imgx=cv2.cvtColor(imgx,cv2.COLOR_BGR2RGB)
        imgx=cv2.resize(imgx,(256,256))
        imgx=imgx/255.0
        test_x.append(imgx)
    X_test=[]

    for features in test_x:
        X_test.append(features)

    X_test = np.array(X_test) 
    X_test=X_test.astype(dtype)
    X_test= torch.from_numpy(X_test)
    X_test=X_test.permute(0,3,1,2)
    y_test=X_test

    # 训练集转为dataloader，设置batchsize和numwork
    dataset = dataf.TensorDataset(X_train,y_train)
    loader = dataf.DataLoader(dataset, batch_size=batch_size, shuffle=True,num_workers=num_workers)

    # 载入网络
    net = LSSPI(picSize=picSize,patchSize=patchSize)

    # 加载预训练权重
    if pre_weights:
        net = transfer_model(pre_weights_path, net)
        print('load weight %s'%(pre_weights_path))
    else:
        print('No pretrain model found, training will start from scratch')

    # 设置损失函数并移到gpu上
    criterion = nn.MSELoss(size_average=False)
    model = net.cuda()
    criterion.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, betas=(0.5, 0.999))
    scheduler=lr_scheduler.StepLR(optimizer,step_size=80,gamma=0.8)



    # 开始训练
    for epoch in range(n_epochs):
        for i, data in enumerate(loader, 0):
            imgn_train = Variable(data[0]).cuda()
            img_train= Variable(data[1]).cuda()
            optimizer.zero_grad()
            sr = model(imgn_train)


            loss = criterion(sr, img_train)


            loss.backward()
            optimizer.step()
            out_train1 = torch.clamp(sr, 0., 1.)
            psnr_train = batch_PSNR(out_train1, img_train, 1.)
            sys.stdout.write("\r[epoch %d][%d/%d] loss: %.4f PSNR_train: %.4f" %
                (epoch+1, i+1, len(loader), loss.item(), psnr_train))

        scheduler.step()

        # 保存权重
        if checkpoint_interval != -1 and epoch % checkpoint_interval == 0:
            torch.save(model.state_dict(), "saved_models_005/saved_models_005step1/SPOD_%d_psnr%d.pth" % (epoch,psnr_train)) #要改
        
        # 输出测试用例
        if  (epoch+1) % 1 == 0:    
            sample_images(epoch,model,X_test,y_test)


if __name__ == '__main__':
    main()