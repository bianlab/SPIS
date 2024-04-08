import torch
import argparse
import os
import csv
import tqdm
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.datasets
from torchvision.utils import save_image
from torch.utils.data import dataset
import torch.utils.data as dataf
import torchvision.transforms as transforms
import numpy as np
import torch.backends.cudnn as cudnn
import torch.optim as optim
from nets.yolo import YoloBody
from nets.yolo_training import (YOLOLoss, get_lr_scheduler, set_optimizer_lr,
                                weights_init)
from utils.callbacks import LossHistory
from utils.dataloader import YoloDataset, yolo_dataset_collate
from utils.utils import get_classes
from utils.utils_fit import fit_one_epoch
from utils.utils import *
import os

#os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'



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
    # state_dict2 = {k: v for k, v in save_model.items() if k in model_dict.keys()}
    state_dict = {}
    for k, v in pretrained_dict.items():
        if k in model_dict.keys():
            # state_dict.setdefault(k, v)
            state_dict[k] = v
        else:
            print("Missing key(s) in state_dict :{}".format(k))
    return state_dict



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(torch.cuda.get_device_name())



classes_path    = 'model_data/voc_classes.txt'



model_path      = 'saved_models/yolox_s.pth'



input_shape     = [256, 256]



phi             = 's'



mosaic              = True



Init_Epoch          = 0
Freeze_Epoch        = 50
Freeze_batch_size   = 16



UnFreeze_Epoch      = 800
Unfreeze_batch_size = 16


Freeze_Train        = False



Init_lr             = 1e-2
Min_lr              = Init_lr * 0.01



optimizer_type      = "sgd"
momentum            = 0.937
weight_decay        = 5e-4



lr_decay_type       = "cos"



save_period         = 1



save_dir            = 'logs'



num_workers         = 4



train_annotation_path   = '2007_train.txt'
val_annotation_path     = '2007_val.txt'



class_names, num_classes = get_classes(classes_path)


model = YoloBody(num_classes,phi,picSize=256,patchSize=8)
weights_init(model)



if model_path != '':
    print('Load weights {}.'.format(model_path))
    device          = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_dict      = model.state_dict()
    pretrained_dict = torch.load(model_path, map_location = device)
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if np.shape(model_dict[k]) == np.shape(v)}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)



#是否加载预训练模型
pre_weights=''
if pre_weights:
    model = transfer_model(pre_weights, model)
    # model.load_state_dict(torch.load(opt.pre_weights))
    print('load weight %s'%(pre_weights))
else:
    weights_init(model)



yolo_loss    = YOLOLoss(num_classes)
loss_history = LossHistory(save_dir, model, input_shape=input_shape)




model_train = model.train()
Cuda=True
if Cuda:
    model_train = torch.nn.DataParallel(model)
    cudnn.benchmark = True
    model_train = model_train.cuda()



#---------------------------#
#   读取数据集对应的txt
#---------------------------#
with open(train_annotation_path, encoding='utf-8') as f:
    train_lines = f.readlines()
with open(val_annotation_path, encoding='utf-8') as f:
    val_lines   = f.readlines()
num_train   = len(train_lines)
num_val     = len(val_lines)



#------------------------------------------------------#
#   主干特征提取网络特征通用，冻结训练可以加快训练速度
#   也可以在训练初期防止权值被破坏。
#   Init_Epoch为起始世代
#   Freeze_Epoch为冻结训练的世代
#   UnFreeze_Epoch总训练世代
#   提示OOM或者显存不足请调小Batch_size
#------------------------------------------------------#
if True:
    UnFreeze_flag = False
    #------------------------------------#
    #   冻结一定部分训练
    #------------------------------------#
    if Freeze_Train:
        for param in model.backbone.parameters():
            param.requires_grad = False

    #-------------------------------------------------------------------#
    #   如果不冻结训练的话，直接设置batch_size为Unfreeze_batch_size
    #-------------------------------------------------------------------#
    batch_size = Freeze_batch_size if Freeze_Train else Unfreeze_batch_size

    #-------------------------------------------------------------------#
    #   判断当前batch_size，自适应调整学习率
    #-------------------------------------------------------------------#
    nbs             = 64
    lr_limit_max    = 1e-3 if optimizer_type == 'adam' else 5e-2
    lr_limit_min    = 3e-4 if optimizer_type == 'adam' else 5e-4
    Init_lr_fit     = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
    Min_lr_fit      = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)



#---------------------------------------#
#   根据optimizer_type选择优化器
#---------------------------------------#
pg0, pg1, pg2 = [], [], []
for k, v in model.named_modules():
    if hasattr(v, "bias") and isinstance(v.bias, nn.Parameter):
        pg2.append(v.bias)
    if isinstance(v, nn.BatchNorm2d) or "bn" in k:
        pg0.append(v.weight)
    elif hasattr(v, "weight") and isinstance(v.weight, nn.Parameter):
        pg1.append(v.weight)
optimizer = {
    'adam'  : optim.Adam(pg0, Init_lr_fit, betas = (momentum, 0.999)),
    'sgd'   : optim.SGD(pg0, Init_lr_fit, momentum = momentum, nesterov=True)
}[optimizer_type]
optimizer.add_param_group({"params": pg1, "weight_decay": weight_decay})
optimizer.add_param_group({"params": pg2})

#---------------------------------------#
#   获得学习率下降的公式
#---------------------------------------#
lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, UnFreeze_Epoch)

#---------------------------------------#
#   判断每一个世代的长度
#---------------------------------------#
epoch_step      = num_train // batch_size
epoch_step_val  = num_val // batch_size

if epoch_step == 0 or epoch_step_val == 0:
    raise ValueError("数据集过小，无法继续进行训练，请扩充数据集。")

#---------------------------------------#
#   构建数据集加载器。
#---------------------------------------#
train_dataset   = YoloDataset(train_lines, input_shape, num_classes, epoch_length = UnFreeze_Epoch, mosaic=mosaic, train = True)
val_dataset     = YoloDataset(val_lines, input_shape, num_classes, epoch_length = UnFreeze_Epoch, mosaic=False, train = False)
gen             = DataLoader(train_dataset, shuffle = True, batch_size = batch_size, num_workers = num_workers, pin_memory=True,
                            drop_last=True, collate_fn=yolo_dataset_collate)
gen_val         = DataLoader(val_dataset  , shuffle = True, batch_size = batch_size, num_workers = num_workers, pin_memory=True,
                            drop_last=True, collate_fn=yolo_dataset_collate)


#---------------------------------------#
#   开始模型训练
#---------------------------------------#
for epoch in range(Init_Epoch, UnFreeze_Epoch):
    #---------------------------------------#
    #   如果模型有冻结学习部分
    #   则解冻，并设置参数
    #---------------------------------------#
    if epoch >= Freeze_Epoch and not UnFreeze_flag and Freeze_Train:
        batch_size = Unfreeze_batch_size

        #-------------------------------------------------------------------#
        #   判断当前batch_size，自适应调整学习率
        #-------------------------------------------------------------------#
        nbs             = 64
        lr_limit_max    = 1e-3 if optimizer_type == 'adam' else 5e-2
        lr_limit_min    = 3e-4 if optimizer_type == 'adam' else 5e-4
        Init_lr_fit     = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
        Min_lr_fit      = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)
        #---------------------------------------#
        #   获得学习率下降的公式
        #---------------------------------------#
        lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, UnFreeze_Epoch)

        for param in model.backbone.parameters():
            param.requires_grad = True

        epoch_step      = num_train // batch_size
        epoch_step_val  = num_val // batch_size

        if epoch_step == 0 or epoch_step_val == 0:
            raise ValueError("数据集过小，无法继续进行训练，请扩充数据集。")

        gen     = DataLoader(train_dataset, shuffle = True, batch_size = batch_size, num_workers = num_workers, pin_memory=True,
                                    drop_last=True, collate_fn=yolo_dataset_collate)
        gen_val = DataLoader(val_dataset  , shuffle = True, batch_size = batch_size, num_workers = num_workers, pin_memory=True,
                                    drop_last=True, collate_fn=yolo_dataset_collate)

        UnFreeze_flag = True

    gen.dataset.epoch_now       = epoch
    gen_val.dataset.epoch_now   = epoch

    set_optimizer_lr(optimizer, lr_scheduler_func, epoch)

    fit_one_epoch(model_train, model, yolo_loss, loss_history, optimizer, epoch, epoch_step, epoch_step_val, gen, gen_val, UnFreeze_Epoch, Cuda, save_period, save_dir)