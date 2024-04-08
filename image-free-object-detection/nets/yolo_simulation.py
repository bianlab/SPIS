#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F 
import os
import time
import timeit
import copy
import numpy as np 
from torch.nn import ModuleList
from torch.nn import Conv2d
from torch.nn import LeakyReLU
from nets.Transformer import *
from nets.PositionalEncoding import *
from nets.IntmdSequential import *
from utils import *

from .darknet import BaseConv, CSPDarknet, CSPLayer, DWConv

class weightConstraint(object):
    def __init__(self):
        pass

    def __call__(self, module):
        if hasattr(module, 'weight'):
            #print("Entered")
            w = module.weight.data
            w = w.clamp(0, 1)  # 将参数范围限制到0-1之间
            module.weight.data = w

class YOLOXHead(nn.Module):
    def __init__(self, num_classes, width = 1.0, in_channels = [256, 512, 1024], act = "silu", depthwise = False,):
        super().__init__()
        Conv            = DWConv if depthwise else BaseConv
        
        self.cls_convs  = nn.ModuleList()
        self.reg_convs  = nn.ModuleList()
        self.cls_preds  = nn.ModuleList()
        self.reg_preds  = nn.ModuleList()
        self.obj_preds  = nn.ModuleList()
        self.stems      = nn.ModuleList()

        for i in range(len(in_channels)):
            self.stems.append(BaseConv(in_channels = int(in_channels[i] * width), out_channels = int(256 * width), ksize = 1, stride = 1, act = act))
            self.cls_convs.append(nn.Sequential(*[
                Conv(in_channels = int(256 * width), out_channels = int(256 * width), ksize = 3, stride = 1, act = act), 
                Conv(in_channels = int(256 * width), out_channels = int(256 * width), ksize = 3, stride = 1, act = act), 
            ]))
            self.cls_preds.append(
                nn.Conv2d(in_channels = int(256 * width), out_channels = num_classes, kernel_size = 1, stride = 1, padding = 0)
            )
            

            self.reg_convs.append(nn.Sequential(*[
                Conv(in_channels = int(256 * width), out_channels = int(256 * width), ksize = 3, stride = 1, act = act), 
                Conv(in_channels = int(256 * width), out_channels = int(256 * width), ksize = 3, stride = 1, act = act)
            ]))
            self.reg_preds.append(
                nn.Conv2d(in_channels = int(256 * width), out_channels = 4, kernel_size = 1, stride = 1, padding = 0)
            )
            self.obj_preds.append(
                nn.Conv2d(in_channels = int(256 * width), out_channels = 1, kernel_size = 1, stride = 1, padding = 0)
            )

    def forward(self, inputs):
        #---------------------------------------------------#
        #   inputs输入
        #   P3_out  80, 80, 256
        #   P4_out  40, 40, 512
        #   P5_out  20, 20, 1024
        #---------------------------------------------------#
        outputs = []
        for k, x in enumerate(inputs):
            #---------------------------------------------------#
            #   利用1x1卷积进行通道整合
            #---------------------------------------------------#
            x       = self.stems[k](x)
            #---------------------------------------------------#
            #   利用两个卷积标准化激活函数来进行特征提取
            #---------------------------------------------------#
            cls_feat    = self.cls_convs[k](x)
            #---------------------------------------------------#
            #   判断特征点所属的种类
            #   80, 80, num_classes
            #   40, 40, num_classes
            #   20, 20, num_classes
            #---------------------------------------------------#
            cls_output  = self.cls_preds[k](cls_feat)

            #---------------------------------------------------#
            #   利用两个卷积标准化激活函数来进行特征提取
            #---------------------------------------------------#
            reg_feat    = self.reg_convs[k](x)
            #---------------------------------------------------#
            #   特征点的回归系数
            #   reg_pred 80, 80, 4
            #   reg_pred 40, 40, 4
            #   reg_pred 20, 20, 4
            #---------------------------------------------------#
            reg_output  = self.reg_preds[k](reg_feat)
            #---------------------------------------------------#
            #   判断特征点是否有对应的物体
            #   obj_pred 80, 80, 1
            #   obj_pred 40, 40, 1
            #   obj_pred 20, 20, 1
            #---------------------------------------------------#
            obj_output  = self.obj_preds[k](reg_feat)

            output      = torch.cat([reg_output, obj_output, cls_output], 1)
            outputs.append(output)
        return outputs

class YOLOPAFPN(nn.Module):
    def __init__(self, depth = 1.0, width = 1.0, in_features = ("dark3", "dark4", "dark5"), in_channels = [256, 512, 1024], depthwise = False, act = "silu"):
        super().__init__()
        Conv                = DWConv if depthwise else BaseConv
        self.backbone       = CSPDarknet(depth, width, depthwise = depthwise, act = act)
        self.in_features    = in_features

        self.upsample       = nn.Upsample(scale_factor=2, mode="nearest")

        #-------------------------------------------#
        #   20, 20, 1024 -> 20, 20, 512
        #-------------------------------------------#
        self.lateral_conv0  = BaseConv(int(in_channels[2] * width), int(in_channels[1] * width), 1, 1, act=act)
    
        #-------------------------------------------#
        #   40, 40, 1024 -> 40, 40, 512
        #-------------------------------------------#
        self.C3_p4 = CSPLayer(
            int(2 * in_channels[1] * width),
            int(in_channels[1] * width),
            round(3 * depth),
            False,
            depthwise = depthwise,
            act = act,
        )  

        #-------------------------------------------#
        #   40, 40, 512 -> 40, 40, 256
        #-------------------------------------------#
        self.reduce_conv1   = BaseConv(int(in_channels[1] * width), int(in_channels[0] * width), 1, 1, act=act)
        #-------------------------------------------#
        #   80, 80, 512 -> 80, 80, 256
        #-------------------------------------------#
        self.C3_p3 = CSPLayer(
            int(2 * in_channels[0] * width),
            int(in_channels[0] * width),
            round(3 * depth),
            False,
            depthwise = depthwise,
            act = act,
        )

        #-------------------------------------------#
        #   80, 80, 256 -> 40, 40, 256
        #-------------------------------------------#
        self.bu_conv2       = Conv(int(in_channels[0] * width), int(in_channels[0] * width), 3, 2, act=act)
        #-------------------------------------------#
        #   40, 40, 256 -> 40, 40, 512
        #-------------------------------------------#
        self.C3_n3 = CSPLayer(
            int(2 * in_channels[0] * width),
            int(in_channels[1] * width),
            round(3 * depth),
            False,
            depthwise = depthwise,
            act = act,
        )

        #-------------------------------------------#
        #   40, 40, 512 -> 20, 20, 512
        #-------------------------------------------#
        self.bu_conv1       = Conv(int(in_channels[1] * width), int(in_channels[1] * width), 3, 2, act=act)
        #-------------------------------------------#
        #   20, 20, 1024 -> 20, 20, 1024
        #-------------------------------------------#
        self.C3_n4 = CSPLayer(
            int(2 * in_channels[1] * width),
            int(in_channels[2] * width),
            round(3 * depth),
            False,
            depthwise = depthwise,
            act = act,
        )

    def forward(self, input):
        out_features            = self.backbone.forward(input)
        [feat1, feat2, feat3]   = [out_features[f] for f in self.in_features]

        #-------------------------------------------#
        #   20, 20, 1024 -> 20, 20, 512
        #-------------------------------------------#
        P5          = self.lateral_conv0(feat3)
        #-------------------------------------------#
        #  20, 20, 512 -> 40, 40, 512
        #-------------------------------------------#
        P5_upsample = self.upsample(P5)
        #-------------------------------------------#
        #  40, 40, 512 + 40, 40, 512 -> 40, 40, 1024
        #-------------------------------------------#
        P5_upsample = torch.cat([P5_upsample, feat2], 1)
        #-------------------------------------------#
        #   40, 40, 1024 -> 40, 40, 512
        #-------------------------------------------#
        P5_upsample = self.C3_p4(P5_upsample)

        #-------------------------------------------#
        #   40, 40, 512 -> 40, 40, 256
        #-------------------------------------------#
        P4          = self.reduce_conv1(P5_upsample) 
        #-------------------------------------------#
        #   40, 40, 256 -> 80, 80, 256
        #-------------------------------------------#
        P4_upsample = self.upsample(P4) 
        #-------------------------------------------#
        #   80, 80, 256 + 80, 80, 256 -> 80, 80, 512
        #-------------------------------------------#
        P4_upsample = torch.cat([P4_upsample, feat1], 1) 
        #-------------------------------------------#
        #   80, 80, 512 -> 80, 80, 256
        #-------------------------------------------#
        P3_out      = self.C3_p3(P4_upsample)  

        #-------------------------------------------#
        #   80, 80, 256 -> 40, 40, 256
        #-------------------------------------------#
        P3_downsample   = self.bu_conv2(P3_out) 
        #-------------------------------------------#
        #   40, 40, 256 + 40, 40, 256 -> 40, 40, 512
        #-------------------------------------------#
        P3_downsample   = torch.cat([P3_downsample, P4], 1) 
        #-------------------------------------------#
        #   40, 40, 256 -> 40, 40, 512
        #-------------------------------------------#
        P4_out          = self.C3_n3(P3_downsample) 

        #-------------------------------------------#
        #   40, 40, 512 -> 20, 20, 512
        #-------------------------------------------#
        P4_downsample   = self.bu_conv1(P4_out)
        #-------------------------------------------#
        #   20, 20, 512 + 20, 20, 512 -> 20, 20, 1024
        #-------------------------------------------#
        P4_downsample   = torch.cat([P4_downsample, P5], 1)
        #-------------------------------------------#
        #   20, 20, 1024 -> 20, 20, 1024
        #-------------------------------------------#
        P5_out          = self.C3_n4(P4_downsample)

        return (P3_out, P4_out, P5_out)

class YoloBody(nn.Module):
    def __init__(self, num_classes, phi,patchSize,picSize=256,
        FeaturemapNum=96,
        Numchannels=3,
        embedding_dim=96,
        num_layers=8,
        num_heads=8,
        hidden_dim=64,
        dropout_rate=0.0,
        conv_patch_representation=True,
        positional_encoding_type="learned",
        num_patches=96, #通道数
        attn_dropout_rate=0.0):
        super().__init__()
        self.picSize = picSize
        self.patchSize = patchSize
        self.FeaturemapNum= FeaturemapNum
        self.Numchannels=Numchannels
        self.outchannels=Numchannels
        self.dropout_rate = dropout_rate
        self.num_patches=num_patches
        self.embedding_dim = embedding_dim  #32
        self.seq_length = self.num_patches
        self.attn_dropout_rate = attn_dropout_rate  #注意力模块的dropout比率
        self.conv_patch_representation = conv_patch_representation  #True
        self.FeatureMap = nn.Sequential(
            nn.Conv2d(in_channels=self.Numchannels, out_channels=self.FeaturemapNum,
                kernel_size=32, stride=32,bias=False,padding=0),
            #nn.BatchNorm2d(self.FeaturemapNum,0.8),
            #nn.ReLU(inplace=True)
            )
        #self.Conv_in=nn.Conv2d(32, out_channels=self.FeaturemapNum,kernel_size=32, stride=32,bias=False,padding=0)

        #线性编码
        #self.linear_encoding = nn.Linear(self.flatten_dim, self.embedding_dim)
        #位置编码
        if positional_encoding_type == "learned":
            self.position_encoding = LearnedPositionalEncoding(
                self.seq_length, self.embedding_dim, self.seq_length
            )
        elif positional_encoding_type == "fixed":
            self.position_encoding = FixedPositionalEncoding(
                self.embedding_dim,
            )

        self.pe_dropout = nn.Dropout(p=self.dropout_rate)

        self.transformer = TransformerModel(
            embedding_dim, #32
            num_layers, #4
            num_heads,  #8
            hidden_dim,  #64

            self.dropout_rate,
            self.attn_dropout_rate,
        )

        #layer Norm
        self.pre_head_ln = nn.LayerNorm(embedding_dim)

        if self.conv_patch_representation:

            self.Conv_x = nn.Conv2d(
                256,
                self.embedding_dim,  #32
                kernel_size=3,  
                stride=1,
                padding=1
            )

        self.bn = nn.BatchNorm2d(96) 
        self.relu = nn.ReLU(inplace=True)

        self.Conv_0 = nn.Conv2d(self.embedding_dim,96,kernel_size=1,stride=1)


        self.up=nn.Upsample(scale_factor=2)
        self.relu=nn.ReLU(inplace=True)
        #8*8*self.FeaturemapNum
        self.conv1=nn.Sequential(
            #nn.Upsample(scale_factor=2),
            nn.Conv2d(96,96,kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(96,0.8),
            nn.ReLU(inplace=True),
            nn.Conv2d(96,96,kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(96,0.8),
            )
        #16*16*64
        self.conv2=nn.Sequential(
            nn.Conv2d(96,96,kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(96,0.8),
            nn.ReLU(inplace=True),
            nn.Conv2d(96,96,kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(96,0.8),
            )
        #32*32*64
        self.conv3=nn.Sequential(
            nn.Conv2d(96,96,kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(96,0.8),
            nn.ReLU(inplace=True),
            nn.Conv2d(96,96,kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(96,0.8),
            )
        #64*64*64
        self.conv4=nn.Sequential(
            nn.Conv2d(96,96,kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(96,0.8),
            nn.ReLU(inplace=True),
            nn.Conv2d(96,96,kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(96,0.8),
            )
        #128*128*64
        self.conv5=nn.Sequential(
            nn.Conv2d(96,96,kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(96,0.8),
            nn.ReLU(inplace=True),
            nn.Conv2d(96,96,kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(96,0.8),
            )
        #256*256*64
        #self.conv5=nn.Conv2d(128,1,kernel_size=3,stride=1,padding=1,bias=True,dilation=2)
        self.conv6= nn.Sequential(
            nn.Conv2d(96,self.outchannels, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(3),
            nn.ReLU(inplace=True)
        )
        #256*256*1
        self.init_weights()





        depth_dict = {'nano': 0.33, 'tiny': 0.33, 's' : 0.33, 'm' : 0.67, 'l' : 1.00, 'x' : 1.33,}
        width_dict = {'nano': 0.25, 'tiny': 0.375, 's' : 0.50, 'm' : 0.75, 'l' : 1.00, 'x' : 1.25,}
        depth, width    = depth_dict[phi], width_dict[phi]
        depthwise       = True if phi == 'nano' else False 

        self.backbone   = YOLOPAFPN(depth, width, depthwise=depthwise)
        self.head       = YOLOXHead(num_classes, width, depthwise=depthwise)

    def reshape_output(self,x): #将transformer的输出resize为原来的特征图尺寸
        x = x.view(
            x.size(0),
            self.patchSize,
            self.patchSize,
            #int(self.img_dim / self.patch_dim),
            #int(self.img_dim / self.patch_dim),
            self.embedding_dim,
            )#B,16,16,512
        x = x.permute(0, 3, 1, 2).contiguous()

        return x

    def init_weights(net, init_type='normal', gain=0.02):
        def init_func(m):
            classname = m.__class__.__name__
            if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if init_type == 'normal':
                    init.normal_(m.weight.data, 0.0, gain)
                elif init_type == 'xavier':
                    init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == 'kaiming':
                    init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    init.orthogonal_(m.weight.data, gain=gain)
                else:
                    raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
                if hasattr(m, 'bias') and m.bias is not None:
                    init.constant_(m.bias.data, 0.0)
            elif classname.find('BatchNorm2d') != -1:
                init.normal_(m.weight.data, 1.0, gain)
                init.constant_(m.bias.data, 0.0)

        print('initialize network with %s' % init_type)
        net.apply(init_func)    
    
    
    @ staticmethod
    def linear_norm(x):
        # print(x.shape)
        y = x.view(x.size(0), -1)
        y_max = y.max(1).values.unsqueeze(1)
        y_min = y.min(1).values.unsqueeze(1)
        # print(y.shape)
        # print(y_max.shape)
        # print(y_min.shape)
        y = (y - y_min) / (y_max - y_min)
        return y.view(x.size())

    def forward(self, x):
        #print(x.shape)
        #x = self.FeatureMap(x)
        #feature = x
        x=self.bn(x)
        x=self.relu(x)
        #x = self.linear_norm(x)
        #8*8*32
        #spatial-wise transformer-based attention
        residual=x
        #将测量值输入transformer中，测量值尺寸为8*8*32
        #x= self.Conv_in(x)
        x= x.permute(0, 2, 3, 1).contiguous()  #B,32,8,8-->B,8,8,32
        x= x.view(x.size(0), -1, self.embedding_dim) #B,8,8,32->B,8*8,32 线性映射层
        x= self.position_encoding(x) #位置编码
        x= self.pe_dropout(x) #预dropout层
        x= self.transformer(x)
        x= self.pre_head_ln(x)
        x= self.reshape_output(x) #out->32*8*8 shape->B,32,8,8
        x=x+residual


        x=self.Conv_0(x)



        #8*8*self.FeaturemapNum
        x=self.up(x)
        residual=x
        x=self.conv1(x)
        x=torch.add(x,residual)
        x=self.relu(x)


        #16*16*64
        x=self.up(x)
        residual=x
        x=self.conv2(x)
        x=torch.add(x,residual)
        x=self.relu(x)

        #32*32*64
        x=self.up(x)
        residual=x
        x=self.conv3(x)
        x=torch.add(x,residual)
        x=self.relu(x)

        #64*64*64
        x=self.up(x)
        residual=x
        x=self.conv4(x)
        x=torch.add(x,residual)
        x=self.relu(x)

        #128*128*64
        x=self.up(x)
        residual=x
        x=self.conv5(x)
        x=torch.add(x,residual)
        x=self.relu(x)

        #256*256*64
        x=self.conv6(x)
        #print(x.shape)
        fpn_outs    = self.backbone.forward(x)
        outputs     = self.head.forward(fpn_outs)
        return outputs
