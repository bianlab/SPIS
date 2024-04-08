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
from Transformer import *
from PositionalEncoding import *
from IntmdSequential import *
from utils import *


class LSSPI(nn.Module):
    def __init__(self,picSize,patchSize,
        FeaturemapNum=32,
        Numchannels=1,
        embedding_dim=32,
        num_layers=8,
        num_heads=8,
        hidden_dim=64,
        dropout_rate=0.0,
        conv_patch_representation=True,
        positional_encoding_type="learned",
        num_patches=32, #通道数
        attn_dropout_rate=0.0):
        super(LSSPI, self).__init__()
        self.picSize = picSize
        self.patchSize = patchSize
        self.FeaturemapNum= FeaturemapNum
        self.Numchannels=Numchannels
        self.dropout_rate = dropout_rate
        self.num_patches=num_patches
        self.embedding_dim = embedding_dim  #32
        self.seq_length = self.num_patches
        self.attn_dropout_rate = attn_dropout_rate  #注意力模块的dropout比率
        self.conv_patch_representation = conv_patch_representation  #True
        #self.flatten_dim = 128 * num_channels
        self.FeatureMap = nn.Sequential(
            nn.Conv2d(in_channels=self.Numchannels, out_channels=self.FeaturemapNum,
                kernel_size=32, stride=32,bias=False,padding=0),
            nn.BatchNorm2d(self.FeaturemapNum,0.8),
            nn.ReLU(inplace=True)
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

        self.bn = nn.BatchNorm2d(32) 
        self.relu = nn.ReLU(inplace=True)

        self.Conv_0 = nn.Conv2d(self.embedding_dim,64,kernel_size=1,stride=1)


        self.up=nn.Upsample(scale_factor=2)
        self.relu=nn.ReLU(inplace=True)
        #8*8*self.FeaturemapNum
        self.conv1=nn.Sequential(
            #nn.Upsample(scale_factor=2),
            nn.Conv2d(64,64,kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(64,0.8),
            nn.ReLU(inplace=True),
            nn.Conv2d(64,64,kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(64,0.8),
        	)
        #16*16*64
        self.conv2=nn.Sequential(
            nn.Conv2d(64,64,kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(64,0.8),
            nn.ReLU(inplace=True),
            nn.Conv2d(64,64,kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(64,0.8),
        	)
        #32*32*64
        self.conv3=nn.Sequential(
            nn.Conv2d(64,64,kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(64,0.8),
            nn.ReLU(inplace=True),
            nn.Conv2d(64,64,kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(64,0.8),
        	)
        #64*64*64
        self.conv4=nn.Sequential(
            nn.Conv2d(64,64,kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(64,0.8),
            nn.ReLU(inplace=True),
            nn.Conv2d(64,64,kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(64,0.8),
        	)
        #128*128*64
        self.conv5=nn.Sequential(
            nn.Conv2d(64,64,kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(64,0.8),
            nn.ReLU(inplace=True),
            nn.Conv2d(64,64,kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(64,0.8),
        	)
        #256*256*64
        #self.conv5=nn.Conv2d(128,1,kernel_size=3,stride=1,padding=1,bias=True,dilation=2)
        self.conv6= nn.Sequential(
            nn.Conv2d(64,1, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True)
        )
        #256*256*1
        self.init_weights()


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


    def forward(self, x):
        x = self.FeatureMap(x)
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

        #256*256*1
        return x

