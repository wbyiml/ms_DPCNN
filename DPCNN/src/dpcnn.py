import os
import json
import math
import random

import numpy as np

from mindspore import Tensor, nn, context, Parameter, ParameterTuple
from mindspore.common.initializer import initializer
from mindspore.ops import operations as P
import mindspore.ops as ops
import mindspore.ops.functional as F
import mindspore.common.dtype as mstype


class DPCNN(nn.Cell):
    """Sentiment network structure."""

    def __init__(self, seq_len, embed_size, hid_channels, kernelsize, num_classes):
        super(DPCNN, self).__init__()
        self.seq_len = seq_len
        self.embed_size = embed_size


        self.nochange_conv = nn.Conv2d(1, 1, (1, 1), stride=1)
        self.nochange_conv.weight.set_data(Tensor([[[[1]]]]))
        self.nochange_conv.weight.requires_grad = False
   


        # conv增大channel维度, kernel每三个word(整个word的embed)
        self.conv_channel = nn.Conv2d(1, hid_channels, (kernelsize, embed_size), stride=1,pad_mode="valid",padding=0)


        self.leaky_relu = nn.LeakyReLU()

        self.conv_word1 = nn.Conv2d(hid_channels, hid_channels, (kernelsize, 1), stride=1)  # pad_mode same
        self.conv_word2 = nn.Conv2d(hid_channels, hid_channels, (kernelsize, 1), stride=1)  # pad_mode same
        self.conv_word3 = nn.Conv2d(hid_channels, hid_channels, (kernelsize, 1), stride=1)  # pad_mode same
        self.conv_word4 = nn.Conv2d(hid_channels, hid_channels, (kernelsize, 1), stride=1)  # pad_mode same
        
        self.concat = P.Concat(1)

        self.pool = nn.AvgPool2d(kernel_size=(kernelsize,1), stride=2) 

        self.fc = nn.Dense(hid_channels, num_classes)


        # if context.get_context("device_target") == "CPU":
        #     # stack lstm by user 自定义层
        #     pass
        # elif context.get_context("device_target") == "GPU":
        #     # standard lstm  nn自带
        # else:
        #     raise ValueError("device is not CPU or GPU")
        

    def construct(self, inputs): 
        inputs = self.nochange_conv(inputs)

        # (bs, hid_channels, seq_len-2, 1)  
        embeddings = self.conv_channel(inputs)
        # TODO: conv_channel pad_mode same 

        # (64, 250, 48, 1)
        embeddings = self.norm(embeddings)
        embeddings = self.leaky_relu(embeddings)
        embeddings = self.conv_word1(embeddings)

        embeddings = self.norm(embeddings)
        embeddings = self.leaky_relu(embeddings)
        embeddings = self.conv_word2(embeddings) 



        while embeddings.shape[2] > 2:
            embeddings = self._block(embeddings)

        embeddings = embeddings.squeeze()
        
        # 输出2,softmax loss
        # TODO:  输出1，sigmoid判断正负
        outputs= self.fc(embeddings)  
 
        return outputs

    def _block(self, x):
        # x = self.pad_pool(x)
        px = self.pool(x)  

        x = self.norm(px)
        x = self.leaky_relu(x)
        x = self.conv_word3(x)

        x = self.norm(x)
        x = self.leaky_relu(x)
        x = self.conv_word4(x)


        # Short Cut
        x = x + px

        return x
    
    def norm(self,x,eps=1e-6):
        # layer norm
        bs = x.shape[0]

        mean = x.reshape(bs,-1).mean(-1).reshape(bs,1,1,1)
        std = ((x-mean)**2).reshape(bs,-1).mean(-1).reshape(bs,1,1,1) # ops.pow()

        norm_x = (x - mean) * ((std + eps)**-0.5)  # std ops.Rsqrt()

        return norm_x

