import os
import json
import math
import random

import numpy as np

import mindspore
from mindspore import Tensor, nn, context, Parameter, ParameterTuple
from mindspore.common.initializer import initializer
from mindspore.ops import operations as P
import mindspore.ops as ops
import mindspore.ops.functional as F
import mindspore.common.dtype as mstype


class DPCNN(nn.Cell):
    """Sentiment network structure."""

    def __init__(self, seq_len, embed_size, hid_channels, kernelsize, num_classes,weight_path,vocab_size):
        super(DPCNN, self).__init__()
        self.seq_len = seq_len
        self.embed_size = embed_size

        embedding_table = np.loadtxt(weight_path).astype(np.float32)
        self.unsup_embedding = nn.Embedding(vocab_size, embed_size, embedding_table=Tensor(embedding_table))
        self.unsup_embedding.embedding_table.requires_grad = False

        # self.onehot = nn.OneHot(depth=vocab_size, axis=-1)
        # self.basic_embedding = nn.Dense(vocab_size, embed_size)  # onehot bow n-gram embedding
        self.basic_embedding = nn.Embedding(vocab_size, embed_size)
        # self.basic_embedding.embedding_table.requires_grad = True

        
        # aclimdb pretrained
        # 增大模型 hid_channels kernelsize  
        # 调整参数
        # self.nochange_conv = nn.Conv2d(1, 1, (1, 1), stride=1)
        # self.nochange_conv.weight.set_data(Tensor([[[[1]]]]))
        # self.nochange_conv.weight.requires_grad = False


        # conv增大channel维度, kernel每三个word(整个word的embed)
        self.conv_region = nn.Conv2d(2, hid_channels, (kernelsize, embed_size), stride=1,pad_mode="valid",padding=0)


        self.leaky_relu = nn.LeakyReLU()

        self.conv_word1 = nn.Conv2d(hid_channels, hid_channels, (kernelsize, 1), stride=1)  # pad_mode same
        self.conv_word2 = nn.Conv2d(hid_channels, hid_channels, (kernelsize, 1), stride=1)  # pad_mode same
        self.conv_word3 = nn.Conv2d(hid_channels, hid_channels, (kernelsize, 1), stride=1)  # pad_mode same
        self.conv_word4 = nn.Conv2d(hid_channels, hid_channels, (kernelsize, 1), stride=1)  # pad_mode same

        self.conv_word5 = nn.Conv2d(hid_channels, hid_channels, (kernelsize, 1), stride=1,pad_mode="valid",padding=0)  # pad_mode same
        
        self.concat = P.Concat(1)

        self.pool = nn.AvgPool2d(kernel_size=(2,1), stride=2) 

        # self.dropout = nn.Dropout(keep_prob=0.8)

        self.fc = nn.Dense(hid_channels, num_classes)

        

        # if context.get_context("device_target") == "CPU":
        #     # stack lstm by user 自定义层
        #     pass
        # elif context.get_context("device_target") == "GPU":
        #     # standard lstm  nn自带
        # else:
        #     raise ValueError("device is not CPU or GPU")
        

    def construct(self, inputs):  # bs seq_len
        # inputs = inputs.reshape((inputs.shape[0],1,1,inputs.shape[1]))
        # inputs = inputs.astype(mindspore.float32)
        # inputs = self.nochange_conv(inputs)
        # inputs = inputs.reshape((inputs.shape[0],inputs.shape[-1]))
        # inputs = inputs.astype(mindspore.int32)

        unsup_embedding = self.unsup_embedding(inputs) # bs seq_len embed_size
        # unsup_embedding = unsup_embedding.reshape((unsup_embedding.shape[0],1,unsup_embedding.shape[1],unsup_embedding.shape[2]))

        # onehot = self.onehot(inputs) # bs seq_len vocab_size
        # onehot = onehot.reshape((-1,onehot.shape[-1]))  
        # onehot = self.basic_embedding(onehot)
        # onehot = onehot.reshape((inputs.shape[0],1,inputs.shape[1],onehot.shape[-1]))
        basic_embedding = self.basic_embedding(inputs) # bs seq_len embed_size
        # basic_embedding = basic_embedding.reshape((basic_embedding.shape[0],1,basic_embedding.shape[1],basic_embedding.shape[2]))

        # inputs1 = unsup_embedding+basic_embedding
        inputs1 = ops.Stack(axis=1)([unsup_embedding, basic_embedding])

        # (bs, hid_channels, seq_len-2, 1)  
        region_embeddings = self.conv_region(inputs1)
        # TODO: conv_region pad_mode same 

        # (64, 250, 48, 1)
        embeddings = self.norm(region_embeddings)
        embeddings = self.leaky_relu(embeddings)
        embeddings = self.conv_word1(embeddings)

        embeddings = self.norm(embeddings)
        embeddings = self.leaky_relu(embeddings)
        embeddings = self.conv_word2(embeddings) 

        embeddings = region_embeddings+embeddings



        while embeddings.shape[2] > 3:
            embeddings = self._block(embeddings)



        embeddings = self.norm(embeddings)
        embeddings = self.leaky_relu(embeddings)
        embeddings = self.conv_word5(embeddings) 
        # embeddings = self.norm(embeddings)
        # embeddings = self.leaky_relu(embeddings)
        embeddings = embeddings.squeeze()


        # embeddings = self.dropout(embeddings)
        
        # 输出2,softmax loss
        # TODO:  输出1，sigmoid判断正负
        outputs= self.fc(embeddings)  
 
        return outputs

    def _block(self, x):
        # x = self.pad_pool(x)
        px = self.pool(x)  # TODO: conv stride2

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

