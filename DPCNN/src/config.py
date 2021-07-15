
"""
network config setting
"""
from easydict import EasyDict as edict

dpcnn_cfg = edict({
    'num_classes': 2,
    'momentum': 0.9,
    'learning_rate': 0.00001, # 0.0001 0.000001
    'num_epochs': 20, #20
    'batch_size': 64, # 64
    'seq_len': 26, # 34(18)  36(20)
    'embed_size': 300, # 300
    'hid_channels': 500, # 100 500
    'kernelsize': 3, # 3 5
    'keep_checkpoint_max': 10 # 10
})


# parser glove   加入test数据   pretrained 0.0001 0.00001
