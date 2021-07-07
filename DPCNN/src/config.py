
"""
network config setting
"""
from easydict import EasyDict as edict

dpcnn_cfg = edict({
    'num_classes': 2,
    'momentum': 0.9,
    'num_epochs': 20, #20
    'batch_size': 64, # 64
    'seq_len': 33, # 33 498
    'embed_size': 50, # 300
    'hid_channels': 10, # 100
    'kernelsize': 3,
    'save_checkpoint_steps': 298, 
    'keep_checkpoint_max': 10, # 10
    'learning_rate': 0.001 # 0.0001
})

