
import argparse
import os
import shutil

import numpy as np

from mindspore import Tensor, nn, Model, context
from mindspore.nn import Accuracy
from mindspore.train.callback import LossMonitor, TimeMonitor
from mindspore.train.serialization import load_param_into_net, load_checkpoint
from mindspore.communication.management import init, get_rank
from mindspore.context import ParallelMode

from mindspore.train.callback import CheckpointConfig, ModelCheckpoint
# from src.callback import CheckpointConfig, ModelCheckpoint
from src.config import dpcnn_cfg
from src.dataset import create_dataset
from src.dpcnn import DPCNN
from src.ema import ModelEMA
from src.lr_scheduler import CosineAnnealingLR

# CUDA_VISIBLE_DEVICES=1 python train.py --device_target GPU --data_path ./resource/rt-polaritydata --glove_path ./resource/glove.6B 
# --pretrained pretrained/aclimdb.ckpt

# 数据增强 两种替换词+换序
# TODO: 学习率衰减
# TODO: 预训练
import mindspore.dataset.core.config as config  # ds.config 或 ds.core.config
config.set_num_parallel_workers(2)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MindSpore LSTM Example')
    parser.add_argument('--data_path', type=str, default="./resource/rt-polaritydata", help='path where the dataset is stored.')
    parser.add_argument('--glove_path', type=str, default="./glove", help='path where the GloVe is stored.')
    parser.add_argument('--ckpt_path', type=str, default="./outputs", help='the path to save the checkpoint file.')
    parser.add_argument('--pretrained', type=str, default=None, help='the pretrained checkpoint file path.')
    parser.add_argument('--device_target', type=str, default="CPU", choices=['GPU', 'CPU'], help='the target device to run, support "GPU", "CPU". Default: "CPU".')
    args = parser.parse_args()

    context.set_context(
        # mode=context.GRAPH_MODE,
        mode=context.PYNATIVE_MODE,
        save_graphs=False,
        device_target=args.device_target)


    if os.path.exists(args.ckpt_path):
        shutil.rmtree(args.ckpt_path)

    cfg = dpcnn_cfg


    ds_train,vocab_size = create_dataset(cfg.batch_size, args.data_path, args.glove_path, cfg.seq_len, cfg.embed_size, is_train=True)

    
    weight_path = os.path.join(args.data_path, 'processed', 'weight_'+str(cfg.embed_size)+'d.txt')
    network = DPCNN(seq_len=cfg.seq_len,
                    embed_size=cfg.embed_size,
                    hid_channels=cfg.hid_channels,
                    kernelsize=cfg.kernelsize,
                    num_classes=cfg.num_classes,
                    weight_path = weight_path,
                    vocab_size = vocab_size)

    # pretrained
    if args.pretrained:
        ckpt = load_checkpoint(args.pretrained)
        load_param_into_net(network, ckpt)

    # ema = ModelEMA(network)

    loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
    


    # lr_scheduler = CosineAnnealingLR(cfg.learning_rate, cfg.num_epochs, ds_train.get_dataset_size(), cfg.num_epochs, warmup_epochs=0, eta_min=cfg.learning_rate/10)
    # lr_schedule = lr_scheduler.get_lr()
    # optimizer = nn.Momentum(network.trainable_params(), learning_rate=Tensor(lr_schedule), momentum=cfg.momentum, weight_decay=1e-4) # 0.9 
    
    optimizer = nn.Momentum(network.trainable_params(), cfg.learning_rate, cfg.momentum, weight_decay=1e-6)
    # optimizer = nn.Adam( network.trainable_params() , cfg.learning_rate , beta1=0.9 , beta2=0.999 , eps=1e-8 , weight_decay=1e-6)
    # optimizer = nn.SGD(network.trainable_params(), cfg.learning_rate, cfg.momentum, weight_decay=1e-6)


    model = Model(network, loss, optimizer, {'acc': Accuracy()})

    print("============== Starting Training ==============")
    config_ck = CheckpointConfig(save_checkpoint_steps=ds_train.get_dataset_size()*2,
                                 keep_checkpoint_max=cfg.keep_checkpoint_max)
    ckpoint_cb = ModelCheckpoint(prefix="dpcnn", directory=args.ckpt_path, config=config_ck) #,ema=ema
    time_cb = TimeMonitor(data_size=ds_train.get_dataset_size())
    loss_cb = LossMonitor()
    if args.device_target == "CPU":
        model.train(cfg.num_epochs, ds_train, callbacks=[time_cb, loss_cb, ckpoint_cb], dataset_sink_mode=False)
    else:
        model.train(cfg.num_epochs, ds_train, callbacks=[time_cb, loss_cb, ckpoint_cb])
    print("============== Training Success ==============")
    
