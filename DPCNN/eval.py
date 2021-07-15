
import argparse
import os

import numpy as np


from mindspore import Tensor, nn, Model, context
from mindspore.nn import Accuracy, Recall, F1
from mindspore.train.serialization import load_checkpoint, load_param_into_net

from src.config import dpcnn_cfg
from src.dataset import create_dataset
from src.dpcnn import DPCNN



# CUDA_VISIBLE_DEVICES=0 python eval.py --device_target GPU --data_path ./resource/rt-polaritydata --glove_path ./resource/glove.6B --ckpt_path outputs/dpcnn-20_149.ckpt

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MindSpore LSTM Example')
    parser.add_argument('--data_path', type=str, default="./resource/rt-polaritydata", help='path where the dataset is stored.')
    parser.add_argument('--glove_path', type=str, default="./glove", help='path where the GloVe is stored.')
    parser.add_argument('--ckpt_path', type=str, default=None, help='the checkpoint file path used to evaluate model.')
    parser.add_argument('--device_target', type=str, default="CPU", choices=['GPU', 'CPU'], help='the target device to run, support "GPU", "CPU". Default: "CPU".')
    args = parser.parse_args()

    context.set_context(
        mode=context.PYNATIVE_MODE, # GRAPH_MODE
        save_graphs=False,
        device_target=args.device_target)

    
    cfg = dpcnn_cfg

    ds_eval,vocab_size = create_dataset(cfg.batch_size, args.data_path, args.glove_path, cfg.seq_len, cfg.embed_size, is_train=False)

    weight_path = os.path.join(args.data_path, 'processed', 'weight_'+str(cfg.embed_size)+'d.txt')
    network = DPCNN(seq_len=cfg.seq_len,
                    embed_size=cfg.embed_size,
                    hid_channels=cfg.hid_channels,
                    kernelsize=cfg.kernelsize,
                    num_classes=cfg.num_classes,
                    weight_path = weight_path,
                    vocab_size = vocab_size)

    loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')

    model = Model(network, loss, metrics={'acc': Accuracy(), 'recall': Recall(), 'f1': F1()})

    print("============== Starting Testing ==============")
    param_dict = load_checkpoint(args.ckpt_path)
    load_param_into_net(network, param_dict)
    if args.device_target == "CPU":
        acc = model.eval(ds_eval, dataset_sink_mode=False)
    else:
        acc = model.eval(ds_eval)
    print("============== {} ==============".format(acc))
