from copy import deepcopy
import math

import mindspore
from mindspore import Parameter
from mindspore import Tensor

def copy_attr(a, b, include=(), exclude=()):
    # Copy attributes from b to a, options to only include [...] and to exclude [...]
    for k, v in b.__dict__.items():
        if (len(include) and k not in include) or k.startswith('_') or k in exclude:
            continue
        else:
            setattr(a, k, v)

class ModelEMA:
    """ Model Exponential Moving Average from https://github.com/rwightman/pytorch-image-models
    Keep a moving average of everything in the model state_dict (parameters and buffers).
    This is intended to allow functionality like
    https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage
    A smoothed version of the weights is necessary for some training schemes to perform well.
    This class is sensitive where it is initialized in the sequence of model init,
    GPU assignment and distributed training wrappers.
    """

    def __init__(self, model, decay=0.9999, updates=0):
        # Create EMA
        self.ema = deepcopy(model)  # FP32 EMA
        # if next(model.parameters()).device.type != 'cpu':
        #     self.ema.half()  # FP16 EMA
        self.updates = updates  # number of EMA updates
        self.decay = lambda x: decay * (1 - math.exp(-x / 2000))  # decay exponential ramp (to help early epochs)  10000
        for p in self.ema.get_parameters(expand=True):
            p.requires_grad = False


    def update(self, model):
        # Update EMA parameters
        self.updates += 1
        d = self.decay(self.updates)

        msd = model.parameters_dict()  # model state_dict
        for p in self.ema.get_parameters():
            if p.dtype in [mindspore.float16 , mindspore.half,mindspore.float32 , mindspore.single,mindspore.float64 , mindspore.double]:
                v = p*d
                m = msd[p.name].clone()  #Parameter(msd[k].asnumpy())
                m.requires_grad = False
                v += (1. - d) * m #msd[k].detach()

                p.set_data(v)

        # parameters_dict调用get_parameters调用parameters_and_names
        # net.parameters_dict() dict 拷贝的
        # net.get_parameters() iter 引用的
        # net.parameters_and_names() iter
        

    def update_attr(self, model, include=(), exclude=('process_group', 'reducer')):
        # Update EMA attributes
        copy_attr(self.ema, model, include, exclude)

from mindspore.train.callback._callback import Callback
class EMACheckpoint(Callback):
    def __init__(self, ema=None,ckpoint_cb=None):
        self.ema = ema
        self.ckpoint_cb =ckpoint_cb

    def step_end(self, run_context):
        self.ema.update(run_context.original_args().train_network)
        self.ckpoint_cb._config.saved_network = self.ema.ema
