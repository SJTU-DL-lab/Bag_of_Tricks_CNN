import torch
from torch.optim import SGD, lr_scheduler
from configs.base_config import args

lr_lambda = lambda num:
lr_scheduler.LambdaLR(optimizer, lr_lambda)
