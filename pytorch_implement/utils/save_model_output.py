import torch
import torch.nn as nn
import argparse
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10, CIFAR100


DATASET_LIST = ['CIFAR10', 'CIFAR100']
MODEL_LIST = ['resnet', ]

parser = argparse.ArgumentParser(description='Save model output for knowledge distillation')
parser.add_argument("--dataset", required=True, choices=['CIFAR10', 'CIFAR100'])
parser.add_argument("--dataroot", required=True, help='path to images')
parser.add_argument("--model", default='resnet')
parser.add_argument("--modelpkl", required=True, help='path to restore the model')

args = parser.parse_args()
dataset_type = args.dataset
model_type = args.model
CIFAR_transform = 

if dataset_type == 'CIFAR10':
    dataset = CIFAR10(args.dataroot, train=True, transform=)
if dataset_type not in DATASET_LIST:
    raise NotImplementedError('dataset type {} is not found'.format(args.dataset))
if model_type not in MODEL_LIST:
    raise NotImplementedError('model type {} is not found'.format(args.models))
