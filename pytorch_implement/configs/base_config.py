import argparse
import os
import datetime
import torch


parser = argparse.ArgumentParser(description="Initialize parameters")

parser.add_argument("--dataroot", required=True, help='path to images')
parser.add_argument("--input_size", default=(32, 32))
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--epoch", type=int, default=183)
parser.add_argument("--lr", type=float, default=0.1)
parser.add_argument("--lr_decay_iters", default=[91, 136])
parser.add_argument("--lr_policy", type=str, default='multistep')
parser.add_argument("--init_type", type=str, default='kaiming')
parser.add_argument("--init_gain", type=float, default=0.02)
parser.add_argument("--momentum", type=float, default=0.9)
parser.add_argument("--weight_decay", type=float, default=1e-4)
parser.add_argument("--lr_warmup_type", default=None, choices=['iter', 'epoch', None])
parser.add_argument("--lr_warmup_iters", type=int, default=5)
parser.add_argument("--lr_decay_type", default='epoch', choices=['iter', 'epoch'])
parser.add_argument("--gpu_ids", type=str, default='0')
parser.add_argument("--niter", type=int, default=178)
parser.add_argument("--no_shuffle", action='store_true')
parser.add_argument("--stage_channels", default=[16, 32, 64])
parser.add_argument("--in_channels", type=int, default=3)
parser.add_argument("--num_repeat", type=int, default=9, help="the repeat numbers of stage block, this only works in cifar resnet")
parser.add_argument("--num_classes", type=int, default=10)
parser.add_argument("--tweak_type", type=str, default='A')
parser.add_argument("--zero_gamma", action='store_true')
parser.add_argument("--no_bias_decay", action='store_true')
parser.add_argument("--label_smooth", action='store_true')
parser.add_argument("--mixup_alpha", type=float, default=-1)
parser.add_argument("--dataset", default='cifar', choices=['cifar', 'imagenet'])
parser.add_argument("--summary_dir", type=str, default='./summary/resnet50')
parser.add_argument("--add_stamp", action='store_true')
args = parser.parse_args()

# add dataset and time stamp to summary dir
if args.add_stamp:
    time_now = datetime.datetime.now()
    time_str = '{}-{}-{}'.format(str(time_now.date()), time_now.hour, time_now.minute)
    args.summary_dir += '_{}_{}'.format(args.dataset, time_str)

# set gpu ids
str_ids = args.gpu_ids.split(',')
args.gpu_ids = []
for str_id in str_ids:
    id = int(str_id)
    if id >= 0:
        args.gpu_ids.append(id)
if len(args.gpu_ids) > 0:
    torch.cuda.set_device(args.gpu_ids[0])
