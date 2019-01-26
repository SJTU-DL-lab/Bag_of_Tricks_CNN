import argparse
import os
import datetime
import torch

time_now = datetime.datetime.now()
time_str = '{}-{}-{}'.format(str(time_now.date()), time_now.hour, time_now.minute)
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
parser.add_argument("--gpu_ids", type=str, default='0')
parser.add_argument("--niter", type=int, default=9e3)
parser.add_argument("--shuffle", action='store_true')
parser.add_argument("--stage_channels", default=[16, 32, 64])
parser.add_argument("--in_channels", type=int, default=3)
parser.add_argument("--num_repeat", type=int, default=9, help="the repeat numbers of stage block, this only works in cifar resnet")
parser.add_argument("--num_classes", type=int, default=10)
parser.add_argument("--tweak_type", type=str, default='A')
parser.add_argument("--dataset", default='cifar', choices=['cifar', 'imagenet'])
args = parser.parse_args()

parser.add_argument("--summary_dir", type=str, default='./summary/resnet50_{}_{}'.format(args.dataset, time_str))
args = parser.parse_args()

# set gpu ids
str_ids = args.gpu_ids.split(',')
args.gpu_ids = []
for str_id in str_ids:
    id = int(str_id)
    if id >= 0:
        args.gpu_ids.append(id)
if len(args.gpu_ids) > 0:
    torch.cuda.set_device(args.gpu_ids[0])


def print_argsions(args):
    message = ''
    message += '----------------- argsions ---------------\n'
    for k, v in sorted(vars(args).items()):
        comment = ''
        default = self.parser.get_default(k)
        if v != default:
            comment = '\t[default: %s]' % str(default)
        message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
    message += '----------------- End -------------------'
    print(message)

    # save to the disk
    expr_dir = os.path.join(args.checkpoints_dir, args.name)
    util.mkdirs(expr_dir)
    file_name = os.path.join(expr_dir, 'args.txt')
    with open(file_name, 'wt') as args_file:
        args_file.write(message)
        args_file.write('\n')
