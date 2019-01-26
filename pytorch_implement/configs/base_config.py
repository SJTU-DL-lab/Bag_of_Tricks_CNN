import argparse
import os
import datetime

time_now = datetime.datetime.now()
time_str = '{}-{}-{}'.format(str(time_now.date()), time_now.hour, time_now.minute)
parser = argparse.ArgumentParser(description="Initialize parameters")

parser.add_argument("--input_size", default=(32, 32))
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--epochs", type=int, default=10)
parser.add_argument("--lr", type=float, default=0.1)
parser.add_argument("--lr_decay_iters", default=32e3)
parser.add_argument("--lr_policy", type=str, default='step')
parser.add_argument("--momentum", type=float, default=0.9)
parser.add_argument("--weight_decay", type=float, default=1e-4)
parser.add_argument("--gpu_ids", type=list, default=[])
parser.add_argument("--niter", type=int, default=9e3)
parser.add_argument("--shuffle", action='store_true')
parser.add_argument("--stage_channels", default=[16, 32, 64])
parser.add_argument("in_channels", type=int, default=3)
parser.add_argument("num_repeat", type=int, default=9, help="the repeat numbers of stage block, this only works in cifar resnet")
parser.add_argumnet("num_classes", type=int, default=10)
parser.add_argument("--tweak_type", type=str, default='A')
parser.add_argument("--dataset", default='cifar', choices=['cifar', 'imagenet'])
args = parser.parse_args()

parser.add_argument("--summary_dir", type=str, default='./summary/resnet50_{}_{}'.format(args.dataset, time_str))
args = parser.parse_args()


def print_options(self, opt):
    message = ''
    message += '----------------- Options ---------------\n'
    for k, v in sorted(vars(opt).items()):
        comment = ''
        default = self.parser.get_default(k)
        if v != default:
            comment = '\t[default: %s]' % str(default)
        message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
    message += '----------------- End -------------------'
    print(message)

    # save to the disk
    expr_dir = os.path.join(opt.checkpoints_dir, opt.name)
    util.mkdirs(expr_dir)
    file_name = os.path.join(expr_dir, 'opt.txt')
    with open(file_name, 'wt') as opt_file:
        opt_file.write(message)
        opt_file.write('\n')
