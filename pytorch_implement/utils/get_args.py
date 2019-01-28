import argparse
import os
import pickle as pkl


def print_argsions(args, save_dir):
    message = ''
    message += '----------------- argsions ---------------\n'
    for k, v in sorted(vars(args).items()):
        message += '{:>25}: {:<30}\n'.format(str(k), str(v))
    message += '----------------- End -------------------'
    print(message)

    # save to the disk
    expr_dir = save_dir
    file_name = os.path.join(expr_dir, 'args.txt')
    with open(file_name, 'wt') as args_file:
        args_file.write(message)
        args_file.write('\n')


parser = argparse.ArgumentParser(description='Print and save arguments')
parser.add_argument('--arg_dir', required=True)

arg_dir = parser.parse_args()
basename = os.path.dirname(arg_dir.arg_dir)
with open(arg_dir.arg_dir, 'rb') as f:
    args = pkl.load(f)
print_argsions(args, basename)
