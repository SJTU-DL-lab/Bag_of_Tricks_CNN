import torch
import torch.nn as nn
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
import sys
import datetime
import time
import copy
import os
import pickle as pkl
import torchvision
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
from models.resnet_cifar import Resnet50
from models.network_util import get_scheduler, init_net
from tensorboardX import SummaryWriter
from configs.base_config import args

data_transform = {
    'train': transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(args.input_size, 4),
        transforms.ToTensor()
    ]),
    'test': transforms.Compose([
        transforms.ToTensor()
    ])
}

dataset = {
    'train': CIFAR10(args.dataroot, train=True,
                     transform=data_transform['train'],
                     download=True),
    'test': CIFAR10(args.dataroot, train=False,
                    transform=data_transform['test'],
                    download=True)
    }

dataloader = {x: DataLoader(dataset[x],
                            batch_size=args.batch_size,
                            shuffle=args.shuffle) for x in ['train', 'test']}

writer = SummaryWriter(log_dir=args.summary_dir)
with open(os.path.join(args.summary_dir, 'args.pkl'), 'wb') as f:
    pkl.dump(args, f)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = Resnet50(args.stage_channels, args.in_channels,
                 args.num_classes, args.tweak_type,
                 args.num_repeat)
model = init_net(model, init_type=args.init_type,
                 init_gain=args.init_gain, gpu_ids=args.gpu_ids)

loss_func = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(),
                            lr=args.lr,
                            momentum=args.momentum,
                            weight_decay=args.weight_decay)
if args.lr_warmup_type is not None:
    lr_lambda = lambda num: (num+1) / args.lr_warmup_iters if num <= args.lr_warmup_iters else args.lr_warmup_iters
    lr_scheduler_warmup = lr_scheduler.LambdaLR(optimizer, lr_lambda)
lr_scheduler = get_scheduler(optimizer, args)

val_acc_history = []
best_model_wts = copy.deepcopy(model.state_dict())
best_acc = 0.0
num_iters = 1
since = time.time()

for ep in range(args.epoch):
    print()
    print("epoch {}/{}".format(ep+1, args.epoch))
    print("-" * 10)
    if args.lr_warmup_type == 'epoch' and ep < args.lr_warmup_iters:
        lr_scheduler_warmup.step()
    else:
        lr_scheduler.step()

    for stage in ['train', 'test']:
        if stage == 'train':
            model.train()
        else:
            model.eval()
        running_loss = 0.0
        running_corrects = 0.0

        for i, (X, y) in enumerate(dataloader[stage]):
            X = X.to(device)
            y = y.to(device)
            optimizer.zero_grad()

            with torch.set_grad_enabled(stage == 'train'):
                y_score = model(X)
                loss = loss_func(y_score, y)
                _, y_pred = torch.max(y_score, 1)

                if stage == 'train':
                    if args.lr_warmup_type == 'iter' and num_iters <= args.lr_warmup_iters:
                        lr_scheduler_warmup.step()
                    loss.backward()
                    optimizer.step()

            step_corrects = torch.sum(y_pred == y.data)
            running_loss += loss.item() * X.size(0)
            running_corrects += step_corrects
            if stage == 'train':
                writer.add_scalar('train/running_loss', loss.item(), num_iters)
                writer.add_scalar('train/running_acc', step_corrects.double() / args.batch_size, num_iters)
                writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], num_iters)
                num_iters += 1

        epoch_loss = running_loss / len(dataloader[stage].dataset)
        epoch_acc = running_corrects.double() / len(dataloader[stage].dataset)
        writer.add_scalar('{}/epoch_loss'.format(stage), epoch_loss, ep+1)
        writer.add_scalar('{}/epoch_acc'.format(stage), epoch_acc, ep+1)

        print('{} Loss: {:.4f}, acc: {:.4f}'.format(stage, epoch_loss, epoch_acc))

        if stage == 'test' and epoch_acc > best_acc:
            best_acc = epoch_acc
            best_model_wts = copy.deepcopy(model.state_dict())

time_elapsed = time.time() - since
print('Training complete in {:.0f}h {:.0f}m {:.0f}s'.format(time_elapsed // 3600, time_elapsed // 60, time_elapsed % 60))
print('Best val Acc: {:4f}'.format(best_acc))

with open(os.path.join(args.summary_dir, 'model_bestValACC_{:.3f}.pkl'.format(best_acc)), 'wb') as f:
    pkl.dump(best_model_wts, f)
