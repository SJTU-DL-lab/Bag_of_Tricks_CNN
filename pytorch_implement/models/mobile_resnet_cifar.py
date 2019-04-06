import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict


class BuildingBlock(nn.Module):

    def __init__(self, in_channels, out_channels,
                 downsample=False, expand_ratio=3, tweak_type='A'):
        super(BuildingBlock, self).__init__()
        # mid_channels = in_channels // 2
        stride = 2 if downsample else 1
        hidden_channels = round(in_channels * expand_ratio)
        self.relu = nn.ReLU()
        self.downsample = downsample
        self.residual = nn.Sequential(
                        OrderedDict([
                            ('res_avgPool', nn.AvgPool2d(2, 2)),
                            ('res_conv', nn.Conv2d(in_channels, out_channels, 1, 1)),
                            ('res_bn', nn.BatchNorm2d(out_channels))
                                    ])
                        )
        self.build_block = nn.Sequential(
                           OrderedDict([
                               ('conv1', nn.Conv2d(in_channels, hidden_channels, 1, 1, 0)),
                               ('bn1', nn.BatchNorm2d(hidden_channels)),
                               ('relu1', nn.ReLU()),

                               ('conv2', nn.Conv2d(hidden_channels, hidden_channels, 3, stride, 1,
                                                   groups=hidden_channels)),
                               ('bn2', nn.BatchNorm2d(hidden_channels)),
                               #('relu2', nn.ReLU()),

                               ('conv3', nn.Conv2d(hidden_channels, out_channels, 1, 1, 0)),
                               ('bn3', nn.BatchNorm2d(out_channels)),
                                       ])
                          )

    def forward(self, x):
        identity = x

        output = self.build_block(x)
        if self.downsample:
            identity = self.residual(x)
        output += identity

        return output


class Resnet50(nn.Module):

    def __init__(self, stage_channels=[16, 32, 64],
                 in_channels=3, num_classes=10, tweak_type='A',
                 num_repeat=9):
        super(Resnet50, self).__init__()

        self.first_layer = nn.Sequential(
                           OrderedDict([
                               ('first_layer_conv', nn.Conv2d(in_channels, stage_channels[0], 3, 1, 1)),
                               ('first_layer_bn', nn.BatchNorm2d(stage_channels[0])),
                               ('first_layer_relu', nn.ReLU())
                                       ])
                           )

        self.stages = []
        stage_channels.insert(0, stage_channels[0])

        for i in range(len(stage_channels)-1):
            if i == 0:
                downsample = False
            else:
                downsample = True
            self.stages += self.stage_block(BuildingBlock, stage_channels[i],
                                            stage_channels[i+1], num_repeat, downsample,
                                            tweak_type)

        self.stages = nn.Sequential(*self.stages)

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(stage_channels[-1], num_classes)

    def stage_block(self, model_block, in_channels, out_channels,
                    num_repeat, downsample=True, tweak_type='A'):
        stage = [model_block(in_channels, out_channels, downsample=downsample, tweak_type=tweak_type)]
        for i in range(num_repeat - 1):
            stage += [model_block(out_channels, out_channels, tweak_type=tweak_type)]

        return stage

    def forward(self, x):
        x = self.first_layer(x)
        x = self.stages(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x
