import torch
import torch.nn as nn
import torch.nn.functional as F


class Bottleneck(nn.Module):

    def __init__(self, in_channels, out_channels, downsample=False, tweak_type='A'):
        super(Bottleneck, self).__init__()
        mid_channels = in_channels // 2
        self.relu = nn.ReLU()
        if downsample and tweak_type == 'A':
            self.residual = nn.Sequential(
                            nn.Conv2d(in_channels, out_channels, 1, 2),
                            nn.BatchNorm2d(out_channels)
                            )
            self.bottleneck = nn.Sequential(
                               nn.Conv2d(in_channels, mid_channels, 1, 2),
                               nn.BatchNorm2d(mid_channels),
                               nn.ReLU(),
                               nn.Conv2d(mid_channels, mid_channels, 3, 1, 1),
                               nn.BatchNorm2d(mid_channels),
                               nn.ReLU(),
                               nn.Conv2d(mid_channels, out_channels, 1, 1),
                               nn.BatchNorm2d(out_channels)
                              )

        elif downsample and tweak_type == 'B':
            self.residual = nn.Sequential(
                            nn.Conv2d(in_channels, out_channels, 1, 2),
                            nn.BatchNorm2d(out_channels)
                            )
            self.bottleneck = nn.Sequential(
                               nn.Conv2d(in_channels, mid_channels, 1, 1),
                               nn.BatchNorm2d(mid_channels),
                               nn.ReLU(),
                               nn.Conv2d(mid_channels, mid_channels, 3, 2, 1),
                               nn.BatchNorm2d(mid_channels),
                               nn.ReLU(),
                               nn.Conv2d(mid_channels, out_channels, 1, 1),
                               nn.BatchNorm2d(out_channels)
                              )

        elif downsample and (tweak_type == 'D' or tweak_type == 'E'):
            self.residual = nn.Sequential(
                            nn.AvgPool2d(2, 2),
                            nn.Conv2d(in_channels, out_channels, 1, 1),
                            nn.BatchNorm2d(out_channels)
                            )
            self.bottleneck = nn.Sequential(
                               nn.Conv2d(in_channels, mid_channels, 1, 1),
                               nn.BatchNorm2d(mid_channels),
                               nn.ReLU(),
                               nn.Conv2d(mid_channels, mid_channels, 3, 2, 1),
                               nn.BatchNorm2d(mid_channels),
                               nn.ReLU(),
                               nn.Conv2d(mid_channels, out_channels, 1, 1),
                               nn.BatchNorm2d(out_channels)
                              )

        else:
            self.residual = nn.Sequential(
                            nn.Conv2d(in_channels, out_channels, 1),
                            nn.BatchNorm2d(out_channels)
                            )
            self.bottleneck = nn.Sequential(
                               nn.Conv2d(in_channels, mid_channels, 1, 1),
                               nn.BatchNorm2d(mid_channels),
                               nn.ReLU(),
                               nn.Conv2d(mid_channels, mid_channels, 3, 1, 1),
                               nn.BatchNorm2d(mid_channels),
                               nn.ReLU(),
                               nn.Conv2d(mid_channels, out_channels, 1, 1),
                               nn.BatchNorm2d(out_channels)
                              )

    def forward(self, x):
        output = self.bottleneck(x)
        residual_x = self.residual(x)
        output += residual_x
        output = self.relu(self.relu)

        return output


class Resnet50(nn.Module):

    def __init__(self, in_channels=3, num_classes=10,
                 input_size=224, tweak_type='A'):
        super(Resnet50, self).__init__()

        if tweak_type == 'C' or tweak_type == 'E':
            self.downsample = nn.Sequential(
                              nn.Conv2d(in_channels, 32, 3, 2, 1),
                              nn.BatchNorm2d(32),
                              nn.ReLU(),
                              nn.Conv2d(32, 32, 3, 1, 1),
                              nn.BatchNorm2d(32),
                              nn.ReLU(),
                              nn.Conv2d(32, 64, 3, 1, 1),
                              nn.BatchNorm2d(64),
                              nn.ReLU(),
                              nn.MaxPool2d(3, 2, 1)
                              )

        else:
            self.downsample = nn.Sequential(
                              nn.Conv2d(in_channels, out_channels=64,
                                        kernel_size=7, stride=2, padding=3),
                              nn.BatchNorm2d(64),
                              nn.ReLU(),
                              nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
                              )

        self.stages = self.stage_block(Bottleneck, 64, 256, 3, False, tweak_type)  # stage1
        self.stages += self.stage_block(Bottleneck, 256, 512, 4, tweak_type)   # stage2
        self.stages += self.stage_block(Bottleneck, 512, 1024, 6, tweak_type)   # stage3
        self.stages += self.stage_block(Bottleneck, 1024, 2048, 3, tweak_type)  # stage4

        self.stages = nn.Sequential(*self.stages)

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(2048, num_classes)

    def stage_block(self, model_block, in_channels, out_channels,
                    num_repeat, downsample=True, tweak_type='A'):
        stage = [model_block(in_channels, out_channels, downsample=downsample, tweak_type=tweak_type)]
        for i in range(num_repeat - 1):
            stage += [model_block(out_channels, out_channels, tweak_type=tweak_type)]

        return stage

    def forward(self, x):
        x = self.downsample(x)
        x = self.stages(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x
