import torch
import torch.nn as nn
import torch.nn.functional as F


class BuildingBlock(nn.Module):

    def __init__(self, in_channels, out_channels, downsample=False, tweak_type='A'):
        super(Bottleneck, self).__init__()
        # mid_channels = in_channels // 2

        if downsample and tweak_type == 'A':
            self.residual = nn.Sequential(
                            nn.Conv2d(in_channels, out_channels, 1, 2),
                            nn.BatchNorm2d(out_channels),
                            nn.ReLU()
                            )
            self.build_block = nn.Sequential(
                               nn.Conv2d(in_channels, out_channels, 3, 2, 1),
                               nn.BatchNorm2d(out_channels),
                               nn.ReLU(),
                               nn.Conv2d(out_channels, out_channels, 3, 1, 1),
                               nn.BatchNorm2d(out_channels),
                               nn.ReLU(),
                              )

        # elif downsample and tweak_type == 'B':
        #     self.residual = nn.Sequential(
        #                     nn.Conv2d(in_channels, out_channels, 1, 2),
        #                     nn.BatchNorm2d(out_channels),
        #                     nn.ReLU()
        #                     )
        #     self.build_block = nn.Sequential(
        #                        nn.Conv2d(in_channels, mid_channels, 1, 1),
        #                        nn.BatchNorm2d(mid_channels),
        #                        nn.ReLU(),
        #                        nn.Conv2d(mid_channels, mid_channels, 3, 2, 1),
        #                        nn.BatchNorm2d(mid_channels),
        #                        nn.ReLU(),
        #                        nn.Conv2d(mid_channels, out_channels, 1, 1),
        #                        nn.BatchNorm2d(out_channels),
        #                        nn.ReLU()
        #                       )

        elif downsample and (tweak_type == 'D' or tweak_type == 'E'):
            self.residual = nn.Sequential(
                            nn.AvgPool2d(2, 2),
                            nn.Conv2d(in_channels, out_channels, 1, 1),
                            nn.BatchNorm2d(out_channels),
                            nn.ReLU()
                            )
            self.build_block = nn.Sequential(
                               nn.Conv2d(in_channels, out_channels, 3, 2, 1),
                               nn.BatchNorm2d(out_channels),
                               nn.ReLU(),
                               nn.Conv2d(out_channels, out_channels, 3, 1, 1),
                               nn.BatchNorm2d(out_channels),
                               nn.ReLU(),
                              )

        else:
            self.residual = nn.Sequential(
                            nn.Conv2d(in_channels, out_channels, 1),
                            nn.BatchNorm2d(out_channels),
                            nn.ReLU()
                            )
            self.build_block = nn.Sequential(
                               nn.Conv2d(in_channels, out_channels, 3, 1, 1),
                               nn.BatchNorm2d(out_channels),
                               nn.ReLU(),
                               nn.Conv2d(out_channels, out_channels, 3, 1, 1),
                               nn.BatchNorm2d(out_channels),
                               nn.ReLU(),
                              )

    def forward(self, x):
        output = self.bottleneck(x)
        residual_x = self.residual(x)
        output += residual_x

        return output


class Resnet50(nn.Module):

    def __init__(self, stage_channels=[16, 32, 64],
                 in_channels=3, num_classes=10,
                 input_size=32, tweak_type='A'):
        super(Resnet50, self).__init__()

        self.first_layer = nn.Conv2d(in_channels, stage_channels[0], 3, 1, 1)
        self.stages = []
        stage_channels.insert(0, stage_channels[0])

        for i in range(len(stage_channels)-1):
            if i == 0:
                downsample = False
            else:
                downsample = True
            self.stages += self.stage_block(Bottleneck, stage_channels[i],
                                            stage_channels[i+1], 18, downsample,
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
