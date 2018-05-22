import os
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torch.nn.functional as F


class AddBlock(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, blocks):
        super(AddBlock, self).__init__()
        self.block1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.block2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.block3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)



    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Block(nn.Module):
    def __init__(self, in_planes, planes):
        super(Block, self).__init__()
        plane1, plane2, plane3 = planes
        self.conv1 = nn.Conv2d(in_planes, plane1, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(plane1)
        self.conv2 = nn.Conv2d(plane1, plane2, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(plane2)
        self.conv3 = nn.Conv2d(plane2, plane3, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(plane3)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = F.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        out = F.relu(out)
        return out


class HiResA(nn.Module):
    def __init__(self, num_blocks, num_classes=10):
        super(HiResA, self).__init__()
        self.in_planes = 32

        self.conv1 = nn.Conv2d(3, 32, kernel_size=5, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.layer1 = self._make_layer(Block, [128, 32, 32], num_blocks[0])
        self.layer2 = self._make_layer(Block, [256, 64, 64], num_blocks[1])
        self.layer3 = self._make_layer(Block, [512, 128, 128], num_blocks[2])
        self.linear = nn.Linear(6272, num_classes)

    def _make_layer(self, block, planes, num_blocks):
        layers = []
        for _ in range(num_blocks):
            layers.append(block(self.in_planes, planes))
            self.in_planes = planes[-1]
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        path = F.avg_pool2d(out, 28)
        out = self.layer1(out)
        path = torch.cat((path, F.avg_pool2d(out, 28)), 1)
        # path += F.avg_pool2d(out, 28)

        out = F.max_pool2d(out, 2)
        out = self.layer2(out)
        path = torch.cat((path, F.avg_pool2d(out, 14)), 1)
        # path += F.avg_pool2d(out, 14)

        out = F.max_pool2d(out, 2)
        out = self.layer3(out)
        path = torch.cat((path, F.avg_pool2d(out, 7)), 1)
        # path += F.avg_pool2d(out, 7)

        # out = F.avg_pool2d(path, 4)
        out = out.view(path.size(0), -1)
        out = self.linear(out)
        return out
