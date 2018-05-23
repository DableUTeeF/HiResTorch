import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torch.nn.functional as F


class Block(nn.Module):
    def __init__(self, in_planes, plane):
        super(Block, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, plane*4, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(plane*4)
        self.conv2 = nn.Conv2d(plane*4, plane, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(plane)
        self.conv3 = nn.Conv2d(plane, plane, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(plane)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = F.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        # out = F.relu(out)
        return out


class BasicBlock(nn.Module):
    def __init__(self, in_planes, plane):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, plane, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(plane)
        self.conv2 = nn.Conv2d(plane, plane, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(plane)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        # out = F.relu(out)
        return out


class HiResA(nn.Module):
    def __init__(self, block, num_blocks, initial_kernal=64, num_classes=10):
        super(HiResA, self).__init__()
        self.num_blocks = num_blocks
        self.conv1 = nn.Conv2d(3, initial_kernal, kernel_size=5, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.layer1 = []
        self.layer2 = []
        self.layer3 = []
        for _ in range(num_blocks[0]):
            self.layer1.append(block(initial_kernal, [initial_kernal*4, initial_kernal, initial_kernal]))
        for _ in range(num_blocks[1]):
            self.layer2.append(block(initial_kernal * 2, [initial_kernal*4 if _ == 0 else initial_kernal*8
                , initial_kernal*2, initial_kernal*2]))
        for _ in range(num_blocks[2]):
            self.layer3.append(block(initial_kernal * 4, [initial_kernal*8 if _ == 0 else initial_kernal*16
                , initial_kernal*4, initial_kernal*4]))
        self.linear = nn.Linear(initial_kernal*4, num_classes)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        path = F.avg_pool2d(out, 28)
        for i in range(self.num_blocks[0] - 1):
            out = self.layer1[i](out)
            path += F.avg_pool2d(out, 28)

        out = self.layer1[-1](out)
        path = torch.cat((path, F.avg_pool2d(out, 28)), 1)
        out = F.max_pool2d(out, 2)

        for i in range(self.num_blocks[1] - 1):
            out = self.layer2[i](out)
            path += F.avg_pool2d(out, 14)
        out = self.layer2[-1](out)
        path = torch.cat((path, F.avg_pool2d(out, 14)), 1)
        out = F.max_pool2d(out, 2)

        for i in range(self.num_blocks[1] - 1):
            out = self.layer3[i](out)
            path += F.avg_pool2d(out, 7)
        out = self.layer3[-1](out)
        path = torch.cat((path, F.avg_pool2d(out, 7)), 1)

        out = path.view(path.size(0), -1)
        out = self.linear(out)
        return out


class HiResC(nn.Module):
    def __init__(self, block, num_blocks, initial_kernal=64, num_classes=10):
        super(HiResC, self).__init__()
        self.num_blocks = num_blocks
        self.conv1 = nn.Conv2d(3, initial_kernal, kernel_size=5, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(initial_kernal)
        self.layer1 = []
        self.layer2 = []
        self.layer3 = []
        for _ in range(num_blocks[0]):
            self.layer1.append(block(initial_kernal, initial_kernal))
        for _ in range(num_blocks[1]):
            self.layer2.append(block(initial_kernal if _ == 0 else initial_kernal*2, initial_kernal*2))
        for _ in range(num_blocks[2]):
            self.layer3.append(block(initial_kernal*2 if _ == 0 else initial_kernal*4, initial_kernal*4))
        self.linear = nn.Linear(initial_kernal*4, num_classes)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        path = F.avg_pool2d(out, 28)
        for i in range(self.num_blocks[0] - 1):
            out = self.layer1[i].cuda()(out)
            path += F.avg_pool2d(out, 28)
            out = F.relu(out)
            path = F.relu(path)

        out = self.layer1[-1].cuda()(out)
        path = torch.cat((path, F.avg_pool2d(out, 28)), 1)
        out = F.relu(out)
        path = F.relu(path)
        out = F.max_pool2d(out, 2)

        for i in range(self.num_blocks[1] - 1):
            out = self.layer2[i].cuda()(out)
            path += F.avg_pool2d(out, 14)
            out = F.relu(out)
            path = F.relu(path)

        out = self.layer2[-1].cuda()(out)
        path = torch.cat((path, F.avg_pool2d(out, 14)), 1)
        out = F.relu(out)
        path = F.relu(path)
        out = F.max_pool2d(out, 2)

        for i in range(self.num_blocks[1] - 1):
            out = self.layer3[i].cuda()(out)
            path += F.avg_pool2d(out, 7)
            out = F.relu(out)
            path = F.relu(path)

        out = self.layer3[-1].cuda()(out)
        path += F.avg_pool2d(out, 7)
        path = F.relu(path)

        out = path.view(path.size(0), -1)
        out = self.linear(out)
        return out
