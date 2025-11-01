"""
resnet.py

考虑使用resnet用于DDPM预测噪声
目前用于Unet提升网络性能
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    """
    ResNet18/34基本残差单元实现
    """

    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        """
        初始化函数
        
        parameters
        -----
        in_channels: int
                    输入通道
        out_channels: int
                    输出通道
        stride: int
                卷积核步长
        downsample: class
                    对应的残差结构
        """

        super(BasicBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        """
        前向过程
        
        parameters
        -----
        x: tensor
            输入张量
        """

        identity = x    # 捷径输出值
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = out + identity
        out = self.relu(out)

        return out
    
class Bottleneck(nn.Module):
    """
    ResNet50/101/152残差基本单元实现
    """

    expansion = 4
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        """
        初始化函数
        
        parameters
        -----
        in_channels: int
                    输入通道数
        out_channels: int
                    输出通道数
        stride: int
                卷积核步长
        downsampel: class
                    对应的残差结构
        """

        super(Bottleneck, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1, stride=stride)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels * self.expansion, kernel_size=1, stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.downsample = downsample

    def forward(self, x):
        """
        前向过程
        
        parameters
        -----
        x: tensor
            输入张量
        """

        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out = out + identity
        out = self.relu(out)

        return out

class resnet(nn.Module):
    """
    resnet网络实现
    """

    def __init__(self, block, block_num, num_classes=1000, include_top=True):
        """
        初始化函数

        parameters
        -----
        block: 残差块
        block_num: list
                    残差块数量
        num_classes: int
                    分类数
        """

        super(resnet, self).__init__()
        self.include_top = include_top
        self.in_channels = 64

        self.stem_conv = nn.Sequential(
            nn.Conv2d(3, self.in_channels, kernel_size=7, padding=3, stride=2),
            nn.BatchNorm2d(self.in_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.layer1 = self._make_layer(block=block, channels=64, block_num=block_num[0], stride=1)
        self.layer2 = self._make_layer(block=block, channels=128, block_num=block_num[1], stride=2)
        self.layer3 = self._make_layer(block=block, channels=256, block_num=block_num[2], stride=2)
        self.layer4 = self._make_layer(block=block, channels=512, block_num=block_num[3], stride=2)

        if self.include_top:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')


    def _make_layer(self, block, channels, block_num, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, channels * block.expansion, kernel_size=1, stride=stride),
                nn.BatchNorm2d(channels * block.expansion)
            )
        
        layers = []
        layers.append(block(self.in_channels, channels, downsample=downsample, stride=stride))
        self.in_channels = channels * block.expansion

        for _ in range(1, block_num):
            layers.append(block(self.in_channels, channels))

        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.stem_conv(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        if self.include_top:
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)

        return x

def resnet34(num_classes=1000, include_top=True):
    return resnet(block=BasicBlock, block_num=[3, 4, 6, 3], num_classes=num_classes, include_top=include_top)

def resnet101(num_classes=1000, include_top=True):
    return resnet(block=Bottleneck, block_num=[3, 4, 23, 3], num_classes=num_classes, include_top=include_top)