"""
resnet.py

A Pytorch implementation of the ResNet

Author: ZhangXin
Date: 2025/11/7
"""

import datetime
import argparse
import os
import json

import tqdm
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import (
    transforms,
    datasets
)

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

class ResNet(nn.Module):
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

        super(ResNet, self).__init__()
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
    return ResNet(block=BasicBlock, block_num=[3, 4, 6, 3], num_classes=num_classes, include_top=include_top)

def resnet101(num_classes=1000, include_top=True):
    return ResNet(block=Bottleneck, block_num=[3, 4, 23, 3], num_classes=num_classes, include_top=include_top)

def train():

    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=4, help="size of batches")
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
    args = parser.parse_args()

    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    print(device)

    weight_path = os.path.join("checkpoints/net", "alexnet.pth")

    image_path = os.path.join("assets", "flower_photos/train")

    transformer = transforms.Compose([
        transforms.RandomResizedCrop([227, 227]),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    # 数据集准备
    image_set = datasets.ImageFolder(image_path, transform=transformer)

    class_dict = dict((index, classes) for (classes, index) in image_set.class_to_idx.items())

    with open("class_indices.json", "w") as file:
        class_str = json.dumps(class_dict, indent=4)
        file.write(class_str)

    image_loader = DataLoader(dataset=image_set, batch_size=args.batch_size, shuffle=True, num_workers=0)

    net = resnet34(num_classes=5).to(device)
    # load weight
    if os.path.exists(weight_path):
        net.load_state_dict(torch.load(weight_path))
        print("load weight successfully")
    else:
        print("load weight fail")

    optimizer = optim.Adam(net.parameters(), lr=args.lr)
    # loss function
    criterion = nn.CrossEntropyLoss()

    epoch = 0

    start_time = datetime.datetime.now()

    while True:
        running_loss = 0.0
        for i, (image, label) in enumerate(tqdm.tqdm(image_loader)):
            net.train()
            image = image.to(device)
            label = label.to(device)
            optimizer.zero_grad()
            output = net(image)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"{datetime.datetime.now() - start_time} epoch: {epoch} train_loss: {running_loss/len(image_loader)}")
        epoch += 1
        torch.save(net.state_dict(), weight_path)

def predict():

    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    print(device)

    val_path = os.path.join("assets", "flower_photos/train")

    transformer = transforms.Compose([
        transforms.Resize((227, 227)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    data_set = datasets.ImageFolder(val_path, transformer)

    data_loader = DataLoader(dataset=data_set, batch_size=2, shuffle=True, num_workers=0)

    checkpoints_path = os.path.join("checkpoints", "net/resnet34.pth")

    net = resnet34(num_classes=5).to(device)
    if os.path.exists(checkpoints_path):
        net.load_state_dict(torch.load(checkpoints_path))
        print("load weight successfully")
    else:
        print("load weight fail")

    prediction_log = open("checkpoints_path.txt", "w")

    with open("class_indices.json", "r") as file:
        class_index = json.load(file)

    correct_count = 0
    total_samples = len(data_set)

    for images, labels in tqdm.tqdm(data_loader):
        net.eval()
        with torch.no_grad():
            images = images.to(device)
            labels = labels.to(device)
            output = net(images)
            prediction = torch.argmax(output, dim=1)
            correct_count += (prediction == labels).sum().item()
            for i in range(len(output)):
                if (prediction == labels)[i]:
                    prediction_log.write(f"{class_index[str(labels[i].item())]}  {class_index[str(prediction[i].item())]}  True\n")
                else:
                    prediction_log.write(f"{class_index[str(labels[i].item())]}  {class_index[str(prediction[i].item())]}  False\n")
    prediction_log.write(f"right classify percent: {correct_count / total_samples}")

    print(f"right classify percent: {correct_count / total_samples}")

def main():

    predict()

if __name__ == "__main__":
    main()