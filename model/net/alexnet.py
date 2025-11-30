"""
AlexNet.py

A Pytorch implementation of the AlexNet deep convolutional neural network

Author: ZhangXin
Date: 2025/11/04
"""

import datetime
import os
import json
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import (
    transforms,
    datasets
)
import tqdm

class AlexNet(nn.Module):
    """
    AlexNet模型实现类
    """

    def __init__(self, in_channels, out_features):
        super(AlexNet, self).__init__()

        # first convolutional layer
        self.layer_1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels,
                      out_channels=96,
                      kernel_size=11,
                      stride=4,
                      padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3,
                         stride=2,
                         padding=0)
        )

        # second convolutional layer
        self.layer_2 = nn.Sequential(
            nn.Conv2d(in_channels=96,
                      out_channels=256,
                      kernel_size=5,
                      stride=1,
                      padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3,
                         stride=2,
                         padding=0)
        )

        # third convolutional layer
        self.layer_3 = nn.Sequential(
            nn.Conv2d(in_channels=256,
                      out_channels=384,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU()
        )

        # fourth convolutional layer
        self.layer_4 = nn.Sequential(
            nn.Conv2d(in_channels=384,
                      out_channels=384,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU()
        )

        # fifth convolutional layer
        self.layer_5 = nn.Sequential(
            nn.Conv2d(in_channels=384,
                      out_channels=256,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3,
                         stride=2,
                         padding=0)
        )

        # fully-connected layer
        self.layer_6 = nn.Sequential(
            nn.Linear(in_features=9216,
                      out_features=4096),
            nn.ReLU(),
            nn.Dropout()
        )

        # fully-connected layer
        self.layer_7 = nn.Sequential(
            nn.Linear(in_features=4096,
                      out_features=4096),
            nn.ReLU(),
            nn.Dropout()
        )

        # fully-connected layer
        self.classifier = nn.Linear(in_features=4096,
                                 out_features=out_features)

        self.initialize()

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(tensor=m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.Linear):
                nn.init.normal_(tensor=m.weight, mean=0, std=0.01)
                nn.init.constant_(tensor=m.bias, val=0)

    def forward(self, x):

        x = self.layer_1(x)
        x = self.layer_2(x)
        x = self.layer_3(x)
        x = self.layer_4(x)
        x = self.layer_5(x)
        x = torch.flatten(x, start_dim=1)
        x = self.layer_6(x)
        x = self.layer_7(x)

        return self.classifier(x)

def train():

    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=4, help="size of batches")
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
    args = parser.parse_args()

    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    print(device)

    weight_path = os.path.join("checkpoints/net", "alexnet.pth")

    image_path = os.path.join("assets", "cifar-10")

    transformer = transforms.Compose([
        transforms.RandomResizedCrop([227, 227]),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    # 数据集准备
    image_set = datasets.CIFAR10(root=image_path, train=True, download=False, transform=transformer)

    class_dict = dict((index, classes) for (classes, index) in image_set.class_to_idx.items())

    with open("class_indices.json", "w") as file:
        class_str = json.dumps(class_dict, indent=4)
        file.write(class_str)

    image_loader = DataLoader(dataset=image_set, batch_size=args.batch_size, shuffle=True, num_workers=0)

    net = AlexNet(in_channels=3, out_features=10).to(device)
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

    val_path = os.path.join("assets", "cifar-10")

    transformer = transforms.Compose([
        transforms.Resize((227, 227)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    data_set = datasets.CIFAR10(root=val_path, train=False, transform=transformer, download=False)

    data_loader = DataLoader(dataset=data_set, batch_size=2, shuffle=True, num_workers=0)

    checkpoints_path = os.path.join("checkpoints", "net/alexnet.pth")

    net = AlexNet(3, 10).to(device)
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



