"""
train.py

模型训练文件
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
import os
import json
from model.net import resnet

def main():

    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    print(device)
    # 模型保存路径
    save_path = "checkpoints/net/resnet101.pth"

    data_transform = {
        "train": transforms.Compose([
            transforms.RandomResizedCrop((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]),
        "val": transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    }

    image_path = os.path.join("assets", "flower_photos")

    train_dataset = datasets.ImageFolder(os.path.join(image_path, "train"), transform=data_transform["train"])
    val_dataset = datasets.ImageFolder(os.path.join(image_path, "val"), transform=data_transform["val"])

    train_num = len(train_dataset)
    val_num = len(val_dataset)

    flower_list = train_dataset.class_to_idx
    cla_dict = dict((val, key) for key, val in flower_list.items())

    json_str = json.dumps(cla_dict, indent=4)
    with open("class_indices.json", "w") as json_file:
        json_file.write(json_str)

    batch_size = 64
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    net = resnet.resnet101(num_classes=5, include_top=True)
    if os.path.exists(save_path):
        net.load_state_dict(torch.load(save_path))
    net.to(device)

    # 损失函数使用交叉熵
    loss_function = nn.CrossEntropyLoss()
    # 使用adam优化器
    optimizer = optim.Adam(net.parameters(), lr=0.0002)
    best_acc = 0.0
    for epoch in range(1000):
        net.train()
        running_loss = 0.0
        for step, data in enumerate(train_loader, start=1):
            images, labels = data
            optimizer.zero_grad()
            outputs = net(images.to(device))
            loss = loss_function(outputs, labels.to(device))
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            rate = step /len(train_loader)
            a = "*" * int(rate * 50)
            b = "." * int((1 - rate) * 50)
            print(f"\rtrain loss: {int(rate * 100)}%[{a}->{b}]{loss.item()}")
        print()

        net.eval()
        acc = 0.0
        with torch.no_grad():
            for val_data in val_loader:
                val_images, val_labels = val_data
                outputs = net(val_images.to(device))
                predict_y = torch.max(outputs, dim=1)[1]
                acc += (predict_y == val_labels.to(device)).sum().item()

            val_accurate = acc / val_num 
            if val_accurate > best_acc:
                best_acc = val_accurate
                torch.save(net.state_dict(), save_path)
            
            print(f"[epoch {epoch + 1}] train_loss: {running_loss / step} test_accuracy: {best_acc}")
    
    print("finished_training")

if __name__ == "__main__":
    main()
    
