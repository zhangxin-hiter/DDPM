"""
train.py

模型训练文件
"""

from PIL import Image
import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.utils import save_image
from model.net import unet
import os
import numpy as np
from utils import resize_images

class MyDataset(Dataset):
    def __init__(self, path):
        self.path = path
        self.name = os.listdir(os.path.join(path, 'SegmentationClass'))

    def __len__(self):
        return len(self.name)

    def __getitem__(self, index):
        transform = transforms.Compose([
            transforms.ToTensor()
        ])
        segment_name = self.name[index]  # xx.png
        segment_path = os.path.join(self.path, 'SegmentationClass', segment_name)
        image_path = os.path.join(self.path, 'JPEGImages', segment_name.replace(".png", ".jpg"))
        segment_image = resize_images.keep_image_size_open(segment_path)
        image = resize_images.keep_image_size_open_rgb(image_path)
        return transform(image), torch.Tensor(np.array(segment_image, dtype=np.uint8))

def main():

    VOC_COLORMAP = [
    [0, 0, 0],          # background / void
    [128, 0, 0],        # aeroplane
    [0, 128, 0],        # bicycle
    [128, 128, 0],      # bird
    [0, 0, 128],        # boat
    [128, 0, 128],      # bottle
    [0, 128, 128],      # bus
    [128, 128, 128],    # car
    [64, 0, 0],         # cat
    [192, 0, 0],        # chair
    [64, 128, 0],       # cow
    [192, 128, 0],      # diningtable
    [64, 0, 128],       # dog
    [192, 0, 128],      # horse
    [64, 128, 128],     # motorbike
    [192, 128, 128],    # person
    [0, 64, 0],         # potted plant
    [128, 64, 0],       # sheep
    [0, 192, 0],        # sofa
    [128, 192, 0],      # train
    [0, 64, 128]        # tv/monitor
    ]

    palette = []

    for i in range(256):
        if i < len(VOC_COLORMAP):
            palette.extend(VOC_COLORMAP[i])
        else:
            palette.extend([0, 0, 0])

    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    print(device)
    # 数据集路径
    data_path = "assets/VOC2012"
    # 生成图像保存路径
    save_path = "save_images/net/unet"
    # 模型权重保存路径
    weight_path = "checkpoints/net/unet.pth"

    data_loader = DataLoader(MyDataset(data_path), batch_size=1, shuffle=True, num_workers=0)

    net = unet.UNet(0, ch=8, ch_mult=[4, 2, 2, 2], attn=[1], num_res_block=2, dropout=0.1)
    net.to(device)

    if os.path.exists(weight_path):
        net.load_state_dict(torch.load(weight_path))
        print("load weight successful")
    else:
        print("fail to load weight")

    # 优化器
    optimizer = optim.Adam(net.parameters(), lr=0.0002)

    loss_func = nn.CrossEntropyLoss(ignore_index=255)

    epoch = 0
    
    while True:
        for i, (image, segment_image) in enumerate(tqdm.tqdm(data_loader)):
            image = image.to(device)
            segment_image = segment_image.to(device)
            optimizer.zero_grad()
            out_image = net(image)
            loss = loss_func(out_image, segment_image.long())
            loss.backward()
            optimizer.step()

            print(f"{epoch}-{i}-train_loss===>>{loss.item()}")

            segment_label = segment_image[0].cpu().numpy().astype(np.uint8)
            out_label = torch.argmax(out_image[0], 0).cpu().numpy().astype(np.uint8)

            _segment_image = Image.fromarray(segment_label, "P")
            _out_image = Image.fromarray(out_label, "P")

            _segment_image.putpalette(palette)
            _out_image.putpalette(palette)

            if i % 100 == 0:
                background = Image.new("P", (_segment_image.width + _out_image.width, _segment_image.height))
                background.putpalette(palette)
                background.paste(_segment_image, (0, 0))
                background.paste(_out_image, (_segment_image.width, 0))
                background.save(f"{save_path}/{i}.png")
        
        if epoch % 20 == 0:

            torch.save(net.state_dict(), weight_path)
            print("save weight successfully")
        epoch += 1

if __name__ == "__main__":
    main()