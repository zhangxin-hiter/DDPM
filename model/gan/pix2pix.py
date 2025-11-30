"""
pix2pix.py

Author: ZhangXin
Date: 2025/11/10
"""

import argparse
import os
import glob
from PIL import Image
import datetime

import numpy
import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import (
    Dataset,
    DataLoader
)
from torchvision import transforms
from torchvision.utils import save_image

class ImageDataset(Dataset):
    """
    成对图像数据集加载器
    约定：
        1、所有图片已拼接成一张大图，左半张为A域，右半张为 B 域
        2、训练时将 train 与 test 目录下的图片全部加入，验证/测试只加载对应目录
    """

    def __init__(self, root, mode="train", transformer=None):
        """
        参数
        -----
        root：str
            数据集根目录，其下应包含与 mode 同名文件夹
        mode：str
            当前阶段，通常取 "train" | "test" | "val" 
        transformer: list
            针对单张图片的 torchvision transforms 列表，会在内部组合成 transforms.compose(transformer)
        """
        
        super(ImageDataset, self).__init__()

        if transformer is None:
            transformer = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])

        self.transformer = transformer

        # 首先加载当前 mode 下的图片
        self.files = sorted(glob.glob(os.path.join(root, mode) + "/*.*"))
        
        # 训练阶段额外把 test 目录下的图片加载进来
        if mode == "train":
            self.files.extend(sorted(glob.glob(os.path.join(root, "test") + "/*.*")))

    def __getitem__(self, index):
        """
        按索引返回一对图像（字典形式）
        
        return
        -----
        dict：{"A": Tensor, "B": Tensor}
            A 为左半张图像，B 为右半张，均已作 transformer 处理
        """

        # 循环取模，避免越界
        img = Image.open(self.files[index % len(self.files)])
        w, h = img.size

        # 左右裁剪
        image_a = img.crop((0, 0, w / 2, h))
        image_b = img.crop((w / 2, 0, w, h))

        # 数据增强
        if numpy.random.random() < 0.5:
            image_a = Image.fromarray(numpy.array(image_a)[:, ::-1, :], "RGB")
            image_b = Image.fromarray(numpy.array(image_b)[:, ::-1, :], "RGB")

        # 转化为 Tensor
        image_a = self.transformer(image_a)
        image_b = self.transformer(image_b)

        return {"A": image_a, "B": image_b}
    
    def __len__(self):
        """
        数据集大小
        """

        return len(self.files)

class UNetDown(nn.Module):
    """
    UNet 编码器下采样基本块
    Conv2d -> (InstanceNorm) -> LeakyReLU -> (Dropout)
    默认 stride=2，因此每次调用特征图空间尺寸减半
    """

    def __init__(self, in_channels, out_channels, normalize=True, dropout=0.0):
        """
        parameters
        -----
        in_channels：int
            输入特征图通道数
        out_channels：int
            输出特征图通道数
        normalize：bool
            是否使用 InstanceNorm2d
        dropout：float
            Dropout 概率
        """

        super(UNetDown, self).__init__()

        # 4 x 4 卷积，stride=2 -> 尺寸减半
        layers = [nn.Conv2d(in_channels=in_channels,
                            out_channels=out_channels,
                            kernel_size=4,
                            stride=2,
                            padding=1)]
        
        # 可选 InstanceNorm
        if normalize:
            layers.append(nn.InstanceNorm2d(out_channels))

        # 激活函数
        layers.append(nn.LeakyReLU(0.2))
        
        # 可选 Dropout（通常在较深的 Layer 才启用）
        if dropout:
            layers.append(nn.Dropout(dropout))
        
        # 打包
        self.layer = nn.Sequential(*layers)

    def forward(self, x):
        """
        前向传播
        
        parameters
        -----
        x：Tensor shape（batch，in_channels，height，width）
        
        return
        -----
        Tensor，shape（batch，out_channels，height // 2, width // 2）
        """

        return self.layer(x)

class UNetUp(nn.Module):
    """
    UNet 编码器上采样基本块
    Upsample -> Conv2d -> InstanceNorm -> LeakyReLU -> （Dropout）
    包含一层 Upsample，每次调用特征图空间尺寸增加一倍
    """

    def __init__(self, in_channels, out_channels, dropout=0.0):
        """
        parameters
        -----
        in_channels：int
            输入特征图通道
        out_channels：int
            输出特征图通道
        dropout：float
            Dropout 概率
        """

        super(UNetUp, self).__init__()

        # 最邻近上采样 2 倍
        # 3x3 卷积
        # InstanceNorm + leakyReLU
        layers = [nn.Upsample(scale_factor=2),
                  nn.Conv2d(in_channels=in_channels,
                            out_channels=out_channels,
                            kernel_size=3, 
                            stride=1,
                            padding=1),
                  nn.InstanceNorm2d(out_channels),
                  nn.LeakyReLU(0.2)]
        
        if dropout:
            layers.append(nn.Dropout(dropout))

        self.layer = nn.Sequential(*layers)

    def forward(self, x, skip_input):
        """
        前向传播
        
        parameters
        -----
        x：Tensor，shape（batch，channels，width，height）
            解码器特征图
        skip_input：Tensor，shape（batch，channels，width*2，height*2）
            对应的同级编码器特征图

        return
        -----
        Tensor，shape（batch，channels*2，width*2，height*2）
        """
        x = self.layer(x)
        return torch.cat([x, skip_input], dim=1)

class Generator(nn.Module):
    """
    UNet 生成器
    编码器（8 次下采样）-> 瓶颈 -> 解码器（7 次上采样）-> 输出层
    """
    
    def __init__(self, in_channels, out_channels):
        """
        parameters
        -----
        in_channels：int
            输入特征图通道数
        out_channels：int
            输出特征图通道数
        """

        # ---------------- 编码器 -------------------
        # 每次空间尺寸 /2，通道 x2（第一层除外）
        super(Generator, self).__init__()

        self.down1 = UNetDown(in_channels=in_channels,
                              out_channels=64, 
                              normalize=False)
        self.down2 = UNetDown(in_channels=64,
                              out_channels=128)
        self.down3 = UNetDown(in_channels=128, 
                              out_channels=256)
        self.down4 = UNetDown(in_channels=256,
                              out_channels=512,
                              dropout=0.5)
        self.down5 = UNetDown(in_channels=512,
                              out_channels=512,
                              dropout=0.5)
        self.down6 = UNetDown(in_channels=512,
                              out_channels=512,
                              dropout=0.5)
        self.down7 = UNetDown(in_channels=512,
                              out_channels=512,
                              dropout=0.5)
        self.down8 = UNetDown(in_channels=512,
                              out_channels=512,
                              normalize=False,
                              dropout=0.5)
        
        # ----------------- 解码器 ---------------------
        # 每次空间尺寸 x2
        # 输入通道 = 上一层输出 + skip 同层编码器输出（拼接后）
        self.up1 = UNetUp(in_channels=512,
                          out_channels=512,
                          dropout=0.5)
        self.up2 = UNetUp(in_channels=1024, 
                          out_channels=512, 
                          dropout=0.5)
        self.up3 = UNetUp(in_channels=1024, 
                          out_channels=512, 
                          dropout=0.5)
        self.up4 = UNetUp(in_channels=1024, 
                          out_channels=512, 
                          dropout=0.5)
        self.up5 = UNetUp(in_channels=1024,
                          out_channels=256)
        self.up6 = UNetUp(in_channels=512,
                          out_channels=128)
        self.up7 = UNetUp(in_channels=256,
                          out_channels=64)
        
        # ----------------- 输出层 --------------------
        # 在上采样一次 -> 与原图同尺寸
        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(in_channels=128,
                      out_channels=out_channels,
                      kernel_size=4, 
                      padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        # ------ 编码下采样 ------
        x1 = self.down1(x)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        x5 = self.down5(x4)
        x6 = self.down6(x5)
        x7 = self.down7(x6)
        x8 = self.down8(x7)

        # ------ 解码上采样 ------
        x9 = self.up1(x8, x7)
        x10 = self.up2(x9, x6)
        x11 = self.up3(x10, x5)
        x12 = self.up4(x11, x4)
        x13 = self.up5(x12, x3)
        x14 = self.up6(x13, x2)
        x15 = self.up7(x14, x1)

        # ------ 输出层 ------
        return self.final(x15)

class Discriminator(nn.Module):
    
    def __init__(self, in_channels=3):
        super(Discriminator, self).__init__()

        self.layer = nn.Sequential(
            self.block(in_channels=in_channels * 2,
                       out_channels=64, 
                       normalize=False),
            self.block(in_channels=64, 
                       out_channels=128),
            self.block(in_channels=128,
                       out_channels=256),
            self.block(in_channels=256,
                       out_channels=512),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(in_channels=512,
                      out_channels=1,
                      kernel_size=4,
                      padding=1)
        )

    def block(self, in_channels, out_channels, normalize=True):
        layer = [nn.Conv2d(in_channels=in_channels,
                           out_channels=out_channels,
                           kernel_size=4,
                           stride=2,
                           padding=1)]
        
        if normalize:
            layer.append(nn.InstanceNorm2d(out_channels))

        layer.append(nn.LeakyReLU(0,2))

        return nn.Sequential(*layer)
    
    def forward(self, image_a, image_b):
        return self.layer(torch.cat([image_a, image_b], dim=1))

def train():

    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    print(device)
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=32, help="size of batch")
    parser.add_argument("--lr", type=float, default=0.0002, help="learning rate")
    parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--image_width", type=int, default=256, help="size of image weight")
    parser.add_argument("--image_height", type=int, default=256, help="size of image height")
    parser.add_argument("--channels", type=int, default=3, help="number of image channels")
    parser.add_argument("--data_path", type=str, default=None, help="the path of dataset")
    args = parser.parse_args()

    data_path = os.path.join("assets", f"{args.data_path}")

    generator_weight_path = os.path.join("checkpoints", "gan/pixel2pixel/generator.pth")
    discriminator_weight_path = os.path.join("checkpoints", "gan/pixel2pixel/discriminator.pth")

    transformer = transforms.Compose([
        transforms.Resize((args.image_height, args.image_width)),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * args.channels, [0.5] * args.channels)
    ])

    data_loader = DataLoader(ImageDataset(root=data_path, mode="train", transformer=transformer), batch_size=args.batch_size, shuffle=True, num_workers=0)

    generator = Generator(in_channels=3, out_channels=3).to(device)
    if os.path.exists(generator_weight_path):
        generator.load_state_dict(torch.load(generator_weight_path))
        print("load generator weight successfully")
    else:
        print("load generator weight fail")

    discriminator = Discriminator(in_channels=3).to(device)
    if os.path.exists(discriminator_weight_path):
        discriminator.load_state_dict(torch.load(discriminator_weight_path))
        print("load discriminator weight successfully")
    else:
        print("load discriminator weight fail")

    # loss function
    criterion_gan = nn.MSELoss()
    criterion_pixelwise = nn.L1Loss()

    # optimizer
    optimizer_g = optim.Adam(generator.parameters(), lr=args.lr, betas=(args.b1, args.b2))
    optimizer_d = optim.Adam(discriminator.parameters(), lr=args.lr, betas=(args.b1, args.b2))

    start = datetime.datetime.now()
    epoch = 0
    lambda_pixel = 100
    # 使用特征图判断真假
    patch = [1, args.image_height // 16, args.image_width // 16]

    while True:
        for i, images in enumerate(tqdm.tqdm(data_loader)): 
            real_a = images["A"].to(device)
            real_b = images["B"].to(device)

            valid = torch.ones(real_a.shape[0], *patch, device=device)
            fake = torch.zeros(real_b.shape[0], *patch, device=device)

            # generator training
            generator.train()
            optimizer_g.zero_grad()

            fake_b = generator(real_a)
            pred_fake = discriminator(real_a, fake_b)

            loss_gan = criterion_gan(pred_fake, valid)
            loss_pixel = criterion_pixelwise(fake_b, real_b)

            loss_g = loss_gan + lambda_pixel * loss_pixel
            loss_g.backward()
            optimizer_g.step()

            # discriminator training
            discriminator.train()
            optimizer_d.zero_grad()
            
            # real loss
            pred_real = discriminator(real_a, real_b)
            loss_real = criterion_gan(pred_real, valid)

            # fake loss
            pred_fake = discriminator(real_a, fake_b.detach())
            loss_fake = criterion_gan(pred_fake, fake)

            loss_d = (loss_fake + loss_real) / 2

            loss_d.backward()
            optimizer_d.step()

            if i % 5 == 0:
                save_image(torch.cat([real_a, real_b, fake_b], dim=-1)[:2], f"save_images/gan/pixel2pixel/{i}.png", normalize=True)

        # log process
        print(f"{datetime.datetime.now() - start} epoch:{epoch} loss_g:{loss_g} loss_d:{loss_d}")
        torch.save(generator.state_dict(), generator_weight_path)
        torch.save(discriminator.state_dict(), discriminator_weight_path)
        epoch += 1

def predict():
    pass

def main():
    train()

if __name__ == "__main__":
    main()