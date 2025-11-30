"""
dcgan.py

A Pytorch implementation of the deep convolutional generative adversarial net

Author: ZhangXin
Date: 2025/11/8
"""

import argparse
import os

import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import (
    transforms,
    datasets
)
from torchvision.utils import save_image

class Generator(nn.Module):
    """
    The Generator of the deep convolutional generative adversariar net
    """
    
    def __init__(self, image_size, latent_dim, channels):
        super(Generator, self).__init__()

        self.init_size = image_size // 16
        self.linear_1 = nn.Linear(in_features=latent_dim,
                                  out_features=1024 * self.init_size ** 2)

        self.layer = nn.Sequential(
            nn.BatchNorm2d(num_features=1024),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels=1024, 
                      out_channels=512,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels=512,
                      out_channels=256,
                      kernel_size=3,
                      stride=1,
                      padding=1,),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels=256,
                      out_channels=128,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels=128,
                      out_channels=3,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.Tanh()
        )

        self.initialize()

    def initialize(self):
       for m in self.modules():
           if isinstance(m, nn.Conv2d):
               nn.init.kaiming_normal_(tensor=m.weight, mode="fan_out",nonlinearity="relu")
           elif isinstance(m, nn.Linear):
               nn.init.normal_(tensor=m.weight, mean=0, std=0.02)
               nn.init.constant_(tensor=m.bias,val=0)

    def forward(self, x):
        x = self.linear_1(x).view(x.shape[0], 1024, self.init_size, self.init_size)
        return self.layer(x)

class Discriminator(nn.Module):
    """
    The Discriminator of the deep convolutional generative adversarial net
    """

    def __init__(self, image_size, channels):
        super(Discriminator, self).__init__()

        self.layer = nn.Sequential(
            self.block(in_channels=channels,
                       out_channels=16,
                       bn=False),
            self.block(in_channels=16,
                       out_channels=32),
            self.block(in_channels=32,
                       out_channels=64),
            self.block(in_channels=64,
                       out_channels=128),
            self.block(in_channels=128,
                       out_channels=256)         
        )

        ds_size = image_size // 32
        self.linear_layer = nn.Sequential(
            nn.Linear(in_features=256 * ds_size ** 2, 
                      out_features=1),
            nn.Sigmoid()
        )

        self.initialize()
    
    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(tensor=m.weight, mode="fan_out", nonlinearity="leaky_relu")
            elif isinstance(m, nn.Linear):
                nn.init.normal_(tensor=m.weight, mean=0, std=0.02)
                nn.init.constant_(tensor=m.bias, val=0)

    def block(self, in_channels, out_channels, bn=True):
        layer = []
        layer.append(nn.Conv2d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=3,
                               stride=2,
                               padding=1))
        layer.append(nn.LeakyReLU(0.2))
        if bn:
            layer.append(nn.BatchNorm2d(out_channels))
        return nn.Sequential(*layer)
    
    def forward(self, x):
        output = self.layer(x)
        return  self.linear_layer(torch.flatten(output, start_dim=1))

def train():

    device = torch.device("cuda:1" if  torch.cuda.is_available() else "cpu")
    print(device)

    os.makedirs("save_images/gan/dcgan", exist_ok=True)

    generator_weight_path = os.path.join("checkpoints/gan", "dcgan/dcgan_generator.pth")
    discriminator_weight_path = os.path.join("checkpoints/gan", "dcgan/dcgan_discriminator.pth")
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=32, help="size of batches")
    parser.add_argument("--image_channels", type=int, default=3, help="number of image channels")
    parser.add_argument("--image_size", type=int, default=256, help="size of image")
    parser.add_argument("--lr", type=float, default=0.0002, help="learning rate")
    parser.add_argument("--latent_dim", type=int, default=1000, help="latent dimension")
    args = parser.parse_args()

    transformer = transforms.Compose([
        transforms.Resize([256, 256]),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * args.image_channels, [0.5] * args.image_channels)
    ])

    data_set = datasets.ImageFolder(root="assets/flower_photos/train/", transform=transformer)
    
    data_loader = DataLoader(dataset=data_set,
                             batch_size=args.batch_size,
                             shuffle=True,
                             num_workers=0)
    
    generator = Generator(image_size=args.image_size,
                          latent_dim=args.latent_dim,
                          channels=args.image_channels).to(device)
    if os.path.exists(generator_weight_path):
        generator.load_state_dict(torch.load(generator_weight_path))
        print("load generator weight successfully")
    else:
        print("load generator weight fail")
    
    discriminator = Discriminator(image_size=args.image_size,
                                  channels=args.image_channels).to(device)
    if os.path.exists(discriminator_weight_path):
        discriminator.load_state_dict(torch.load(discriminator_weight_path))
        print("load discriminator weight successfully")
    else:
        print("load discriminator weight fail")

    # optimizer
    optimizer_g = optim.Adam(generator.parameters(), lr=args.lr)
    optimizer_d = optim.Adam(discriminator.parameters(), lr=args.lr)

    # loss function
    criterion = nn.MSELoss()

    epoch = 0
    runing_loss_g = 0
    runing_loss_d = 0

    while True:
        for i, (real_images, _) in enumerate(tqdm.tqdm(data_loader)):
            real_images = real_images.to(device)

            # label
            valid = torch.ones(real_images.shape[0], 1, device=device)
            fake = torch.zeros(real_images.shape[0], 1, device=device)

            # generator training
            generator.train()
            z = torch.randn(real_images.shape[0], args.latent_dim, device=device)
            optimizer_g.zero_grad()
            generate_images = generator(z)
            generate_loss = criterion(discriminator(generate_images), valid)
            generate_loss.backward()
            optimizer_g.step()
            
            # discriminator training
            discriminator.train()
            optimizer_d.zero_grad()
            real_loss = criterion(discriminator(real_images), valid)
            fake_loss = criterion(discriminator(generate_images.detach()), fake)
            d_loss = (real_loss + fake_loss) / 2
            d_loss.backward()
            optimizer_d.step()

            runing_loss_g += generate_loss.item()
            runing_loss_d += d_loss.item()

            if i % 10 == 0:
                save_image(generate_images[:4], f"save_images/gan/dcgan/{i}.png", nrow=2, normalize=True)

        print(f"epoch: {epoch} D loss: {runing_loss_d/len(data_loader)} G loss: {runing_loss_g/len(data_loader)}")
        torch.save(generator.state_dict(), generator_weight_path)
        torch.save(discriminator.state_dict(), discriminator_weight_path)
        epoch += 1
        runing_loss_g = 0
        runing_loss_d = 0           

def predict():
    pass
    

def main():
    train()

if __name__ == "__main__":
    main()
            