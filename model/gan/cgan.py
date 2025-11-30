"""
cgan.py

The Pytorch implementation of the conditional generative adversarail network

Author: ZhangXin
Date: 2025/11/09
"""

import argparse
import os
import datetime
import random

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
    The generator of the conditional generative adversarial network
    """

    def __init__(self, num_classes, latent_dim, image_shape):

        super(Generator, self).__init__()

        self.image_shape = image_shape

        # embedding
        self.label_emb = nn.Embedding(num_embeddings=num_classes,
                                      embedding_dim=num_classes)
        
        self.layer = nn.Sequential(
            self.block(num_classes + latent_dim, 128),
            self.block(128, 256),
            self.block(256, 512),
            self.block(512, 1024),
            nn.Linear(1024, int(torch.prod(torch.Tensor(image_shape)).item())),
            nn.Tanh()
        )
    
    def block(self, in_features, out_features, normalize=True):
        layer = [nn.Linear(in_features=in_features,
                           out_features=out_features)]
        
        if normalize:
            layer.append(nn.BatchNorm1d(num_features=out_features))
        
        layer.append(nn.LeakyReLU(0.2))

        return nn.Sequential(*layer)
    
    def forward(self, noise, label):
        label.long()
        input = torch.cat([noise, self.label_emb(label)], dim=-1)
        return self.layer(input).view(noise.shape[0], 1, *self.image_shape)

class Discriminator(nn.Module):
    """
    The Discriminator of the conditional generative adversarial networks
    """

    def __init__(self, num_classes, image_shape):
        
        super(Discriminator, self).__init__()
        self.image_shape = image_shape

        self.label_embedding = nn.Embedding(num_classes, num_classes)

        self.layer = nn.Sequential(
            nn.Linear(num_classes + int(torch.prod(torch.Tensor(image_shape)).item()), 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 512),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 512),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1)
        )
    
    def forward(self, label, images):
        label = label.to(dtype=torch.long)
        label = self.label_embedding(label)

        input = torch.cat([label, images.view(images.shape[0], -1)], dim=1)
        return self.layer(input)

def train():

    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=32, help="size of batch")
    parser.add_argument("--channels", type=int, default=1, help="number of channels")
    parser.add_argument("--lr", type=float, default=0.0002, help="learning rate")
    parser.add_argument("--latent_dim", type=int, default=100, help="dimension of latent space")
    parser.add_argument("--image_size", type=int, default=28, help="size of image")
    args = parser.parse_args()

    start = datetime.datetime.now()

    generator_weight_path = os.path.join("checkpoints/gan/cgan", "generator.pth")
    discriminator_weight_path = os.path.join("checkpoints/gan/cgan", "discriminator.pth")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    transformer = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5] * args.channels, [0.5] * args.channels)
    ])

    data_set = datasets.MNIST(root="assets/", train=True, transform=transformer, download=False)

    data_loader = DataLoader(dataset=data_set, batch_size=args.batch_size, shuffle=True, num_workers=0)

    generator = Generator(num_classes=10, latent_dim=args.latent_dim, image_shape=[args.image_size] * 2).to(device)
    if os.path.exists(generator_weight_path):
        generator.load_state_dict(torch.load(generator_weight_path))
        print("load generator weight successfully")
    else:
        print("load generator weight fail")
    
    discriminator = Discriminator(num_classes=10, image_shape=[args.image_size] * 2).to(device)
    if os.path.exists(discriminator_weight_path):
        discriminator.load_state_dict(torch.load(discriminator_weight_path))
        print("load discriminator weight successfully")
    else:
        print("load discriminator weight fail")

    optimizer_g = optim.Adam(generator.parameters(), lr=args.lr)
    optimizer_d = optim.Adam(discriminator.parameters(), lr=args.lr)

    criterion = nn.MSELoss()

    epoch = 0
    runingloss_g = 0
    runingloss_d = 0

    while True:
        for i , (real_images, labels) in enumerate(tqdm.tqdm(data_loader)):
            valid = torch.ones(real_images.shape[0], 1, device=device)
            fake = torch.zeros(real_images.shape[0], 1, device=device)

            real_images = real_images.to(device)
            labels = labels.to(device)

            generator.train()
            optimizer_g.zero_grad()

            # generator training
            z = torch.randn(real_images.shape[0], args.latent_dim, device=device)
            gen_labels = torch.randint(0, 10, (real_images.shape[0], ), device=device)

            gen_images = generator(z, gen_labels)

            validty = discriminator(gen_labels, gen_images)
            loss = criterion(validty, valid)
            loss.backward()
            optimizer_g.step()
            runingloss_g += loss

            # discriminator training
            optimizer_d.zero_grad()

            output_d = discriminator(labels, real_images)
            loss_d = criterion(output_d, valid)
            output_g = discriminator(labels, gen_images.detach())
            loss_g = criterion(output_g, fake)
            loss = (loss_d + loss_g) / 2
            loss.backward()
            optimizer_d.step()
            runingloss_d += loss

            if i % 100 == 0:
                save_image(gen_images[ :25], f"save_images/gan/cgan/{i}.png", nrow=5, normalize=True)

        print(f"{datetime.datetime.now() - start} epoch: {epoch} d_loss: {runingloss_d/len(data_loader)} g_loss: {runingloss_g/len(data_loader)}")
        runingloss_d = 0
        runingloss_g = 0
        epoch += 1
        torch.save(generator.state_dict(), generator_weight_path)
        torch.save(discriminator.state_dict(), discriminator_weight_path)

def predict():
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    parser = argparse.ArgumentParser()
    parser.add_argument("--num", type=int, default=1, help="number of generating")
    args = parser.parse_args()

    label = torch.tensor(random.sample([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], k=args.num), device=device)
    latent = torch.randn(args.num, 100, device=device)

    generator = Generator(10, 100, [28, 28]).to(device=device)
    if os.path.exists("checkpoints/gan/cgan/generator.pth"):
        generator.load_state_dict(torch.load("checkpoints/gan/cgan/generator.pth"))
        print("load weight successfully")
    generator.eval()
    gen_images = generator(latent, label)
    print(label)
    save_image(gen_images, "image.png", normalize=True)

def main():
    train()

if __name__ == "__main__":
    main()



