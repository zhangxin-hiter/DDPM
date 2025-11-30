"""
betavae.py

Author: ZhangXin1`
Date: 2025/11/14
"""

import os
import argparse
import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import (
    transforms,
    datasets
)
from torchvision.utils import save_image
import tqdm

class BetaVAE(nn.Module):
    """
    Beta-VAE
        - 支持两种损失：H 型（标准 β-VAE）和 B 型（Capacity-based）
        - 编码器：卷积下采样 -> flatten -> μ/log_var
        - 解码器：线性映射 -> reshape -> 转置卷积上采样 
    """

    # --------------------------------------------------------------------------------------------
    # 初始化
    # --------------------------------------------------------------------------------------------
    def __init__(self,
                 in_channels,                               # 输入图像通道数
                 out_channels,                              # 输出图像通道数
                 latent_dim,                                # 潜在向量维度
                 hidden_dims,                               # 每层通道数列表
                 beta,                                      # KL项权重
                 gamma,
                 loss_type,
                 max_capacity,                              # KL 容量上限
                 capacity_max_iter):                # KL 容量上限最大迭代次数
        super(BetaVAE, self).__init__()

        self.latent_dim = latent_dim
        self.beta = beta
        self.gamma = gamma
        self.loss_type = loss_type
        self.C_max = max_capacity
        self.C_stop_iter = capacity_max_iter
        self.num_iter = 0

        # 若未指定，使用默认通道数
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512]

        # ------------------------- 编码器构建 -------------------------
        modules = []
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels=in_channels,
                              out_channels=h_dim,
                              kernel_size=3,
                              stride=2,
                              padding=1),
                    nn.BatchNorm2d(num_features=h_dim),
                    nn.LeakyReLU()
                )
            )
            in_channels = h_dim
        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(in_features=hidden_dims[-1], out_features=latent_dim)
        self.fc_var = nn.Linear(in_features=hidden_dims[-1], out_features=latent_dim)

        # ------------------------- 解码器构建 -------------------------
        self.decoder_input = nn.Linear(in_features=latent_dim,
                                       out_features=hidden_dims[-1])
        
        hidden_dims.reverse()                           # 翻转方便构建上采样
        modules = []
        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(in_channels=hidden_dims[i],
                                       out_channels=hidden_dims[i + 1],
                                       kernel_size=3,
                                       stride=2,
                                       padding=1,
                                       output_padding=1),
                    nn.BatchNorm2d(num_features=hidden_dims[i + 1]),
                    nn.LeakyReLU())
            )
        self.decoder = nn.Sequential(*modules)

        # 最后一层卷积再放大一次 + 3 x 3 conv 输出图像
        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(in_channels=hidden_dims[-1],
                               out_channels=hidden_dims[-1],
                               kernel_size=3, 
                               stride=2,
                               padding=1,
                               output_padding=1),
            nn.BatchNorm2d(num_features=hidden_dims[-1]),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=hidden_dims[-1],
                      out_channels=out_channels,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.Tanh()
        )

    # -----------------------------------------------------------------------------
    # 编码：x -> μ，log_var
    # ----------------------------------------------------------------------------- 
    def encode(self, input):
        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)
        return (mu, log_var)
    
    # -----------------------------------------------------------------------------
    # 解码：z -> 重构图像
    # ----------------------------------------------------------------------------- 
    def decode(self, z):
        result = self.decoder_input(z)
        result = result.view(-1, 512, 1, 1)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result

    # -----------------------------------------------------------------------------
    # 重参数化：μ，log_var -> z
    # -----------------------------------------------------------------------------
    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps * std + mu
    
    # -----------------------------------------------------------------------------
    # 前向：x -> 重构
    # -----------------------------------------------------------------------------
    def forward(self, input):
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        return [self.decode(z), input, mu, log_var]
    
    # -----------------------------------------------------------------------------
    # 损失函数：H 型或 B 型
    # -----------------------------------------------------------------------------
    def loss_function(self, *args, **kwargs):
        self.num_iter += 1
        recons = args[0]            # 重构图
        input = args[1]             # 原图
        mu = args[2]                
        log_var = args[3]   
        kld_weight = kwargs["M_N"]

        # 重构损失
        recons_loss = F.mse_loss(recons, input)

        # KL 散度（逐样本求和再取平均）
        kld_loss = torch.mean(torch.sum(-1 + torch.exp(log_var) + mu ** 2 - log_var, dim=1) * 0.5, dim=0)

        # 损失函数分支
        if self.loss_type == "H":           # Higgins
            loss = recons_loss + self.beta * kld_loss * kld_weight
        elif self.loss_type == "B":         # Burgess
            C = torch.clamp(self.C_max / self.C_stop_iter * self.num_iter, 0, self.C_max)
            loss = recons_loss + self.gamma * kld_weight * torch.abs(kld_loss - C)
        
        return {"loss": loss, "reconstruction_loss": recons_loss, "KLD": kld_loss}
    
    # -------------------------------------------------------------------------------
    # 随机采样
    # -------------------------------------------------------------------------------
    def sample(self, num_samples, current_device):
        z = torch.randn(num_samples, self.latent_dim, device=current_device)
        return self.decode(z)
    
    # -------------------------------------------------------------------------------
    # 给定输入，返回重构
    # -------------------------------------------------------------------------------
    def generate(self, x):
        return self.forward(x)
    
def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=32, help="size of batch")
    parser.add_argument("--lr", type=float, default=0.005, help="learning rate")
    parser.add_argument("--channels", type=int, default=3, help="number of channels")
    parser.add_argument("--latent_dim", type=int, default=128, help="dimension of latent space")
    parser.add_argument("--hidden_dims", type=list, default=None, help="list of encoder layer channels")
    parser.add_argument("--beta", type=int, default=10, help="beta")
    parser.add_argument("--gamma", type=int, default=1, help="gamma")
    parser.add_argument("--loss_type", type=str, default="H", help="type of loss function")
    parser.add_argument("--max_capacity", type=int, default=25, help="max capacity")
    parser.add_argument("--capacity_max_iter", type=int, default=100000, help="capacity max iteration")
    args = parser.parse_args()

    weight_path = os.path.join("checkpoints", "vae/betavae.pth")
    os.makedirs("save_images/vae/betavae", exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    transformer = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5] * args.channels, [0.5] * args.channels)
    ])

    data_set = datasets.CIFAR10(root="assets/cifar-10", train=True, transform=transformer, download=False)

    data_loader = DataLoader(dataset=data_set, batch_size=args.batch_size, shuffle=True, num_workers=0)

    model = BetaVAE(in_channels=args.channels, out_channels=args.channels, latent_dim=args.latent_dim, hidden_dims=None, beta=args.beta, gamma=args.gamma, loss_type=args.loss_type, max_capacity=args.max_capacity, capacity_max_iter=args.capacity_max_iter).to(device)
    if os.path.exists(weight_path):
        model.load_state_dict(torch.load(weight_path))
        print("load weight successfully")
    else:
        print("load weight fail")

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    epoch = 0
    start = datetime.datetime.now()
    run_loss = 0
    run_recons_loss = 0
    run_kld = 0

    while True:
        for i, (images, _) in enumerate(tqdm.tqdm(data_loader)):
            model.train()
            optimizer.zero_grad()
            images = images.to(device)
            recons, _, mu, log_var = model(images)
            loss_dict = model.loss_function(recons, images, mu, log_var, M_N=1/len(data_loader))
            loss_dict["loss"].backward()
            optimizer.step()
            run_loss += loss_dict["loss"].item()
            run_recons_loss += loss_dict["reconstruction_loss"].item()
            run_kld += loss_dict["KLD"].item()
            
            if i % 100 == 0:
                save_image(recons[:25], f"save_images/vae/betavae/{i}.png", nrow=5, normalize=True)

        print(f"{datetime.datetime.now() - start} epoch:{epoch} loss:{run_loss/len(data_loader)} reconstruction_loss:{run_recons_loss/len(data_loader)} KLD:{run_kld/len(data_loader)}")
        torch.save(model.state_dict(), weight_path)
        epoch += 1
        run_loss = 0
        run_recons_loss = 0
        run_kld = 0

def main():
    train()

if __name__ == "__main__":
    main()