"""
diffusionmodel.py
"""

import os
import sys
sys.path.append(os.getcwd())

import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F
from torchvision import (
    transforms,
    datasets
)
from torchvision.utils import save_image

from model.net.unet import UNet
from utils import scheduler

def extract(v, t, x_shape):
    """
    从 v 中按 t 指定的索引取出对应的时间步的值

    parameters
    -----
    v：Tensor
        长度为 T 的 1-D 张量
    t：Tensor
        形状为 [B] 的整型张量，存放每个样本的扩散时刻
    x_shape: Tensor
        原始图像张量的形状，如 [B, C, H, W]

    return
    -----
    Tensor：形状为 [B, 1, 1, ...] 的张量，可与 x 逐元素相乘 
    """

    device = t.device
    out = torch.gather(v, index=t, dim=0).float().to(device=device)
    return out.view([t.shape[0]] + [1] * (len(x_shape) - 1))

class DiffusionTrainer(nn.Module):
    """
    构造 DDPM 前向过程并计算 MSE 损失
    """

    def __init__(self, model, beta_1, beta_T, T):
        """
        parameters
        -----
        model：Module
            噪声预测网络
        beta_1: float
            初始线性方差调度
        beta_T：float
            终止线性方差调度
        T：int 
            总扩散步数
        """

        super(DiffusionTrainer, self).__init__()
        
        self.model = model
        self.T = T

        # 线性方差调度 β_t
        self.register_buffer('betas', torch.linspace(beta_1, beta_T, steps=T).double())
        alphas = 1 - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)

        # 预计算两个常用系数
        self.register_buffer("sqrt_alphas_bar", torch.sqrt(alphas_bar))
        self.register_buffer("sqrt_one_minus_alphas_bar", torch.sqrt(1 - alphas_bar))

    def forward(self, x_0):
        """
        一次训练迭代

        x_0：Tensor
        """

        # 采样时刻
        t = torch.randint(self.T, size=(x_0.shape[0], ), device=x_0.device)

        # 采样噪声
        noise = torch.randn_like(x_0)

        # 扩散到 x_t
        x_t = extract(self.sqrt_alphas_bar, t, x_0.shape) * x_0 + extract(self.sqrt_one_minus_alphas_bar, t, x_0.shape) * noise

        pred_noise = self.model(x_t, t)
        loss = F.mse_loss(pred_noise, noise, reduction="none")
        
        return loss
    
class DiffusionSampler(nn.Module):
    """
    从纯噪声 x_T 逐步去噪到 x_0
    """

    def __init__(self, model, beta_1, beta_T, T):
        super(DiffusionSampler, self).__init__()
        
        self.model = model
        self.T = T

        # 线性方差调度
        self.register_buffer("betas", torch.linspace(beta_1, beta_T, T).double())
        alphas = 1 - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)
        alphas_bar_prev = F.pad(alphas_bar, [1, 0], value=1)[:T]

        # 计算反向过程均值系数
        self.register_buffer("coeff1", torch.sqrt(1. / alphas))
        self.register_buffer("coeff2", self.coeff1 * (1. - alphas) / torch.sqrt(1. - alphas_bar))

        # 计算反向过程方差
        self.register_buffer("posterior_var", self.betas * (1. - alphas_bar_prev) / (1. - alphas_bar))

    # ----------------------------------- 辅助函数 ----------------------------------------
    def predict_xt_prev_mean_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape

        return extract(self.coeff1, t, x_t.shape) * x_t - extract(self.coeff2, t, x_t.shape) * eps
    
    def p_mean_variance(self, x_t, t):
        """
        返回 reverse process 的均值与方差
        """

        # 拼接方差
        var = self.posterior_var
        var = extract(var, t, x_t.shape)

        # 网络预测噪声
        eps = self.model(x_t, t)
        xt_prev_mean = self.predict_xt_prev_mean_from_eps(x_t, t, eps)
        return xt_prev_mean, var

    # ---------------------------------- 主采样循环 -----------------------------------------
    def forward(self, x_T):
        """
        从 x_T ~ N(0, I) 逐步采样到 x_0
        """

        x_t = x_T
        for time_step in reversed(range(self.T)):
            print(time_step)
            t = x_t.new_ones([x_T.shape[0]], dtype=torch.long) * time_step
            mean, var = self.p_mean_variance(x_t, t)

            # 只在 t>0 时加噪声
            if time_step > 0:
                noise = torch.randn_like(x_t)
            else:
                noise = 0

            x_t = mean + torch.sqrt(var) * noise

        x_0 = x_t
        return torch.clip(x_0, -1 , 1)

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    weight_path = os.path.join("checkpoints", "diffusion/diffusionmodel.pth")

    transformer = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    dataset = datasets.CIFAR10(root="assets/cifar-10", train=True, transform=transformer, download=False)
    dataloader = DataLoader(dataset=dataset, batch_size=32, shuffle=True, num_workers=0)

    net = UNet(T=1000,
               ch=128,
               ch_mult=[1, 2, 3, 4],
               attn=[2],
               num_res_block=2,
               dropout=0.15).to(device=device)
    if os.path.exists(weight_path):
        net.load_state_dict(torch.load(weight_path))
        print("load weight successfully")
    else:
        print("load weight fail")

    optimizer = optim.Adam(net.parameters(), lr=1e-4, weight_decay=1e-4)
    cosinescheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=200, eta_min=0, last_epoch=-1)
    warmupscheduler = scheduler.GradualWarmupScheduler(optimizer=optimizer, multiplier=2, warm_epoch=20, after_scheduler=cosinescheduler)
    trainer = DiffusionTrainer(net, beta_1=1e-4, beta_T=0.02, T=1000).to(device=device)
    # sampler = DiffusionSampler(net, beta_1=1e-4, beta_T=0.02, T=1000).to(device=device)

    epoch = 0
    running_loss = 0

    # training
    while(True):
        for i, (images, _) in enumerate(tqdm.tqdm(dataloader)):
            net.train()
            optimizer.zero_grad()
            images = images.to(device=device)
            loss = trainer(images).sum() / 32.
            running_loss += loss.item()
            loss.backward()
            optimizer.step()

            # if i % 500 == 0:
            #     with torch.no_grad():
            #         x_T = torch.randn_like(images)
            #     save_image(sampler(x_T)[:25], f"save_images/diffusion/diffusionmodel/{i}.png", nrow=5, normalize=True)

        warmupscheduler.step()
        print(f"epoch:{epoch} loss:{running_loss / len(dataloader)}")
        epoch += 1
        torch.save(net.state_dict(), "checkpoints/diffusion/diffusionmodel.pth")
        running_loss = 0

def predict():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    net = UNet(T=1000,
               ch=128,
               ch_mult=[1, 2, 3, 4],
               attn=[2],
               num_res_block=2,
               dropout=0.15).to(device=device)
    net.load_state_dict(torch.load("checkpoints/diffusion/diffusionmodel.pth"))
    sampler = DiffusionSampler(model=net,
                               beta_1=1e-4, 
                               beta_T=0.02, 
                               T=1000).to(device)
    with torch.no_grad():
        x = torch.randn(1, 3, 32, 32).to(device=device)
        save_image(sampler(x), f"save_images/diffusion/diffusionmodel/img.png", normalize=True)

def main():
    predict()

if __name__ == "__main__":
    main()
    