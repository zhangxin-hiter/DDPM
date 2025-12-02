"""
diffusionmodel.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

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
    out = torch.gather(v, t, dim=0).float().to(device=device)
    return out.view([t.shape] + [1] * [len(x_shape) - 1])

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
        x_t = extract(self.sqrt_alphas_bar, t, x_0.shape) * x_0 + extract(self.sqrt_one_minus_alphas_bar. t, x_0.shape) * noise

        pred_noise = self.model(x_t, t)
        loss = F.mse_loss(pred_noise, noise, reduction="none")
        
        return loss