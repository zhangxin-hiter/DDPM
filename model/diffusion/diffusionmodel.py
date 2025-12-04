"""
diffusionmodel.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from model.net.unet import UNet

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
        self.register_buffer("posterior_var", self.betas * (1. - self.alphas_bar_prev) / (1. - self.alphas_bar))

    # ----------------------------------- 辅助函数 ----------------------------------------
    def predict_xt_prev_mean_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape

        return extract(self.coeff1, t, x_t.shape) * x_t - extract(self.coeff2, t, x_t.shape) * eps
    
    def p_mean_variance(self, x_t, t):
        """
        返回 reverse process 的均值与方差
        """

        # 拼接方差
        var = torch.cat([self.posterior_var[1:2], self.betas[1:]])
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
    pass

def predict():
    pass

def main():
    train()

if __name__ == "__main__":
    main()