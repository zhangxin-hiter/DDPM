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

        