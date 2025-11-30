"""
UNet.py
"""

from numpy import dtype
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import math

class Swish(nn.Module):
    """
    Swish激活函数，一种平滑的非线性函数
    效果优于Relu激活函数
    """

    def __init__(self):
        super(Swish, self).__init__()

    def forward(self, x):
        """
        前向过程
        
        parameters
        -----
        x: tensor
            输入张量
        """

        return x * torch.sigmoid(x)
    
class TimeEmbedding(nn.Module):
    """
    位置编码实现，将一个整数时间步映射到一个向量空间
    """

    def __init__(self, T, d_model, dim):
        """
        初始化函数

        parameters
        -----
        T: int
            最大时间步数
        d_model: int
                生成的初始维度
        dim: int
            最终映射维度
        """

        assert d_model % 2 == 0, "d_model必须为偶数"
        super(TimeEmbedding, self).__init__()
        # 构造位置编码矩阵
        emb = torch.arange(0, d_model, step=2) / d_model * math.log(1000)
        emb = torch.exp(-emb)
        pos = torch.arange(T).float()
        # 每个时间步t与不同频率相乘，得到t * freq
        emb = pos[:, None] * emb[None, :]
        # 正弦余弦展开
        emb = torch.stack((torch.sin(emb), torch.cos(emb)), dim=-1)
        assert list(emb.shape) == [T, d_model // 2, 2]
        # 合并
        emb = emb.view(T, d_model)
        
        self.timembedding = nn.Sequential(
            nn.Embedding.from_pretrained(emb),
            nn.Linear(d_model, dim),
            Swish(),
            nn.Linear(dim, dim)
        )

        self.initialize()

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, 
                                        mode="fan_out", 
                                        nonlinearity="relu")
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, t):
        emb = self.timembedding(t)
        return emb
    
class DownSample(nn.Module):
    """
    下采样过程，用于编码
    """
    def __init__(self, in_ch):
        """
        初始化函数
        
        parameters
        -----
        in_ch: int
                输入通道
        """

        super(DownSample, self).__init__()
        self.main = nn.Conv2d(in_channels=in_ch, 
                              out_channels=in_ch, 
                              kernel_size=3, 
                              stride=2, 
                              padding=1)

        self.initialize()

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, 
                                        mode="fan_out", 
                                        nonlinearity="leaky_relu")
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.main(x)
        return x
    
class UpSample(nn.Module):
    """
    上采样阶段
    """

    def __init__(self, in_ch):
        super(UpSample, self).__init__()
        self.main = nn.Conv2d(in_channels=in_ch, 
                              out_channels=in_ch, 
                              kernel_size=3, 
                              stride=1, 
                              padding=1)

        self.initialize()

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, 
                                        mode="fan_out", 
                                        nonlinearity="leaky_relu")
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        _, _, H, W = x.shape
        x = F.interpolate(x, 
                          scale_factor=2, 
                          mode="nearest")
        x = self.main(x)
        return x
    
class AttnBlock(nn.Module):
    """
    自注意力模块，常用卷积特征图上提取全局依赖
    用于U-Net的中间层或高分辨率特征层"""

    def __init__(self, in_ch):
        """
        初始化函数
        
        parameters
        -----
        in_ch: int
                输入通道
        """

        super(AttnBlock, self).__init__()
        self.group_norm = nn.GroupNorm(4, in_ch)
        self.proj_q = nn.Conv2d(in_channels=in_ch, 
                                out_channels=in_ch, 
                                kernel_size=1)
        self.proj_k = nn.Conv2d(in_channels=in_ch, 
                                out_channels=in_ch, 
                                kernel_size=1)
        self.proj_v = nn.Conv2d(in_channels=in_ch, 
                                out_channels=in_ch, 
                                kernel_size=1)
        self.proj = nn.Conv2d(in_channels=in_ch, 
                              out_channels=in_ch, 
                              kernel_size=1)

        self.initialize()

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, 
                                        mode="fan_out", 
                                        nonlinearity="leaky_relu")
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.group_norm(x)
        # 生成Q/K/V
        q = self.proj_q(h)
        k = self.proj_k(h)
        v = self.proj_v(h)

        # reshape为矩阵乘法形式
        q = q.permute(0, 2, 3, 1).view(B, H * W, C)
        k = k.view(B, C, H * W)

        # 计算注意力权重
        w = torch.bmm(q, k) * (C ** -0.5)
        assert list(w.shape) == [B, H * W, H * W]
        w = F.softmax(w, dim=-1)

        # 对V作加权平均
        v = v.permute(0, 2, 3, 1).view(B, H * W, C)
        h = torch.bmm(w, v)
        assert list(h.shape) == [B, H * W, C]

        # reshape回卷积格式
        h = h.view(B, H, W, C).permute(0, 3, 1, 2)
        h = self.proj(h)

        return x + h


class ResBlock(nn.Module):
    """
    残差块实现
    """

    def __init__(self, in_ch: int, out_ch: int, dropout: float, attn: bool = False):
        """
        初始化函数
        
        parameters
        -----
        in_ch: int
                输入通道数
        out_ch: int
                输出通道数
        tdim: int
                时间embedding的维度
        dropout: float
                dropout的比例
        attn: bool
                是否加入注意力机制
        """

        super(ResBlock, self).__init__()

        # 第一层卷积块
        self.block1 = nn.Sequential(
            nn.GroupNorm(num_groups=4, 
                         num_channels=in_ch),
            Swish(),
            nn.Conv2d(in_channels=in_ch, 
                      out_channels=out_ch, 
                      kernel_size=3, 
                      stride=1, 
                      padding=1)
        )
        # # 时间嵌入投影
        # self.temb_proj = nn.Sequential(
        #     Swish(), 
        #     nn.Linear(in_features=tdim, out_features=out_ch)
        # )
        # 第二层卷积块
        self.block2 = nn.Sequential(
            nn.GroupNorm(num_groups=4, num_channels=out_ch),
            Swish(),
            nn.Dropout(p=dropout),
            nn.Conv2d(in_channels=out_ch, 
                      out_channels=out_ch, 
                      kernel_size=3, 
                      stride=1, 
                      padding=1)
        )

        # 残差连接
        if in_ch != out_ch:
            self.short_cut = nn.Conv2d(in_channels=in_ch, 
                                       out_channels=out_ch, 
                                       kernel_size=1, 
                                       stride=1, 
                                       padding=0)
        else:
            self.short_cut = nn.Identity()
        
        # 注意力模块
        if attn:
            self.attn = AttnBlock(in_ch=out_ch)
        else:
            self.attn = nn.Identity()

        self.initialize()

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, 
                                        mode="fan_out", 
                                        nonlinearity="leaky_relu")
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)    

    def forward(self, x: Tensor) -> Tensor:
        """
        前向过程
        """

        h = self.block1(x)
        # h += self.temb_proj(temb)[:, :, None, None]
        h = self.block2(h)
        h += self.short_cut(x)
        h = self.attn(h)
        return h
    
class UNet(nn.Module):
    """
    UNet实现
    """    

    def __init__(self, T: int, ch: int, ch_mult: list, attn: list, num_res_block: int, dropout: float):
        """
        初始化
        
        paramters
        -----
        T: int
            最大时间步数
        ch: int
            初始通道数
        ch_mult: list
                通道数的倍数列表
        attn: list
                在哪些stage里加入自注意力机制
        num_res_block: int
                        在每个stage里加入多少个ResBlock
        dropout: float
                ResBlock里的dropout概率
        """

        super(UNet, self).__init__()
        assert all([i < len(ch_mult) for i in attn]), "attn index out of bound"
        # 时间嵌入
        tdim = ch * 4
        self.tim_embedding = TimeEmbedding(T, ch, tdim)
        # 输入层
        self.head = nn.Conv2d(in_channels=3,
                              out_channels=ch,
                              kernel_size=3, 
                              stride=1,
                              padding=1)
        # 下采样路径（encoder）
        self.downblocks = nn.ModuleList()
        chs = [ch]
        now_ch = ch
        for i, mult in enumerate(ch_mult):
            out_ch = ch * mult
            for _ in range(num_res_block):
                self.downblocks.append(ResBlock(
                    in_ch=now_ch,
                    out_ch=out_ch,
                    dropout=dropout,
                    attn=(i in attn)
                ))
                now_ch = out_ch
                chs.append(now_ch)
            if i != len(ch_mult) - 1:
                self.downblocks.append(DownSample(in_ch=now_ch))
                chs.append(now_ch)
        # 中间层（bottleneck）
        self.middleblocks = nn.ModuleList([
            ResBlock(in_ch=now_ch,
                     out_ch=now_ch,
                     dropout=dropout,
                     attn=True),
            ResBlock(in_ch=now_ch,
                     out_ch=now_ch,
                     dropout=dropout,
                     attn=False)
        ])
        # 上采样路径（decoder）
        self.upblocks = nn.ModuleList()
        for i, mult in reversed(list(enumerate(ch_mult))):
            out_ch = ch * mult
            for _ in range(num_res_block + 1):
                self.upblocks.append(ResBlock(
                    in_ch=chs.pop() + now_ch,
                    out_ch=out_ch,
                    dropout=dropout,
                    attn=(i in attn)
                ))
                now_ch = out_ch
            if i != 0:
                self.upblocks.append(UpSample(now_ch))
        assert len(chs) == 0
        # 输出层
        self.tail = nn.Sequential(
            nn.GroupNorm(32, now_ch),
            Swish(),
            nn.Conv2d(in_channels=now_ch,
                      out_channels=21,
                      kernel_size=3,
                      stride=1,
                      padding=1)
        )
        self.initialize()

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, 
                                        mode="fan_out", 
                                        nonlinearity="leaky_relu")
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)  

    def forward(self, x: Tensor) -> Tensor:
        # Timestep embedding
        # temb = self.tim_embedding(t)
        # Downsampling
        h = self.head(x)
        hs = [h]
        for layer in self.downblocks:
            h = layer(h)
            hs.append(h)
        # Middle
        for layer in self.middleblocks:
            h = layer(h)
        # Unsampling
        for layer in self.upblocks:
            if isinstance(layer, ResBlock):
                h = torch.cat([h, hs.pop()], dim=1)
            h = layer(h)
        h = self.tail(h)

        assert len(hs) == 0
        return h