# BrushNet 图像补全建模报告

**摘要**：本文围绕图像补全任务，对 RePaint、Stable Diffusion Inpainting 以及 BrushNet 三类代表性方法进行结构与机制分析。首先统一建模设定，随后依次梳理各算法的流程与关键公式，并重点讨论 BrushNet 的结构要求与实现取舍。最后给出一页式的算法对比小结，为后续综述或开题报告提供材料。

---

## 1 问题来源与建模设定

我们考虑 **图像补全（image inpainting）** 任务。
设：

* 原始完整图像：$x_0$
* 二值 mask：$M\in\{0,1\}^{H\times W}$，其中

  * $M=0$：保留区域（已知像素）
  * $M=1$：待补全区域（缺失像素）

为了方便对比 RePaint、Stable Diffusion inpainting 和 BrushNet，我们假定所有扩散过程都在 **latent 空间** 中进行：

* 利用 VAE 编码得到 latent 表示：

  * 完整图像：$z_0 = \text{VAEEnc}(x_0)$
  * 遮挡后图：$x_0^{\text{masked}} = (1-M)\odot x_0$，编码得到
    $$
    z_0^{\text{masked}} = \text{VAEEnc}\big(x_0^{\text{masked}}\big)
    $$

在下面的讨论中，我们**显式忽略编解码细节**，假定所有算法都直接在 latent $z$ 上做 DDPM / 采样 / 拼接，最后再通过 VAE 解码回像素空间进行 loss 计算。

---

## 2 RePaint：基于 DDPM + Masking 的补全

### 2.1 基本思想
RePaint 完全沿用 **标准 DDPM** 的训练流程：对完整 $z_0$ 做正向扩散，训练 UNet 型噪声预测器
$$
\epsilon_\theta(z_t, t) \approx \epsilon
$$
推理阶段同样保持标准 UNet 结构，只是输入噪声 latent $z_t$ 后，在采样环节引入强制性的 mask 约束。

### 2.2 利用 mask 的拼接与重采样
在每次 denoising 后，将已知区域替换为观测图（或其加噪版本）提供的内容：

* 把 unmasked 区域从 noisy 观测 latent 拷贝到当前 $z_t$ 中；
* 或者在像素/latent 空间按 mask 融合，形式化为
  $$
  z_{t-1} \leftarrow z_{t-1} \cdot \big(1 - M^{\text{resized}}\big) + z_{t-1}^{\text{masked}} \cdot M^{\text{resized}}
  \tag{2}
  $$
  其中 $M^{\text{resized}}$ 与 latent 尺度一致，$z_{t-1}^{\text{masked}}$ 来自观测图。

为缓解边界不连续，RePaint 还引入 **重采样 / time-travel**：在部分时间步回跳到更大的噪声尺度或对同一时间步多次采样并融合，从而提升边界一致性，但计算量远大于标准 DDPM。

---

## 3 Stable Diffusion Inpainting：结构编码 mask 与遮挡图

### 3.1 输入扩展
Stable Diffusion 提供的 inpainting 模型直接在 **UNet 输入通道** 上引入遮挡信息，使网络在训练阶段就学习如何利用 mask：

1. 当前 noisy latent：$z_t$
2. 遮挡图 latent：$z_0^{\text{masked}}$
3. 下采样后的 mask：$M^{\text{resized}}$

整体输入拼接为
$$
\text{UNet input} = \big[z_t,\ z_0^{\text{masked}},\ M^{\text{resized}}\big]
$$
输出仍是标准 DDPM 形式（预测噪声 $\hat\epsilon_t$ 或 $\hat z_{t-1}$），loss 也保持不变。

### 3.2 特点与代价
* 条件与生成完全共享同一条 UNet 分支，特征需要在所有层同时承载“生成 + 条件解析”任务；
* 通过训练阶段学习 mask 利用方式，不再依赖 RePaint 的 time-travel 技巧；
* 需要为 inpainting 单独微调一个 UNet 版本，模型迁移成本高于纯提示控制。

---

## 4 BrushNet：对 Stable Diffusion Inpainting 的结构拆解

BrushNet 的核心视角可以理解为：

> **把原来 inpainting UNet 里“条件处理”和“图像生成”解耦成两个分支：**
>
> * 原来的 diffusion UNet 作为 **生成主干**，不再直接吃 mask；
> * 新增 BrushNet 分支来专门处理 $(z_t, z_0^{\text{masked}}, M)$，并逐层向主干注入特征。

### 4.1 建模：主干 UNet + BrushNet 分支

在你给出的公式里，可以把标准 UNet 噪声预测写成 $\epsilon_\theta(z_t, t, C)$，其中 $C$ 表示文本等条件。

BrushNet 插入后的每一层 $i$ 的特征更新可以写成（按你原式稍微整理下）：



$$   
\epsilon_{\theta}^{\text{new}}(z_t, t, C)_i = \epsilon_{\theta}^{\text{base}}(z_t, t, C)_i + w \cdot \mathcal{Z}_i \left( \epsilon_{\theta}^{\text{BrushNet}} \left( [z_t, z_0^{\text{masked}}, M^{\text{resized}}], t \right)_i \right) \tag{1}
  $$



其中：

* $\epsilon_{\theta}^{\text{base}}$：**冻结的原 Stable Diffusion UNet**（生成主干）在第 $i$ 层的特征；
* $\epsilon_{\theta}^{\text{BrushNet}}$：BrushNet 分支在第 $i$ 层输出的特征；
* $\mathcal{Z}_i$：第 $i$ 层的 **ZeroConv**（初始化为 0 的卷积），保证一开始不破坏 base UNet 的行为；([ar5iv][1])
* $w$：控制 BrushNet 引导强度的系数（control scale），推理时可调；
* $[\cdot,\cdot]$：通道维度上的拼接。

这样做的效果是：

* 采样过程、噪声调度、loss 都仍然是 **标准 DDPM/Stable Diffusion**；
* difference 全部体现在**噪声预测网络的参数化方式**：
  主干负责“生成”，BrushNet 负责“条件解析 + per-pixel 引导”。

---

## 5 对 BrushNet 网络结构的要求：是否需要跟 UNet 匹配？

你问的关键问题是：

> 对 BrushNet 的网络结构有没有要求？需要跟 UNet 进行匹配吗？

### 5.1 论文里的“官方”做法

在原论文 4.1 节里，作者明确写到：

> “BrushNet utilizes a **clone of the pre-trained diffusion model** while excluding its cross-attention layers.”([ar5iv][1])

也就是说，**在实现上他们直接复制了一份 UNet 结构（包括各个分辨率、block 数、通道数），然后去掉 cross-attention**。这样有两个好处：

1. **分辨率 / 通道维度天然对齐**：
   每一层 $i$ 的 BrushNet 输出特征 $\epsilon_{\theta}^{\text{BrushNet}}(\cdot)_i$ 与主干 UNet 的特征 $\epsilon_{\theta}^{\text{base}}(\cdot)_i$ 在形状上完全一致，可以直接经过 ZeroConv 后相加；
2. **充分利用预训练先验**：
   复制 UNet 权重（除了 cross-attention）作为 BrushNet 的初始化，相当于用一个已经很强的图像特征提取网络来处理 masked image latent，利于训练收敛。([ar5iv][1])

**所以：原始 BrushNet 的实现里，BrushNet 的结构基本上是“UNet 的一份拷贝（去掉文本 cross-attention）”。**

---

### 5.2 从原理上看：真正“必须匹配”的是什么？

从公式 (1) 的加法结构可以看出，真正必须满足的条件是：

> **在每一个插入点 $i$，BrushNet 经过 ZeroConv 后的输出张量与主干 UNet 的特征张量在 shape 上必须一致**：
>
> * 空间尺寸：$H_i \times W_i$ 一致；
> * 通道数：$C_i$ 一致（或至少通过 $\mathcal Z_i$ 可变换到一致的通道维度）。

因此，从原理上讲，“需要匹配”的更多是 **特征层级的分辨率和维度**，而不是“必须完全照抄 UNet 的所有 block 设计”。也就是说：

* 你可以在每个下采样尺度上保留同样数量的插入点；
* 保证 BrushNet 在这些尺度上输出和 UNet 对应层同形状的特征（可以用 1×1 conv / ZeroConv 调整通道）；
* **这样在数学上就满足 BrushNet 插入的要求**。

---

### 5.3 实践上的建议（给你做结构设计时参考）

结合论文和实际工程经验，可以这么回答你报告里的问题：

1. **原论文中 BrushNet 的结构确实是与 UNet 高度匹配的**

   * 他们直接用“去掉文本 cross-attention 的 UNet 克隆”作为 BrushNet 分支；([ar5iv][1])
   * 这样所有层的分辨率 / 通道都对齐，插入点设计非常自然。

2. **从理论角度，BrushNet 不要求与 UNet 完全同构**

   * 只要在需要插入的那些层 $i$，BrushNet + ZeroConv 的输出形状可以和 UNet 对应层对上，就可以工作；
   * 因此，你可以设计更轻量的 BrushNet（例如减少通道数，然后通过 ZeroConv 映射到 UNet 的通道数）或者减少插入层的数量。

3. **但在你要复现原论文效果 / 直接沿用他们 checkpoint 时，建议保持结构匹配**

   * 如果你打算直接加载官方 BrushNet 权重或和 Stable Diffusion v1.5 一起用，那么基本就要按照论文的克隆结构来做；
   * 一旦你自行修改 BrushNet 的深度或通道数，就等于在做一个“自研版本的 BrushNet-like 模型”，需要重新训练。

所以可以在报告中这样总结回答：

> * **BrushNet 的本质约束是：在每个特征注入层，其输出必须在形状上与主干 UNet 的特征匹配，以便通过 ZeroConv 后进行逐元素相加。**
> * **原论文为了简单和效果，直接使用“去掉 cross-attention 的 UNet 克隆”作为 BrushNet 分支，因此在实现上与 UNet 几乎完全结构匹配。**
> * **如果只是从原理出发做自己的变体，BrushNet 不要求严格拷贝 UNet，但需要保证多尺度特征的分辨率和通道维度能够和主干网络对齐，同时要意识到这会偏离论文的标准实现，需要重新训练与验证。**

---

## 6 算法对比小结

| 方法 | 扩散过程建模 | UNet 输入（latent 空间） | 网络结构是否改动 | 采样策略是否改动 | 利用 mask 的方式 | 计算量特征 |
| --- | --- | --- | --- | --- | --- | --- |
| RePaint | 标准 DDPM | $z_t$（生成 latent）及来自观测图的 $z_t^{\text{obs}}$ | 否，沿用普通 UNet | 是：引入 time-travel / 重采样 | 在采样阶段对 latent/像素做硬拼接，保证已知区域 | 采样步数显著增加，计算量最高 |
| Stable Diffusion Inpainting | 标准 DDPM / SD | $[z_t,\ z_0^{\text{masked}},\ M^{\text{resized}}]$ | 是：扩展输入通道的 inpainting UNet | 否：采样日程与标准 DDPM 一致 | 在网络输入层显式编码 mask 与遮挡信息 | 较常规 SD 略增，仍在可接受范围 |
| BrushNet | 标准 DDPM / SD | 主干：$z_t$；BrushNet 分支：$[z_t,\ z_0^{\text{masked}},\ M^{\text{resized}}]$ | 是：主干 UNet + BrushNet 双分支经 ZeroConv 融合 | 否：采样与 SD 相同 | BrushNet 分支逐层注入条件 residual，推理可调节权重 | 多一条分支前向但无重采样，整体可控 |

该表在“扩散建模 / 输入模态 / 结构调整 / 采样策略 / mask 用法 / 计算量”六个维度总结三类方法差异，可直接嵌入综述或开题 PPT。

---

## 7 结论与展望

本文系统梳理了 RePaint、Stable Diffusion Inpainting 与 BrushNet 在图像补全过程中的建模假设、网络设计和采样策略。RePaint 通过多次重采样提高掩模边界一致性；Stable Diffusion Inpainting 在网络输入层融合条件；BrushNet 则以双分支设计实现“生成-条件”解耦，同时保持与原扩散模型的兼容性。对于需要复现官方结果或加载既有权重的场景，建议保持 BrushNet 与主干 UNet 的层级对齐；若追求轻量化，可在满足特征形状匹配的前提下探索更小的条件分支。

[1]: https://ar5iv.org/html/2403.06976v1 "[2403.06976] BrushNet: A Plug-and-Play Image Inpainting Model with Decomposed Dual-Branch Diffusion"



