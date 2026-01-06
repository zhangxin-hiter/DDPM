from dataclasses import dataclass
from typing import (
    Any,
    Union,
    Tuple,
    Optional,
    Dict,
    List
)

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.embeddings import (
    Timesteps,
    TimestepEmbedding,
    TextImageProjection,
    TextTimeEmbedding,
    TextImageTimeEmbedding
)
from diffusers.models.unets.unet_2d_blocks import (
    UNetMidBlock2D,
    UNetMidBlock2DCrossAttn,
    get_down_block
)
from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel
from diffusers.models.attention_processor import (
    ADDED_KV_ATTENTION_PROCESSORS,
    CROSS_ATTENTION_PROCESSORS,
    AttnProcessor,
    AttnAddedKVProcessor
)
from diffusers.utils import (
    BaseOutput,
    logging
)
from diffusers.configuration_utils import register_to_config

logger = logging.get_logger(__name__)

@dataclass
class ControlNetOutput(BaseOutput):
    """
    The Output of ['ControlNetOutput']
    
    Args:
        down_block_res_samples (Tuple[torch.Tensor]):
        mid_block_res_sample (torch.Tensor):
        """
    
    down_block_res_samples: Tuple[torch.Tensor]
    mid_block_res_sample: torch.Tensor

class ControlNetConditioningEmbedding(nn.Module):
    def __init__(
            self,
            conditioning_embedding_channels: int,
            conditioning_channels: int = 3,
            block_out_channels: Tuple[int, ...] = (16, 32, 96, 256)
    ):
        super().__init__()

        self.conv_in = nn.Conv2d(conditioning_embedding_channels, block_out_channels[0], kernel_size=3, padding=1)

        self.blocks = nn.ModuleList([])

        for i in range(len(block_out_channels) - 1):
            channel_in = block_out_channels[i]
            channel_out = block_out_channels[i + 1]
            self.blocks.append(nn.Conv2d(channel_in, channel_in, kernel_size=3, padding=1))
            self.blocks.append(nn.Conv2d(channel_in, channel_out, kernel_size=3, stride=2, padding=1))
        
        self.conv_out = zero_module(
            nn.Conv2d(block_out_channels[-1], conditioning_channels, kernel_size=3, padding=1)
        )

    def forward(self, conditioning):
        embedding = self.conv_in(conditioning)
        embedding = F.silu(embedding)

        for block in self.blocks:
            embedding = block(embedding)
            embedding = F.silu(embedding)

        embedding = self.conv_out(embedding)

        return embedding
    
class ControlNetModel(ModelMixin):
    
    @register_to_config
    def __init__(
        self,
        in_channels: int = 4,                                                                               # 输入通道数
        conditioning_channels: int = 3,
        flip_sin_to_cos: bool = True,
        freq_shift: int = 0,
        down_block_types: Tuple[str, ...] = (                                                               # 下采样 block 的类型
            "CrossAttnDownBlock2D",
            "CrossAttnDownBlock2D",
            "CrossAttnDownBlock2D",
            "DownBlock2D"
        ),
        mid_block_type: Optional[str] = "UNetMidBlock2DCrossAttn",                                          # 中间 block 的类型
        only_cross_attention: Union[bool, Tuple[bool]] = False,                                             # 是否使用 cross attention
        block_out_channels: Tuple[int, ...] = [320, 640, 1280, 1280],                                       # 每个 block 的输出通道数
        layer_per_block: int = 2,                                                                           # 每个 unet block 所含的 resnet 层数
        downsample_padding: int = 1,
        mid_block_scale_factor: int = 1,                                                                    # 缩放 mid block 输出的幅度
        act_fn: str = "silu",                                                                               # 激活函数类型
        norm_num_groups: Optional[int] = 32,
        norm_eps: float = 1e-5,                                                                             # 归一化层防止除零
        cross_attention_dim: int = 1280,                                                                    # 每个 token 的维度
        transformer_layer_per_block: Union[int, Tuple[int, ...]] = 1,
        encoder_hidden_dim: Optional[int] = None,
        encoder_hidden_dim_type: Optional[str] = None,
        attention_head_dim: Union[int, Tuple[int]] = 8,
        num_attention_heads: Optional[Union[int, Tuple[int]]] = None,
        use_linear_projection: bool = False,                                                                # 是否使用线性投影
        class_embed_type: Optional[str] = None,
        addition_emded_type: Optional[str] = None,
        addtion_time_embed_dim: Optional[int] = None,
        num_class_embeds: Optional[int] = None,
        upcast_attention: bool = False,                                                                     # 控制注意力机制中的计算精度             
        resnet_time_scale_shift: str = "default",                   
        projection_class_embedding_input_dim: Optional[int]  = None,
        controlnet_conditioning_channel_order: str = "rgb",
        conditioning_embedding_out_channels: Optional[Tuple[int, ...]] = [16, 32, 96, 256],
        global_pool_conditions: bool = False,
        addtion_embed_type_num_heads: Optional[int] = 64
    ):
        super().__init__()

        num_attention_heads = num_attention_heads or attention_head_dim

        # check input
        if len(block_out_channels) != len(down_block_types):
            raise ValueError(
                f"Must provide the same number of 'block_out_channels' as 'down_block_types'. block_out_channels: {block_out_channels}. down_block_types: {down_block_types}"
            )

        if not isinstance(only_cross_attention, bool) and len(only_cross_attention) != len(down_block_types):
            raise ValueError(
                f"Must provide the same number of 'only_cross_attention' as 'down_block_types'. only_cross_attention: {only_cross_attention}. down_block_types: {down_block_types}"
            )
        
        if not isinstance(num_attention_heads, int) and len(num_attention_heads) != len(down_block_types):
            raise ValueError(
                f"Must provide the same number of 'num_attention_heads' as 'down_block_types'. num_attention_heads: {num_attention_heads}. down_block_types: {down_block_types}"
            )
        
        # input
        # 入口卷积，对齐初始 block 的 channel
        # 保证入口卷积不会改变输入的大小
        conv_in_kernel = 3
        conv_in_padding = (conv_in_kernel - 1) // 2
        self.conv_in = nn.Conv2d(
            in_channels, block_out_channels[0], kernel_size=conv_in_kernel, padding=conv_in_padding
        )

        # time
        # 时间位置编码
        time_embed_dim = block_out_channels[0] * 4
        self.time_proj = Timesteps(block_out_channels[0], flip_sin_to_cos, freq_shift)
        time_input_dim = block_out_channels[0]
        self.time_embedding = TimestepEmbedding(
            time_input_dim,
            time_embed_dim,
            act_fn
        )
        
        # 给出 embedding 维度，没给出接入方式，默认 “text_proj” 方式
        if encoder_hidden_dim_type is None and encoder_hidden_dim is not None:
            encoder_hidden_dim_type = "text_proj"
            # register_to_config()继承自MOdelMixin
            # 将默认值写进模型 config
            self.register_to_config(encoder_hidden_dim_type=encoder_hidden_dim_type)   
            logger.info("encoder_hidden_dim_type default to 'text_proj' as encoder_hidden_dim is defined.")
        
        # 给出接入方式，未给出维度，报错
        if encoder_hidden_dim is None and encoder_hidden_dim_type is not None:
            raise ValueError(
                f"encoder_hidden_dim has to be defined when encoder_hidden_dim_type is set to {encoder_hidden_dim_type}"
            )
        
        # 根据指定的编码器隐藏层类型，创建对应的投影层，将外部特征维度转成 cross attention 需要的维度
        if encoder_hidden_dim_type == "text_proj":
            self.enoder_hid_proj = nn.Linear(encoder_hidden_dim, cross_attention_dim)
        elif encoder_hidden_dim_type == "text_image_proj":
            self.encoder_hid_proj = TextImageProjection(
                text_embed_dim=encoder_hidden_dim,
                image_embed_dim=cross_attention_dim,
                cross_attention_dim=cross_attention_dim
            )
        
        elif encoder_hidden_dim_type is not None:
            raise ValueError(
                f"encoder_hidden_dim_type: {encoder_hidden_dim_type} must be None, 'text_proj' or 'text_image_proj'."
            )
        else:
            self.enoder_hid_proj = None

        # class embedding
        if class_embed_type is None and num_class_embeds is not None:
            # 如果没指定类型，但给定类别数，则默任使用 nn.Embedding
            self.class_embedding = nn.Embedding(num_class_embeds, time_embed_dim)
        elif class_embed_type == "timestep":
            self.class_embedding = TimestepEmbedding(
                time_input_dim,
                time_embed_dim
            )
        elif class_embed_type == "identity":
            self.class_embedding = nn.Identity(time_embed_dim, time_embed_dim)
        elif class_embed_type == "projection":
            if projection_class_embedding_input_dim is None:
                raise ValueError(
                    "class_embed_type: 'projection' requires projection_class_embedding_input_dim be set."
                )
            self.class_embedding = TimestepEmbedding(projection_class_embedding_input_dim, time_embed_dim)
        else:
            self.class_embedding = None

        if addition_emded_type == "text":
            # 如果嵌入类型是 'text'，则设置文本时间嵌入
            if encoder_hidden_dim is not None:
                text_time_embedding_from_dim = encoder_hidden_dim
            else:
                text_time_embedding_from_dim  = cross_attention_dim
            
            self.add_embedding = TextTimeEmbedding(
                text_time_embedding_from_dim, time_embed_dim, num_heads=addtion_embed_type_num_heads
            )
        elif addition_emded_type ==  "text_image":
            self.add_embedding = TextImageTimeEmbedding(
                text_embed_dim=cross_attention_dim,
                image_embed_dim=cross_attention_dim,
                time_embed_dim=time_embed_dim
            )
        elif addition_emded_type == "text_time":
            # 如果嵌入类型是 'text_time'，则处理时间步骤嵌入
            self.add_time_proj = Timesteps(addtion_time_embed_dim, flip_sin_to_cos, freq_shift)
            self.add_embedding = TimestepEmbedding(projection_class_embedding_input_dim, time_embed_dim)
        
        elif addition_emded_type is not None:
            # 如果 'addition_emded_type' 既不是 None，也不是 'text'、'text_image' 或 'text_time'，则抛出 ValueError 异常
            raise ValueError(
                f"addition_emded_type: {addition_emded_type} must be None, 'text' or 'text_image'."
            )
        
        # control net conditioning embedding
        # 处理经过 conv_in 的输入条件图像
        self.controlnet_cond_embedding = ControlNetConditioningEmbedding(
            conditioning_embedding_channels=block_out_channels[0],
            block_out_channels=conditioning_embedding_out_channels,
            conditioning_channels=conditioning_channels
        )

        # 普通 down-block
        self.down_blocks = nn.ModuleList([])
        # 控制网络的 down-block
        self.controlnet_downblocks = nn.ModuleList([])

        # 如果 'only_cross_attention' 是布尔值，转换为布尔列表
        if isinstance(only_cross_attention, bool):
            only_cross_attention = [only_cross_attention] * len(down_block_types)
        
        # 如果 'attention_head_dim' 转换为元组
        if isinstance(attention_head_dim, int):
            attention_head_dim = (attention_head_dim, ) * len(down_block_types)

        # 如果 'num_attention_heads' 转换为元组
        if isinstance(num_attention_heads, int):
            num_attention_heads = (num_attention_heads, ) * len(down_block_types)

        # down
        # 取出下采样阶段第一个块的输出通道数
        output_channels = block_out_channels[0]
        # 创建一个 1x1 卷积，用于处理控制条件（control image）的最顶层特征
        controlnet_block = nn.Conv2d(output_channels, output_channels, kernel_size=1)
        # 将卷积层的权重初始化为 0，使 controlnet 在训练初期不影响原 unet
        controlnet_block = zero_module(controlnet_block)
        # 将该卷积加入 controlnet_downblocks 列表
        self.controlnet_downblocks.append(controlnet_block)

        # 遍历所有下采样阶段
        for i, down_block_type in enumerate(down_block_types):
            # 当前阶段的输入 = 上一阶段的输出
            input_channels = output_channels
            output_channels = block_out_channels[i]
            # 判断是否是最后一个下采样阶段
            is_final_block = i == len(down_block_types) - 1

            down_block = get_down_block(
                down_block_type=down_block_type,                                                                            # 块类型
                num_layers=layer_per_block,                                                                                 # 每个块内 resnet 的层数
                transformer_layers_per_block=transformer_layer_per_block[i],                                                # 该阶段 transformer 层数
                in_channels=input_channels,                                                                                 # 输入通道数
                out_channels=output_channels,                                                                               # 输出通道数
                temb_channels=time_embed_dim,                                                                               # 时间嵌入维度
                add_downsample=not is_final_block,                                                                          # 除最后一个维度，均需要下采样
                resnet_eps=norm_eps,                                                                                        # resnet 归一化
                resnet_act_fn=act_fn,                                                                                       # 激活函数
                resnet_groups=norm_num_groups,                                                                              # groupnorm 组数
                cross_attention_dim=cross_attention_dim,                                                                    # 交叉注意力维度
                num_attention_heads=num_attention_heads[i],                                                                 # 该阶段注意力头数
                attention_head_dim=attention_head_dim[i] if attention_head_dim[i] is not None else output_channels,         # 
                downsample_padding=downsample_padding,                                                                      # 下采样填充
                use_linear_projection=use_linear_projection,                                                                # 使用线性投影
                only_cross_attention=only_cross_attention[i],                                                               # 是否只使用交叉注意力
                upcast_attention=upcast_attention,                                                                          
                resnet_time_scale_shift=resnet_time_scale_shift                
            )
            self.down_blocks.append(down_block)

            # 为每个 resnet 添加对应的零卷积分支
            for _ in range(layer_per_block):
                controlnet_block = nn.Conv2d(output_channels, output_channels, kernel_size=1)
                controlnet_block = zero_module(controlnet_block)
                self.controlnet_downblocks.append(controlnet_block)

            # 如果不是最后一个下采样阶段，需要额外添加一个零卷积
            if not is_final_block:
                controlnet_block = nn.Conv2d(output_channels, output_channels, kernel_size=1)
                controlnet_block = zero_module(controlnet_block)
                self.controlnet_downblocks.append(controlnet_block)

        # mid
        # 下采样阶段最后一层的输出通道数作为 mid block 的通道数
        mid_block_channel = block_out_channels[-1]

        controlnet_block = nn.Conv2d(mid_block_channel, mid_block_channel, kernel_size=1)
        controlnet_block = zero_module(controlnet_block)
        self.controlnet_mid_block = controlnet_block

        if mid_block_type == "UNetMidBlock2DCrossAttn":
            self.mid_block = UNetMidBlock2DCrossAttn(
                transformer_layers_per_block=transformer_layer_per_block[-1],
                in_channels=mid_block_channel,
                temb_channels=time_embed_dim,
                resnet_eps=norm_eps,
                resnet_act_fn=act_fn,
                output_scale_factor=mid_block_scale_factor,
                resnet_time_scale_shift=resnet_time_scale_shift,
                cross_attention_dim=cross_attention_dim,
                num_attention_heads=num_attention_heads[-1],
                resnet_groups=norm_num_groups,
                use_linear_projection=use_linear_projection,
                upcast_attention=upcast_attention
            )
        elif mid_block_type == "UNetMidBlock2D":
            self.mid_block = UNetMidBlock2D(
                in_channels=mid_block_channel,
                temb_channels=time_embed_dim,
                num_layers=0,
                resnet_eps=norm_eps,
                resnet_act_fn=act_fn,
                output_scale_factor=mid_block_scale_factor,
                resnet_groups=norm_num_groups,
                resnet_time_scale_shift=resnet_time_scale_shift,
                add_attention=False
            )
        else:
            raise ValueError(
                f"unknown mid_block_type: {mid_block_type}."
            )
        
    @classmethod
    def from_unet(
        cls,
        unet: UNet2DConditionModel,
        controlnet_conditioning_channel_order: str = "rgb",                                                                 # 控制条件图像的通道顺序
        conditioning_embedding_out_channels: Optional[Tuple[int, ...]] = (16, 32, 96, 256),                                 # 条件嵌入层每个块的输出通道数
        load_weight_from_uent: bool = True,                                                                                 # 是否从传入的 UNet 复制权重到新创建的 ControlNet
        conditioning_channels: int = 3                                                                                      # 条件输入图像的通道数
    ):
        """
        从一个与训练的 UNet2DConditionModel 创建一个 ControlNet 实例
        """
        transformer_layers_per_block = (
            unet.config.transformer_layers_per_block if "transformer_layers_per_block" in unet.config else 1
        )
        encoder_hid_dim = unet.config.encoder_hid_dim if "encoder_hid_dim" in unet.config else None
        encoder_hid_dim_type = unet.config.encoder_hid_dim_type if "encoder_hid_dim_type" in unet.config else None
        addition_embed_type = unet.config.addition_embed_type if "addition_embed_type" in unet.config else None
        addition_time_embed_dim = (
            unet.config.addition_time_embed_dim if "addition_time_embed_dim" in unet.config else None
        )

        controlnet = cls(
            transformer_layers_per_block=transformer_layers_per_block,
            encoder_hid_dim=encoder_hid_dim,
            encoder_hid_dim_type=encoder_hid_dim_type,
            addition_embed_type=addition_embed_type,
            addition_time_embed_dim=addition_time_embed_dim,
            conditioning_channels=conditioning_channels,
            controlnet_conditioning_channel_order=controlnet_conditioning_channel_order,
            conditioning_embedding_out_channels=conditioning_embedding_out_channels,
            in_channels=unet.config.in_channels,
            flip_sin_to_cos=unet.config.flip_sin_to_cos,
            freq_shift=unet.config.freq_shift,
            down_block_types=unet.config.down_block_types,
            only_cross_attention=unet.config.only_cross_attention,
            block_out_channels=unet.config.block_out_channels,
            layers_per_block=unet.config.layers_per_block,
            downsample_padding=unet.config.downsample_padding,
            mid_block_scale_factor=unet.config.mid_block_scale_factor,
            act_fn=unet.config.act_fn,
            norm_num_groups=unet.config.norm_num_groups,
            attention_head_dim=unet.config.attention_head_dim,
            num_attention_heads=unet.config.num_attention_heads,
            use_linear_projection=unet.config.use_linear_projection,
            class_embed_type=unet.config.class_embed_type,
            num_class_embeds=unet.config.num_class_embeds,
            upcast_attention=unet.config.upcast_attention,
            resnet_time_scale_shift=unet.config.resnet_time_scale_shift,
            projection_class_embeddings_input_dim=unet.config.projection_class_embeddings_input_dim,
            mid_block_type=unet.config.mid_block_type
        )

        # 如有需要，从 UNet 复制权重
        if load_weight_from_uent:
            controlnet.conv_in.load_state_dict(unet.conv_in.state_dict())                       # 输入卷积层
            controlnet.time_proj.load_state_dict(unet.time_proj.state_dict())                   # 时间投影
            controlnet.time_embedding.load_state_dict(unet.time_embedding.state_dict())         # 时间嵌入

            if controlnet.class_embedding:
                controlnet.class_embedding.load_state_dict(unet.class_embedding.state_dict())   # 类别嵌入

            controlnet.down_blocks.load_state_dict(unet.down_blocks.state_dict())               # 下采样块
            controlnet.mid_block.load_state_dict(unet.mid_block.state_dict())                   # 中间块

        return controlnet
    
    @property
    def attn_processor(self) -> Dict[str, AttnProcessor]:
        """
        遍历收集模块中所有注意力处理器模块
        并以字典的形式返回

        return: Dict[str, AttnProcessor]
        """

        processors = {}

        def fn_recursive_add_processors(name: str, module: nn.Module, processors: Dict[str, AttnProcessor]):
            # 如果有 get_processor() 方法则说明该模块有注意力处理器模块
            if hasattr(module, "get_processor"):
                processors[f"{name}.processor"] = module.get_processor()
            
            # 递归形式遍历子模块
            for sub_name, child in module.named_children():
                fn_recursive_add_processors(f"{name}.{sub_name}", child, processors)
            
            return processors
        
        for name, module in self.named_children():
            fn_recursive_add_processors(f"{name}", module, processors)

        return processors 

    def set_attn_processor(self, processor: Union[AttnProcessor, Dict[str, AttnProcessor]]):
        """
        设置模型中所有注意力层的处理器（AttnProcessor）
        """

        count = len(self.attn_processor.keys())

        if isinstance(processor, dict) and len(processor) != count:
            raise ValueError(
                f"A dict of processors was passed, but the number of processors {len(processor)} does not match the"
                f"number of attention layers: {count}. please make sure to pass {count} processor class."
            )
        
        def fn_recursive_attn_processor(name: str, module: nn.Module, processor):
            if hasattr(module, "set_processor"):
                if not isinstance(processor, dict):
                    module.set_processor(processor)
                else:
                    module.set_processor(processor.pop(f"{name}.processor"))
            for sub_name, children in module.named_children():
                fn_recursive_attn_processor(f"{name}.{sub_name}", children, processor)

        for name, module in self.named_children():
            fn_recursive_attn_processor(name, module, processor)

    def set_default_attn_processor(self):
        """
        将模型中所有注意力模块的处理器设置为默认类型
        """
        
        # 检查当前所有注意力处理器是否全部属于 ADDED_KV_ATTENTION_PROCESSORS
        # 这类处理器用于支持额外的条件 key/value
        if all(proc.__class__ in ADDED_KV_ATTENTION_PROCESSORS for proc in self.attn_processor.values()):
            # 使用支持 added kv 的默认处理器
            processor = AttnAddedKVProcessor()
        # 检查是否全部属于标准 CROSS_ATTENTION_PROCESSORS、
        # 普通文本到图像扩散模型的默认注意处理器
        elif all(proc.__class__ in CROSS_ATTENTION_PROCESSORS for proc in self.attn_processor.values()):
            # 使用最基础的注意处理器
            processor = AttnProcessor()
        else:
            raise ValueError(
                f"Cannot call 'set_default_attn_processor' when attention processor are of type {next(iter(self.attn_processor.values()))}"
            )
        
        self.set_attn_processor(processor)

    def set_attention_slice(self, slice_size: Union[str, int, List[int]]):
        """
        设置注意力计算的“切片”大小，用于显著降低显存占用
        """
        
        # 收集模型中所有支持 attention slicing 的注意力模块的 head_dim
        sliceable_head_dims = []

        # 递归遍历模块及其子模块，收集支持 set_attention_slice 的模块的 sliceable_head_dims
        def fn_recursive_retrieve_sliceable_dims(module: nn.Module):
            if hasattr(module, "set_attention_slice"):
                sliceable_head_dims.append(module.sliceable_head_dim)

            for child in module.children():
                fn_recursive_retrieve_sliceable_dims(child)
        
        for module in self.children():
            fn_recursive_retrieve_sliceable_dims(module)
        
        # 可切片的注意力层数
        num_sliceable_layers = len(sliceable_head_dims)

        # 处理 slice_size 的不同输入模式
        if slice_size == "auto":
            # 自动模式：每个层切成 head_dim 的一半
            slice_size = [dim // 2 for dim in sliceable_head_dims]
        
        elif slice_size == "max":
            # 最大切片：每次只计算 1 个 head，显存最低
            slice_size = num_sliceable_layers * [1]

        # 如果传入的是单个整数，则扩展成所有层都是用该值
        slice_size = num_sliceable_layers * [slice_size] if not isinstance(slice_size, list) else slice_size

        if len(slice_size) != len(sliceable_head_dims):
            raise ValueError(
                f"You have provided {len(slice_size)}, but {self.config} has {len(sliceable_head_dims)} different attention layers."
                f"please make sure to match 'len(slice_size) to be {len(sliceable_head_dims)}"
            )
        
        # 每个层的 slice_size 不能超过该层的 head_dim
        for i in range(slice_size):
            size = slice_size[i]
            dim = sliceable_head_dims[i]
            if size is not None and size > dim:
                raise ValueError(
                    f"size {size} has to be smaller or equal to {dim}"
                )
        # 递归遍历将 slice_size 应用到每个支持的注意模块
        def fn_recursive_set_attention_slice(module: nn.Module, slice_size: List[int]):
            if hasattr(module, "set_attention_slice"):
                module.set_attention_slice(slice_size.pop())
            
            for child in module.children():
                fn_recursive_set_attention_slice(child, slice_size)

        reversed_slice_size = list(reversed(slice_size))
        for module in self.children():
            fn_recursive_set_attention_slice(module, reversed_slice_size)      

    def forward(
            self,
            sample: torch.Tensor,                                                               # 噪声
            timestep: Union[torch.Tensor, float, int],                                          # 时间步
            encoder_hidden_states: torch.Tensor,                                                # 来自文本编码器的隐藏状态
            controlnet_cond: torch.Tensor,                                                      # 控制图像
            conditioning_scale: float = 1.0,                                                    # ControlNet 输出残差的缩放因子
            class_labels: Optional[torch.Tensor] = None,                                        # 类别标签
            timestep_cond: Optional[torch.Tensor] = None,                                       # 额外时间嵌入         
            attention_mask: Optional[torch.Tensor] = None,
            added_cond_kwargs: Optional[Dict[str, torch.Tensor]] = None,
            cross_attention_kwargs: Optional[Dict[str, Any]] = None,
            guess_mode: bool = False,
            return_dict: bool = True                                                            # 是否以字典形式返回结果
    ) -> Union[ControlNetOutput, Tuple[Tuple[torch.Tensor, ...], torch.Tensor]]:
        
        # check channel order
        channel_order = self.config.controlnet_conditioning_channel_order

        if channel_order == "rgb":
            # 如果配置为 “rgb”，则直接使用控制条件输入，无需通道调整
            ...
        
        elif channel_order == "bgr":
            # 如果是 “bgr”，则需要将通道顺序进行翻转
            controlnet_cond = torch.flip(controlnet_cond, dims=[1])
        
        else:
            # 如果配置了不支持的通道顺序，异常抛出
            raise ValueError(
                f"Unknown 'controlnet_conditioning_channel_order': {channel_order}"
            )
        
        if attention_mask is not None:
            # 将注意力掩码转换为标准负无穷大掩码形式
            attention_mask = (1 - attention_mask.to(sample.dtype)) * -10000.0
            attention_mask = attention_mask.unsqueeze(1)

        # 1. time
        # 处理时间步生成时间嵌入
        # 这里的时间步可能是标量
        timesteps = timestep

        if not torch.is_tensor(timesteps):
            # 标量处理
            is_mps = sample.device.type == "mps"
            is_npu = sample.device.type == "npu"
            if isinstance(timesteps, float):
                dtype = torch.float32 if (is_mps or is_npu) else torch.float64
            else:
                dtype = torch.int32 if (is_mps or is_npu) else torch.int64
            timesteps = torch.tensor([timesteps], device=sample.device, dtype=dtype)
        elif len(timesteps.shape) == 0:
            # 0 维张量处理
            timesteps = timesteps[None].to(sample.device)

        # 广播到 batch 大小
        timesteps = timesteps.expand(sample.shape[0])

        # 时间投影
        t_emb = self.time_proj(timesteps)
        t_emb = t_emb.to(dtype=sample.dtype)

        # 最终时间嵌入
        emb = self.time_embedding(t_emb, timestep_cond)
        aug_emb = None
        
        # 类嵌入
        if self.class_embedding is not None:
            # 如果模型配置了类嵌入，则必须提供 class_labels
            if class_labels is None:
                raise ValueError(
                    f"class_labels shoule be provided when num_class_embeds > 0."
                )
            
            # 如果类嵌入类别为 timestep，则需要额外经过一层时间投影层
            if self.config.class_embed_type == "timestep":
                class_labels = self.time_proj(class_labels)
            
            class_emb = self.class_embedding(class_labels).to(sample.dtype)
            emb = emb + class_emb
            
        # 额外嵌入
        if self.config.addition_embed_type is not None:
            # 只使用文本作为额外嵌入
            if self.config.addition_embed_type == "text":
                # encoder_hidden_states 文本编码器的输出
                aug_emb = self.add_embedding(encoder_hidden_states)
            # 使用文本+时间作为额外嵌入
            elif self.config.addition_embed_type == "text_time":
                # 检查 added_cond_kwargs 是否有 text_embeds
                if "text_embeds" not in added_cond_kwargs:
                    raise ValueError(
                        f"{self.__class__} has the config param 'addition_embed_type' set to 'text_time' which requires the keyword argument 'text_time' to be passed in 'added_cond_kwargs'"
                    )
                text_embeds = added_cond_kwargs["text_embeds"]
                # 检查 added_cond_kwargs 是否有 time_ids
                if "time_ids" not in added_cond_kwargs:
                    raise ValueError(
                        f"{self.__class__} has the config param 'addition_embed_type' set to 'time_ids' which requires the keyword argument 'text_time' to be passed in 'added_cond_kwargs'"
                    )
                time_ids = added_cond_kwargs["time_ids"]
                time_embed = self.add_time_proj(time_ids.flatten())
                time_embed = time_embed.reshape((time_ids.shape[0], -1))
                
                # 拼接 text_embeds 和 time_embed
                add_embed = torch.concat((text_embeds, time_embed), dim=-1)
                add_embed = add_embed.to(emb.dtype)
                aug_emb = self.add_embedding(add_embed)

        emb = emb + aug_emb if aug_emb is not None else emb

        # pre_process
        # 对噪声做卷积预处理
        sample = self.conv_in(sample)

        # 对 controlnet 条件图像做 embedding
        controlnet_cond = self.controlnet_cond_embedding(controlnet_cond)
        # 噪声与条件叠加
        sample = sample + controlnet_cond

        # down
        # 开始下采样阶段
        # 下采样会减小特征图的空间分辨率，同时增加通道数，并保存中间的残差特征，这些残差特征将会在主干 UNet 的上采样阶段使用，恢复细节信息
        down_block_res_samples = (sample, )
        for downsample_block in self.down_blocks:
            if hasattr(downsample_block, "has_cross_attention") and downsample_block.has_cross_attention:
                sample, res_samples = downsample_block(
                    hidden_states=sample,
                    temb=emb,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=attention_mask,
                    cross_attention_kwargs=cross_attention_kwargs
                )
            else:
                # sample 为该 block 的最终输出，res_samples 为 block 残差模块的输出元组
                sample, res_samples = downsample_block(hidden_states=sample, temb=emb)
            
            down_block_res_samples += res_samples
        
        # mid
        # UNet 完成所有下采样后，特征图分辨率最小，通道数最多
        if self.mid_block is not None:
            if hasattr(self.mid_block, "has_cross_attention") and self.mid_block.has_cross_attention:
                sample = self.mid_block(
                    hidden_states=sample,
                    temb=emb,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=attention_mask,
                    cross_attention_kwargs=cross_attention_kwargs
                )
            else:
                sample = self.mid_block(hidden_states=sample, temb=emb)

        # controlnet block
        # controlnet 核心，零卷积，用于生成条件残差特征
        controlnet_down_downblock_res_samples = ()
        
        for down_block_res_sample, controlnet_block in zip(down_block_res_samples, self.controlnet_downblocks):
            down_block_res_sample = controlnet_block(down_block_res_sample)
            controlnet_down_downblock_res_samples = controlnet_down_downblock_res_samples + down_block_res_sample

        down_block_res_samples = controlnet_down_downblock_res_samples

        mid_block_res_sample = self.controlnet_mid_block(sample)

        # scaling
        # controlnet 条件缩放阶段
        if guess_mode and not self.config.global_pool_conditions:
            scales = torch.logspace(-1, 0, len(down_block_res_samples) + 1, device=sample.device)
            scales = scales * conditioning_scale
            down_block_res_samples = [sample * scale for sample, scale in zip(down_block_res_samples, scales)]
            mid_block_res_sample = mid_block_res_sample * scales[-1]
        else:
            down_block_res_samples = [sample * conditioning_scale for sample in down_block_res_samples]
            mid_block_res_sample = mid_block_res_sample * conditioning_scale

        if self.config.global_pool_conditions:
            down_block_res_samples = [
                torch.mean(sample, dim=(2, 3), keepdim=True) for sample in down_block_res_samples
            ]
            mid_block_res_sample = torch.mean(mid_block_res_sample, dim=(2, 3), keepdim=True)

        if not return_dict:
            return (down_block_res_samples, mid_block_res_sample)
        
        return ControlNetOutput(
            down_block_res_samples=down_block_res_samples, mid_block_res_sample=mid_block_res_sample
        )
        

def zero_module(module):
    for p in module.parameters():
        nn.init.zeros_(p)
    return module
