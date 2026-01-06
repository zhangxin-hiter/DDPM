from dataclasses import dataclass
from typing import (
    Tuple,
    Optional,
    Union,
    Dict,
    List,
    Any
)

import torch
import torch.nn as nn
from diffusers.utils.outputs import BaseOutput
from diffusers.utils.logging import get_logger
from diffusers.models.modeling_utils import ModelMixin
from diffusers.configuration_utils import register_to_config
from diffusers.models.embeddings import (
    TextTimeEmbedding,
    Timesteps,
    TimestepEmbedding,
    TextImageProjection,
    TextImageTimeEmbedding
)
from diffusers.models.unets.unet_2d_blocks import (
    get_down_block,
    get_mid_block,
    get_up_block,
    CrossAttnDownBlock2D,
    DownBlock2D
)
from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel
from diffusers.models.attention_processor import (
    AttentionProcessor,
    ADDED_KV_ATTENTION_PROCESSORS,
    AttnAddedKVProcessor,
    CROSS_ATTENTION_PROCESSORS,
    AttnProcessor
)

logger = get_logger(__name__)

@dataclass
class BrushNetOutput(BaseOutput):
    """
    BrushNetModel 的返回值
    """

    up_block_res_samples: Tuple[torch.Tensor]
    dowm_block_res_samples: Tuple[torch.Tensor]
    mid_block_res_sample: torch.Tensor

class BrushNetModel(ModelMixin):
    """
    BrushNet 模型实现
    """

    @register_to_config
    def __init__(
        self,
        in_channels: int = 4,                                               # 噪声输入通道数
        conditioning_channels: int = 5,                                     # 条件图像通道数
        flip_sin_to_cos: bool = True,       
        freq_shift: int = 0,
        down_block_types: Tuple[str, ...] = (                               # 下采样 block 类别
            "DownBlock2D",
            "DownBlock2D",
            "DownBlock2D",
            "DownBlock2D",
        ),
        mid_block_type: Optional[str] = "UNetMidBlock2D",                   # 中间层 block 类别
        up_block_types: Tuple[str, ...] = (                                 # 上采样 block 类别
            "UpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
        ),
        only_cross_attention: Union[bool, Tuple[bool]] = False,
        block_out_channels: Tuple[int, ...] = (320, 640, 1280, 1280),
        layers_per_block: int = 2,                                          # 每个 block 包含多少个 layer
        downsample_padding: int = 1,                                        
        mid_block_scale_factor: float = 1,
        act_fn: str = "silu",                                               # 激活函数
        norm_num_groups: Optional[int] = 32,
        norm_eps: float = 1e-5,
        attention_head_dim: Optional[int] = None,
        num_attention_heads: Union[int, Tuple[int, ...]] = None,
        use_linear_projection: bool = False,
        cross_attention_dim: int = 1280,
        transformer_layers_per_block: Union[int, Tuple[int, ...]] = 1,      # 每个 block 含有多少个 transformer 层
        encoder_hid_dim: Optional[int] = None,
        encoder_hid_dim_type: Optional[str] = None,
        class_embed_type: Optional[str] = None,
        addition_embed_type: Optional[str] = None,
        addition_time_embed_dim: Optional[int] = None,
        num_class_embeds: Optional[int] = None,
        upcast_attention: bool = False,
        resnet_time_scale_shift: str = "default",
        projection_class_embeddings_input_dim: Optional[int] = None,
        brushnet_conditioning_channel_order: str = "rgb",
        conditioning_embedding_out_channels: Optional[Tuple[int, ...]] = (16, 32, 96, 256),
        global_pool_conditions: bool = False,
        addition_embed_type_num_heads: int = 64,
    ):
        super().__init__()

        num_attention_heads = num_attention_heads or attention_head_dim

        # 检查相关输入参数
        if len(down_block_types) != len(up_block_types):
            raise ValueError(
                f"Must provide the same number of 'down_block_types' as 'up_block_types'. 'down_block_types': {down_block_types}. 'up_block_types': {up_block_types}."
            )
        
        if len(down_block_types) != len(block_out_channels):
            raise ValueError(
                f"Must provide the same number of 'down_block_types' as 'block_out_channels'. 'down_block_types': {down_block_types}. 'block_out_channels': {block_out_channels}."
            )
        
        if not isinstance(only_cross_attention, bool) and len(only_cross_attention) != len(down_block_types):
            raise ValueError(
                f"Must provide the same number of 'down_block_types' as 'only_cross_attention'. 'down_block_types': {down_block_types}. 'only_cross_attention': {only_cross_attention}."
            )
        
        if not isinstance(num_attention_heads, int) and len(num_attention_heads) != len(down_block_types):
            raise ValueError(
                f"Must provide the same number of 'down_block_types' as 'num_attention_heads'. 'down_block_types': {down_block_types}. 'num_attention_heads': {num_attention_heads}."
            )
        
        if isinstance(transformer_layers_per_block, int):
            transformer_layers_per_block = [transformer_layers_per_block] * len(down_block_types)
        
        # 输入层
        conv_in_kernel = 3
        conv_in_padding = (conv_in_kernel - 1) // 2
        self.conv_in_condition = nn.Conv2d(
            in_channels=(in_channels + conditioning_channels),
            out_channels=block_out_channels[0],
            kernel_size=conv_in_kernel,
            stride=1,
            padding=conv_in_padding
        )

        # 时间嵌入
        time_embed_dim = block_out_channels[0] * 4
        self.time_proj = Timesteps(block_out_channels[0], flip_sin_to_cos, freq_shift)
        timestep_input_dim = block_out_channels[0]
        self.time_embedding = TimestepEmbedding(
            in_channels=timestep_input_dim,
            time_embed_dim=time_embed_dim,
            act_fn=act_fn
        )

        # 涉及文本嵌入，在 pipeline 中一般会将 [batch_size, seq_len, dim] 转为 [batch_size, hidden_dim]
        if encoder_hid_dim_type is None and encoder_hid_dim is not None:
            encoder_hid_dim_type = "text_proj"
            self.register_to_config(encoder_hid_dim_type=encoder_hid_dim_type)
            logger.info("'encoder_hid_dim_type' defaults to 'text_proj' as 'encoder_hid_dim' is defined.")
        
        if encoder_hid_dim is None and encoder_hid_dim_type is not None:
            raise ValueError(
                f"'encoder_hid_dim' has to be defined when 'encoder_hid_dim_type' is set to {encoder_hid_dim_type}."
            )
        
        if encoder_hid_dim_type == "text_proj":
            self.encoder_hid_proj = nn.Linear(encoder_hid_dim, cross_attention_dim)
        elif encoder_hid_dim_type == "text_image_proj":
            self.encoder_hid_proj = TextImageProjection(
                text_embed_dim=encoder_hid_dim,
                image_embed_dim=cross_attention_dim,
                cross_attention_dim=cross_attention_dim
            )
        elif encoder_hid_dim_type is not None:
            raise ValueError(
                f"'encoder_hid_dim_type': {encoder_hid_dim_type} must be None, 'text_proj' or 'text_image_proj'."
            )
        else: 
            self.encoder_hid_proj = None

        # 类别嵌入
        if class_embed_type is None and num_class_embeds is not None:
            self.class_embedding = nn.Embedding(num_class_embeds, time_embed_dim)
        elif class_embed_type == "timestep":
            self.class_embedding = TimestepEmbedding(timestep_input_dim, time_embed_dim)
        elif class_embed_type == "identity":
            self.class_embedding = nn.Identity(time_embed_dim, time_embed_dim)
        elif class_embed_type == "projection":
            if projection_class_embeddings_input_dim is None:
                raise ValueError(
                    f"'class_embed_type': 'projection' requires 'projection_class_embeddings_input_dim' to set."
                )
            self.class_embedding = TimestepEmbedding(projection_class_embeddings_input_dim, time_embed_dim)
        else:
            self.class_embedding = None
        
        # 额外嵌入
        if addition_embed_type == "text":
            if encoder_hid_dim is not None:
                text_time_embedding_from_dim = encoder_hid_dim
            else:
                text_time_embedding_from_dim = cross_attention_dim
            
            self.add_embedding = TextTimeEmbedding(
                text_time_embedding_from_dim, time_embed_dim, num_heads=addition_embed_type_num_heads
            )
        elif addition_embed_type == "text_image":
            self.add_embedding = TextImageTimeEmbedding(
                text_embed_dim=cross_attention_dim, image_embed_dim=cross_attention_dim, time_embed_dim=time_embed_dim
            )
        elif addition_embed_type == "text_time":
            self.add_time_proj = Timesteps(addition_time_embed_dim, flip_sin_to_cos, freq_shift)
            self.add_embedding = TimestepEmbedding(projection_class_embeddings_input_dim, time_embed_dim)

        elif addition_embed_type is not None:
            raise ValueError(f"'addition_embed_type': {addition_embed_type} must be None, 'text' or 'text_image'.")

        self.down_blocks = nn.ModuleList([])
        self.brushnet_down_blocks = nn.ModuleList([])

        if isinstance(only_cross_attention, int):
            only_cross_attention = [only_cross_attention] * len(down_block_types)

        if isinstance(attention_head_dim, int):
            attention_head_dim = [attention_head_dim] * len(attention_head_dim)

        if isinstance(num_attention_heads, int):
            num_attention_heads = [num_attention_heads] * len(num_attention_heads)

        # 下采样阶段
        output_channel = block_out_channels[0]

        brushnet_block = nn.Conv2d(output_channel, output_channel, kernel_size=1)
        brushnet_block = zero_module(brushnet_block)
        self.brushnet_down_blocks.append(brushnet_block)

        for i, down_block_type in enumerate(down_block_types):
            input_channel = output_channel
            output_channel = block_out_channels[i]
            is_final_block = i == len(block_out_channels) - 1

            down_block = get_down_block(
                down_block_type=down_block_types[i],
                num_layers=layers_per_block,
                in_channels=input_channel,
                out_channels=output_channel,
                temb_channels=time_embed_dim,
                add_downsample=not is_final_block,
                resnet_eps=norm_eps,
                resnet_act_fn=act_fn,
                transformer_layers_per_block=transformer_layers_per_block[i],
                num_attention_heads=num_attention_heads[i],
                resnet_groups=norm_num_groups,
                cross_attention_dim=cross_attention_dim,
                downsample_padding=downsample_padding,
                use_linear_projection=use_linear_projection,
                only_cross_attention=only_cross_attention[i],
                upcast_attention=upcast_attention,
                resnet_time_scale_shift=resnet_time_scale_shift,
                attention_head_dim=attention_head_dim[i] if attention_head_dim[i] is not None else output_channel,
            )
            self.down_blocks.append(down_block)

            for _ in range(layers_per_block):
                brushnet_block = nn.Conv2d(block_out_channels[i], block_out_channels[i], kernel_size=1)
                brushnet_block = zero_module(brushnet_block)
                self.brushnet_down_blocks.append(brushnet_block)

            if not is_final_block:
                brushnet_block = nn.Conv2d(block_out_channels[i], block_out_channels[i], kernel_size=1)
                brushnet_block = zero_module(brushnet_block)
                self.brushnet_down_blocks.append(brushnet_block)

        # 中间层
        mid_block_channel = block_out_channels[-1]

        brushnet_block = nn.Conv2d(mid_block_channel, mid_block_channel, kernel_size=1)
        brushnet_block = zero_module(brushnet_block)
        self.brushnet_mid_block = brushnet_block

        self.mid_block = get_mid_block(
            mid_block_type=mid_block_type,
            temb_channels=time_embed_dim,
            in_channels=mid_block_channel,
            resnet_eps=norm_eps,
            resnet_act_fn=act_fn,
            resnet_groups=norm_num_groups,
            output_scale_factor=mid_block_scale_factor,
            transformer_layers_per_block=transformer_layers_per_block[-1],
            num_attention_heads=num_attention_heads[-1],
            cross_attention_dim=cross_attention_dim,
            use_linear_projection=use_linear_projection,
            upcast_attention=upcast_attention,
            resnet_time_scale_shift=resnet_time_scale_shift,
        )

        self.num_upsamples = 0

        # 上采样阶段
        reversed_block_out_channels = list(reversed(block_out_channels))
        reversed_num_attention_heads = list(reversed(num_attention_heads))
        reversed_tranformer_layers_per_block = list(reversed(transformer_layers_per_block))
        only_cross_attention = list(reversed(only_cross_attention))

        output_channel = reversed_block_out_channels[0]

        self.up_blocks = nn.ModuleList([])
        self.brushnet_up_blocks = nn.ModuleList([])

        for i, up_block_type in enumerate(up_block_types):
            is_final_block = i == len(block_out_channels) - 1

            prev_output_channel = output_channel
            output_channel = reversed_block_out_channels[i]
            input_channel = reversed_block_out_channels[min(i + 1, len(block_out_channels) - 1)]

            if not is_final_block:
                add_upsample = True
                self.num_upsamples += 1
            else:
                add_upsample = False

            up_block = get_up_block(
                up_block_type=up_block_types[i],
                num_layers=layers_per_block,
                in_channels=input_channel,
                out_channels=output_channel,
                prev_output_channel=prev_output_channel,
                temb_channels=time_embed_dim,
                add_upsample=add_upsample,
                resnet_eps=norm_eps,
                resnet_act_fn=act_fn,
                resolution_idx=i,
                transformer_layers_per_block=reversed_tranformer_layers_per_block[i],
                num_attention_heads=reversed_num_attention_heads[i],
                resnet_groups=norm_num_groups,
                cross_attention_dim=cross_attention_dim,
                use_linear_projection=use_linear_projection,
                only_cross_attention=only_cross_attention[i],
                upcast_attention=upcast_attention,
                resnet_out_scale_factor=resnet_time_scale_shift,
                attention_head_dim=attention_head_dim[i],
            )
            self.up_blocks.append(up_block)
            prev_output_channel = output_channel

            for _ in range(layers_per_block):
                brushnet_block = nn.Conv2d(output_channel, output_channel, kernel_size=1)
                brushnet_block = zero_module(brushnet_block)
                self.brushnet_up_blocks.append(brushnet_block)

            if not is_final_block:
                brushnet_block = nn.Conv2d(output_channel, output_channel, kernel_size=1)
                brushnet_block = zero_module(brushnet_block)
                self.brushnet_up_blocks.append(brushnet_block)

    @classmethod
    def from_unet(
        cls,
        unet: UNet2DConditionModel,
        brushnet_conditioning_channel_order: str = "rgb",
        conditioning_embedding_out_channels: Optional[Tuple[int, ...]] = [16, 32, 96, 256],
        load_weight_from_unet: bool = True,
        conditioning_channels: int = 5,
    ):
        transformer_layers_per_block = (
            unet.config.transformer_layers_per_block if "transformer_layers_per_block" in unet.config else 1
        )
        encoder_hid_dim = unet.config.encoder_hid_dim if "encoder_hid_dim" in unet.config else None
        encoder_hid_dim_type = unet.config.encoder_hid_dim_type if "encoder_hid_dim_type" in unet.config else None
        addition_embed_type = unet.config.addition_embed_type if "addition_embed_type" in unet.config else None
        addition_time_embed_dim = (
            unet.cinfig.addition_time_embed_dim if "addition_time_embed_dim" in unet.config else None
        )

        brushnet = cls(
            in_channels = unet.config.in_channels,
            conditioning_channels=conditioning_channels,
            flip_sin_to_cos=unet.config.flip_sin_to_cos,
            freq_shift=unet.config.freq_shift,
            down_block_types=["DownBlock2D" for block_name in unet.config.down_block_types],
            mid_Block_type="MidBlock2D",
            up_block_types=["UpBlock2D" for block_name in unet.config.up_block_types],
            only_cross_attention=unet.config.only_cross_attention,
            block_out_channels=unet.config.block_out_channels,
            layers_per_block=unet.config.layers_per_block,
            downsample_padding=unet.config.downsample_padding,
            mid_block_scale_factor=unet.config.mid_block_scale_factor,
            act_fn=unet.config.act_fn,
            norm_num_groups=unet.config.norm_num_groups,
            norm_eps=unet.config.norm_eps,
            cross_attention_dim=unet.config.cross_attention_dim,
            transformer_layers_per_block=transformer_layers_per_block,
            encoder_hid_dim=encoder_hid_dim,
            encoder_hid_dim_type=encoder_hid_dim_type,
            attention_head_dim=unet.config.attention_head_dim,
            num_attention_heads=unet.config.num_attention_heads,
            use_linear_projection=unet.config.use_linear_projection,
            class_embed_type=unet.config.class_embed_type,
            addition_embed_type=addition_embed_type,
            addition_time_embed_dim=addition_time_embed_dim,
            num_class_embeds=unet.config.num_class_embeds,
            upcast_attention=unet.config.upcast_attention,
            resnet_time_scale_shift=unet.config.resnet_time_scale_shift,
            projection_class_embeddings_input_dim=unet.config.projection_class_embeddings_input_dim,
            brushnet_conditioning_channel_order=brushnet_conditioning_channel_order,
            conditioning_embedding_out_channels=conditioning_embedding_out_channels,
        )

        if load_weight_from_unet:
            conv_in_condition_weight = torch.zeros_like(brushnet.conv_in_condition.weight)
            conv_in_condition_weight[:,:4,...] = unet.conv_in.weight
            conv_in_condition_weight[:,4:8,...] = unet.conv_in.weight
            brushnet.conv_in_condition.weight = torch.nn.parameter(conv_in_condition_weight)
            brushnet.conv_in_condition.bias = unet.conv_in.bias

            brushnet.time_proj.load_state_dict(unet.time_proj.state_dict())
            brushnet.time_embedding.load_state_dict(unet.time_embedding.state_dict())

            if brushnet.class_embedding:
                brushnet.class_embedding.load_state_dict(unet.class_embedding.state_dict())

            brushnet.down_blocks.load_state_dict(unet.down_blocks.state_dict())
            brushnet.mid_block.load_state_dict(unet.mid_block.state_dict())
            brushnet.up_blocks.load_state_dict(unet.up_blocks.state_dict())

        return brushnet
    
    @property
    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.attn_processors
    def attn_processors(self) -> Dict[str, AttentionProcessor]:
        r"""
        Returns:
            `dict` of attention processors: A dictionary containing all attention processors used in the model with
            indexed by its weight name.
        """
        # set recursively
        processors = {}

        def fn_recursive_add_processors(name: str, module: torch.nn.Module, processors: Dict[str, AttentionProcessor]):
            if hasattr(module, "get_processor"):
                processors[f"{name}.processor"] = module.get_processor(return_deprecated_lora=True)

            for sub_name, child in module.named_children():
                fn_recursive_add_processors(f"{name}.{sub_name}", child, processors)

            return processors

        for name, module in self.named_children():
            fn_recursive_add_processors(name, module, processors)

        return processors

    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.set_attn_processor
    def set_attn_processor(self, processor: Union[AttentionProcessor, Dict[str, AttentionProcessor]]):
        r"""
        Sets the attention processor to use to compute attention.

        Parameters:
            processor (`dict` of `AttentionProcessor` or only `AttentionProcessor`):
                The instantiated processor class or a dictionary of processor classes that will be set as the processor
                for **all** `Attention` layers.

                If `processor` is a dict, the key needs to define the path to the corresponding cross attention
                processor. This is strongly recommended when setting trainable attention processors.

        """
        count = len(self.attn_processors.keys())

        if isinstance(processor, dict) and len(processor) != count:
            raise ValueError(
                f"A dict of processors was passed, but the number of processors {len(processor)} does not match the"
                f" number of attention layers: {count}. Please make sure to pass {count} processor classes."
            )

        def fn_recursive_attn_processor(name: str, module: torch.nn.Module, processor):
            if hasattr(module, "set_processor"):
                if not isinstance(processor, dict):
                    module.set_processor(processor)
                else:
                    module.set_processor(processor.pop(f"{name}.processor"))

            for sub_name, child in module.named_children():
                fn_recursive_attn_processor(f"{name}.{sub_name}", child, processor)

        for name, module in self.named_children():
            fn_recursive_attn_processor(name, module, processor)

    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.set_default_attn_processor
    def set_default_attn_processor(self):
        """
        Disables custom attention processors and sets the default attention implementation.
        """
        if all(proc.__class__ in ADDED_KV_ATTENTION_PROCESSORS for proc in self.attn_processors.values()):
            processor = AttnAddedKVProcessor()
        elif all(proc.__class__ in CROSS_ATTENTION_PROCESSORS for proc in self.attn_processors.values()):
            processor = AttnProcessor()
        else:
            raise ValueError(
                f"Cannot call `set_default_attn_processor` when attention processors are of type {next(iter(self.attn_processors.values()))}"
            )

        self.set_attn_processor(processor)

    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.set_attention_slice
    def set_attention_slice(self, slice_size: Union[str, int, List[int]]) -> None:
        r"""
        Enable sliced attention computation.

        When this option is enabled, the attention module splits the input tensor in slices to compute attention in
        several steps. This is useful for saving some memory in exchange for a small decrease in speed.

        Args:
            slice_size (`str` or `int` or `list(int)`, *optional*, defaults to `"auto"`):
                When `"auto"`, input to the attention heads is halved, so attention is computed in two steps. If
                `"max"`, maximum amount of memory is saved by running only one slice at a time. If a number is
                provided, uses as many slices as `attention_head_dim // slice_size`. In this case, `attention_head_dim`
                must be a multiple of `slice_size`.
        """
        sliceable_head_dims = []

        def fn_recursive_retrieve_sliceable_dims(module: torch.nn.Module):
            if hasattr(module, "set_attention_slice"):
                sliceable_head_dims.append(module.sliceable_head_dim)

            for child in module.children():
                fn_recursive_retrieve_sliceable_dims(child)

        # retrieve number of attention layers
        for module in self.children():
            fn_recursive_retrieve_sliceable_dims(module)

        num_sliceable_layers = len(sliceable_head_dims)

        if slice_size == "auto":
            # half the attention head size is usually a good trade-off between
            # speed and memory
            slice_size = [dim // 2 for dim in sliceable_head_dims]
        elif slice_size == "max":
            # make smallest slice possible
            slice_size = num_sliceable_layers * [1]

        slice_size = num_sliceable_layers * [slice_size] if not isinstance(slice_size, list) else slice_size

        if len(slice_size) != len(sliceable_head_dims):
            raise ValueError(
                f"You have provided {len(slice_size)}, but {self.config} has {len(sliceable_head_dims)} different"
                f" attention layers. Make sure to match `len(slice_size)` to be {len(sliceable_head_dims)}."
            )

        for i in range(len(slice_size)):
            size = slice_size[i]
            dim = sliceable_head_dims[i]
            if size is not None and size > dim:
                raise ValueError(f"size {size} has to be smaller or equal to {dim}.")

        # Recursively walk through all the children.
        # Any children which exposes the set_attention_slice method
        # gets the message
        def fn_recursive_set_attention_slice(module: torch.nn.Module, slice_size: List[int]):
            if hasattr(module, "set_attention_slice"):
                module.set_attention_slice(slice_size.pop())

            for child in module.children():
                fn_recursive_set_attention_slice(child, slice_size)

        reversed_slice_size = list(reversed(slice_size))
        for module in self.children():
            fn_recursive_set_attention_slice(module, reversed_slice_size)
    
    def _set_gradient_checkpointing(self, module, value: bool = False):
        if isinstance(module, (CrossAttnDownBlock2D, DownBlock2D)):
            module.gradient_checkpointing = value

    def forward(
            self,
            sample: torch.FloatTensor,
            timestep: Union[torch.Tensor, float, int],
            encoder_hidden_states: torch.Tensor,
            brushnet_cond: torch.FloatTensor,
            conditioning_scale: float = 1.0,
            class_labels: Optional[torch.Tensor] = None,
            timestep_cond: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            added_cond_kwargs: Optional[Dict[str, torch.Tensor]] = None,
            cross_attention_kwargs: Optional[Dict[str, Any]] = None,
            guess_mode: bool = False,
            return_dict: bool = False
    ) -> Union[BrushNetOutput, Tuple[Tuple[torch.FloatTensor, ...], torch.FloatTensor]]:
        
        # 检查通道顺序
        channel_order = self.config.brushnet_conditioning_channel_order

        if channel_order == "rgb":
            ...
        elif channel_order == "bgr":
            brushnet_cond = torch.flip(brushnet_cond, dims=(1))
        else:
            raise ValueError(
                f"Unkown 'brushnet_conditioning_channel_order': {channel_order}."
            )
        
        # 1. 时间嵌入
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            is_mps = sample.device.type == "mps"
            if isinstance(timesteps, float):
                dtype = torch.float32 if is_mps else torch.float64
            else:
                dtype = torch.int32 if is_mps else torch.int64
            timesteps = torch.tensor([timesteps], dtype=dtype, device=sample.device)
        elif len(Timesteps) == 0:
            timesteps = timesteps[None].to(sample.device)
        
        timesteps = timesteps.expand(sample.shape[0])

        t_emb = self.time_proj(timesteps)

        t_emb = t_emb.to(dtype=sample.dtype)

        emb = self.time_embedding(t_emb, timestep_cond)
        aug_emb = None

        # 2. 类别嵌入
        if self.class_embedding is not None:
            if class_labels is None:
                raise ValueError(
                    "'class_labels' should be provided when 'num_class_embeds' > 0."
                )
            
            if self.config.class_embed_type == "timestep":
                class_labels = self.time_proj(class_labels)
            
            class_emb = self.class_embedding(class_labels).to(dtype=self.dtype)
            emb = emb + class_emb
        
        # 3. 额外嵌入
        if self.config.addition_embed_type is not None:
            if self.config.addition_embed_type == "text":
                aug_emb = self.add_embedding(encoder_hidden_states)
            
            elif self.config.addition_embed_type == "text_time":
                if "text_embeds" not in added_cond_kwargs:
                    raise ValueError(
                        f"{self.__class__} has the config param 'addition_embed_type' set to 'text_time' which requires the keyword argument 'text_embeds' to be passed in 'added_cond_kwargs'."
                    )
                text_embeds = added_cond_kwargs.get("text_embeds")
                if "time_ids" not in added_cond_kwargs:
                    raise ValueError(
                        f"{self.__class__} has the config param 'addition_embed_type' set to 'time_ids' which requires the keyword argument 'time_ids' to be passed in 'added_cond_kwargs'."
                    )
                time_ids = added_cond_kwargs.get("time_ids")
                time_embeds = self.add_time_proj(time_ids.flatten())
                time_embeds = time_embeds.reshape((time_embeds.shape[0], -1))

                add_embeds = torch.concat([text_embeds, time_embeds], dim=-1)
                add_embeds = add_embeds.to(emb.dtype)
                aug_emb = self.add_embedding(add_embeds)
            
        emb = emb + aug_emb if aug_emb is not None else emb

        # 4. 预处理
        brushnet_cond = torch.concat([sample, brushnet_cond], 1)
        sample = self.conv_in_condition(brushnet_cond)

        # 5. 下采样
        down_block_res_samples = (sample)
        for downsample_block in self.down_blocks:
            if hasattr(downsample_block, "has_cross_attention") and downsample_block.has_cross_attention is not None:
                sample, res_sample = downsample_block(
                    hidden_states=sample,
                    temb=emb,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=attention_mask,
                    cross_attention_kwargs=cross_attention_kwargs
                )
            else:
                sample, res_sample = downsample_block(hidden_states=sample, temb=emb)
            
            down_block_res_samples += res_sample

        # 6. brushnet 下采样
        brushnet_down_block_res_samples = ()
        for down_block_res_sample, brushnet_down_block in zip(down_block_res_samples, downsample_block):
            down_block_res_sample = brushnet_down_block(down_block_res_sample)
            brushnet_down_block_res_samples = brushnet_down_block_res_samples + (down_block_res_sample, )

        # 7. 中间层
        if self.mid_block is not None:
            if hasattr(self.mid_block, "has_cross_attention"):
                sample = self.mid_block(
                    sample,
                    emb, 
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=attention_mask,
                    cross_attention_kwargs=cross_attention_kwargs
                )
            else:
                sample = self.mid_block(sample, emb)
            
        # 8. brushnet 中间层
        brushnet_mid_block_res_sample = self.brushnet_mid_block(sample)

        # 9. 上采样
        up_block_res_samples = ()
        for i, upsample_block in enumerate(self.up_blocks):
            is_final_block = i == len(self.up_blocks) - 1

            # skip connection
            # 在 down_block_res_samples 中取出与当前 upsample_block.resnets 数量匹配的特征
            res_samples = down_block_res_samples[-len(upsample_block.resnets):]
            # 特征取出后更新 down_block_res_samples
            down_block_res_samples = down_block_res_samples[:-len(upsample_block.resnets)]

            # 如果不是最后一层上采样块，需要明确指定上采样后的目标空间尺寸
            if not is_final_block:
                upsample_size = down_block_res_samples[-1].shape[2:]
            
            if hasattr(upsample_block, "has_cross_attention") and upsample_block.has_cross_attention is not None:
                sample, up_res_samples = upsample_block(
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=res_samples,
                    encoder_hidden_states=encoder_hidden_states,
                    upsample_size=upsample_size,
                    attention_mask=attention_mask,
                    return_res_sample=True
                )
            else:
                sample, up_res_samples = upsample_block(
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=res_samples,
                    upsample_size=upsample_size,
                    return_res_samples=True
                )
            up_block_res_samples = up_block_res_samples + up_res_samples

        # 10 brushnet 上采样
        brushnet_up_block_res_samples = ()
        for up_block_res_sample, brushnet_up_block in zip(up_block_res_samples, self.brushnet_up_blocks):
            up_block_res_sample = brushnet_up_block(up_block_res_sample)
            brushnet_down_block_res_samples = brushnet_down_block_res_samples + up_block_res_sample

        # 11. scaling
        if guess_mode and not self.config.global_pool_conditions:
            scales = torch.logspace(-1, 0, len(brushnet_down_block_res_samples) + 1 + len(brushnet_up_block_res_samples), device=sample.device)
            scales = scales * conditioning_scale

            brushnet_down_block_res_samples = [sample * scale for sample ,scale in zip(brushnet_down_block_res_samples, scales[:len(brushnet_down_block_res_samples)])]
            brushnet_mid_block_res_sample = brushnet_down_block_res_samples * scales[len(brushnet_down_block_res_samples)]
            brushnet_up_block_res_samples = [sample * scale for sample, scale in zip(brushnet_up_block_res_samples, scales[len(brushnet_down_block_res_samples) + 1: ])]

        else:
            brushnet_down_block_res_samples = [sample * conditioning_scale for sample in brushnet_down_block_res_samples]
            brushnet_mid_block_res_sample = brushnet_mid_block_res_sample * conditioning_scale
            brushnet_up_block_res_samples = [sample * conditioning_scale for sample in brushnet_up_block_res_samples]

        if not return_dict:
            return (brushnet_down_block_res_samples, brushnet_mid_block_res_sample, brushnet_up_block_res_samples)
        
        return BrushNetOutput(
            up_block_res_samples=brushnet_up_block_res_samples,
            mid_block_res_sample=brushnet_mid_block_res_sample,
            down_block_res_samples=brushnet_down_block_res_samples
        )

def zero_module(module):
    for p in module.parameters():
        nn.init.zeros_(p)
    return module
