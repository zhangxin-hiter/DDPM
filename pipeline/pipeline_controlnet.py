from ast import Dict
import sys
import inspect
from typing import (
    Any,
    Optional,
    Union,
    List,
    Tuple,
    Callable
)
import PIL

import numpy as np
from pyparsing import Opt
import torch
import torch.nn.functional as F
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
from diffusers.image_processor import PipelineImageInput
from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel
from diffusers.pipelines.stable_diffusion.pipeline_output import StableDiffusionPipelineOutput
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from diffusers.schedulers.scheduling_utils import KarrasDiffusionSchedulers
from diffusers.models.embeddings import ImageProjection
from diffusers.utils import (
    logging,
    deprecate,
    USE_PEFT_BACKEND,
    scale_lora_layers,
    unscale_lora_layers,
    is_torch_xla_available
)
from diffusers.utils.torch_utils import (
    randn_tensor,
    is_compiled_module,
    is_torch_version,
    empty_device_cache
)
from diffusers.models.lora import adjust_lora_scale_text_encoder
from diffusers.image_processor import VaeImageProcessor
from diffusers.loaders import StableDiffusionLoraLoaderMixin, TextualInversionLoaderMixin
from diffusers.callbacks import PipelineCallback
from transformers.models.clip.modeling_clip import (
    CLIPTextModel,
    CLIPVisionModelWithProjection
)
from transformers.models.clip.tokenization_clip import CLIPTokenizer
from transformers.models.clip.image_processing_clip import CLIPImageProcessor

from model.diffusion.controlnet import ControlNetModel

if is_torch_xla_available():
    import torch_xla.core.xla_model as xm

    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False

logger = logging.get_logger(__name__)

def retrieve_timesteps(
        scheduler,                                                      # 当前使用的调度器
        num_inference_steps: Optional[int] = None,                      # 推理步数
        device: Optional[Union[str, torch.device]] = None,              # 指定设备
        timesteps: Optional[List[int]] = None,                          # 自定义的时间步列表
        sigmas: Optional[List[float]] = None,                           # 自定义的标准差列表
        **kwargs,                                                       # 传给 scheduler.set_timesteps() 的附加参数
):
    
    if timesteps is not None and sigmas is not None:
        raise ValueError(
            "Only one of 'timesteps' or 'sigmas' can be passed."
        )
    
    elif timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s 'set_timesteps' does not support custom"
                f"timestep schedulers. Please check whether you are using the correct scheduler"
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    
    elif sigmas is not None:
        accepts_timesteps = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s 'set_timesteps' does not support custom"
                f"sigmas schedulers. Please check whether you are using the correct scheduler"
            )
        scheduler.set_timesteps(timesteps=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    
    return timesteps, num_inference_steps

class StableDiffusionControlNetPipeline(
    DiffusionPipeline,
    StableDiffusionLoraLoaderMixin,
    TextualInversionLoaderMixin
):
    def __init__(
            self,
            vae: AutoencoderKL,
            text_encoder: CLIPTextModel,
            tokenizer: CLIPTokenizer,
            unet: UNet2DConditionModel,
            controlnet: Union[ControlNetModel, List[ControlNetModel], Tuple[ControlNetModel]],
            scheduler: KarrasDiffusionSchedulers,
            safety_checker: StableDiffusionSafetyChecker,
            feature_extractor: CLIPImageProcessor,
            image_encoder: CLIPVisionModelWithProjection = None,
            require_safety_checker: bool = True
    ):
        
        super().__init__()

        if safety_checker is None and require_safety_checker:
            logger.warning(
                f"You have disabled the safety checker for {self.__class__} by passing `safety_checker=None`. Ensure"
                " that you abide to the conditions of the Stable Diffusion license and do not expose unfiltered"
                " results in services or applications open to the public. Both the diffusers team and Hugging Face"
                " strongly recommend to keep the safety filter enabled in all public facing circumstances, disabling"
                " it only for use-cases that involve analyzing network behavior or auditing its results. For more"
                " information, please have a look at https://github.com/huggingface/diffusers/pull/254 ."
            )

        if safety_checker is not None and feature_extractor is None:
            raise ValueError(
                f"Make sure to define a feature extractor when loading {self.__class__} if you want to use the safety"
                "checker. if you do not want to use the safety checker, you can pass 'safety_checker=None' instead"
            )
        
        # 注册模块
        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            controlnet=controlnet,
            scheduler=scheduler,
            safety_checker=safety_checker,
            feature_extractor=feature_extractor,
            image_encoder=image_encoder
        )
        # vae 缩放因子
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1) if getattr(self, "vae", None) else 8
        # 用于将图像转化为 vae 输入，并确保为 RGB 格式
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor, do_convert_rgb=True)
        # 用于处理controlnet的图像
        self.control_image_processor = VaeImageProcessor(
            vae_scale_factor=self.vae_scale_factor, do_convert_rgb=True, do_normalize=False
        )

        self.register_to_config(require_safety_checker=require_safety_checker)

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline._encode_prompt
    def _encode_prompt(
        self,
        prompt,
        device,
        num_images_per_prompt,
        do_classifier_free_guidance,
        negative_prompt=None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        lora_scale: Optional[float] = None,
        **kwargs,
    ):
        deprecation_message = "`_encode_prompt()` is deprecated and it will be removed in a future version. Use `encode_prompt()` instead. Also, be aware that the output format changed from a concatenated tensor to a tuple."
        deprecate("_encode_prompt()", "1.0.0", deprecation_message, standard_warn=False)

        prompt_embeds_tuple = self.encode_prompt(
            prompt=prompt,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            do_classifier_free_guidance=do_classifier_free_guidance,
            negative_prompt=negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            lora_scale=lora_scale,
            **kwargs,
        )

        # concatenate for backwards comp
        prompt_embeds = torch.cat([prompt_embeds_tuple[1], prompt_embeds_tuple[0]])

        return prompt_embeds

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.encode_prompt
    def encode_prompt(
        self,
        prompt,                                                     # 正向提示词
        device,                                                     # 目标设备
        num_images_per_prompt,                                      # 每个提示词生成多少张图
        do_classifier_free_guidance,                                # 是否使用无分类器引导（CFG）
        negative_prompt=None,                                       # 负向提示词
        prompt_embeds: Optional[torch.Tensor] = None,               # 预计算的正向提示词嵌入
        negative_prompt_embeds: Optional[torch.Tensor] = None,      # 预计算的负向提示词嵌入
        lora_scale: Optional[float] = None,                         # LoRA 权重缩放系数
        clip_skip: Optional[int] = None,                            # CLIP 跳层数
    ):
        r"""
        Encodes the prompt into text encoder hidden states.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            device: (`torch.device`):
                torch device
            num_images_per_prompt (`int`):
                number of images that should be generated per prompt
            do_classifier_free_guidance (`bool`):
                whether to use classifier free guidance or not
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            lora_scale (`float`, *optional*):
                A LoRA scale that will be applied to all LoRA layers of the text encoder if LoRA layers are loaded.
            clip_skip (`int`, *optional*):
                Number of layers to be skipped from CLIP while computing the prompt embeddings. A value of 1 means that
                the output of the pre-final layer will be used for computing the prompt embeddings.
        """
        # set lora scale so that monkey patched LoRA
        # function of text encoder can correctly access it
        # LoRA 权重缩放处理
        if lora_scale is not None and isinstance(self, StableDiffusionLoraLoaderMixin):
            self._lora_scale = lora_scale

            # dynamically adjust the LoRA scale
            if not USE_PEFT_BACKEND:
                adjust_lora_scale_text_encoder(self.text_encoder, lora_scale)
            else:
                scale_lora_layers(self.text_encoder, lora_scale)

        # 确定 batch size
        # 根据输入类型判断批次大小
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        # 正向提示词编码
        if prompt_embeds is None:
            # textual inversion: process multi-vector tokens if necessary
            if isinstance(self, TextualInversionLoaderMixin):
                prompt = self.maybe_convert_prompt(prompt, self.tokenizer)

            # 使用 clip tokenizer 将文本转为 token ids
            text_inputs = self.tokenizer(
                prompt,
                padding="max_length",                           # 填充到模型最大长度
                max_length=self.tokenizer.model_max_length,     
                truncation=True,                                # 超出长度自动截断
                return_tensors="pt",                            # 是否返回张量
            )
            # 形状：[batch_size, 77]
            text_input_ids = text_inputs.input_ids

            # 检查是否发生截断，并发出警告
            untruncated_ids = self.tokenizer(prompt, padding="longest", return_tensors="pt").input_ids
            if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
                text_input_ids, untruncated_ids
            ):
                # 解码被截断的部分
                removed_text = self.tokenizer.batch_decode(
                    untruncated_ids[:, self.tokenizer.model_max_length - 1 : -1]
                )
                logger.warning(
                    "The following part of your input was truncated because CLIP can only handle sequences up to"
                    f" {self.tokenizer.model_max_length} tokens: {removed_text}"
                )

            # 是否使用 attension_mask
            if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
                attention_mask = text_inputs.attention_mask.to(device)
            else:
                attention_mask = None

            # clip_skip：使用更浅层的隐藏状态
            if clip_skip is None:
                # 正常情况：直接取text_encoder 的最终输出
                prompt_embeds = self.text_encoder(text_input_ids.to(device), attention_mask=attention_mask)
                prompt_embeds = prompt_embeds[0]
            else:
                prompt_embeds = self.text_encoder(
                    text_input_ids.to(device), attention_mask=attention_mask, output_hidden_states=True
                )
                # Access the `hidden_states` first, that contains a tuple of
                # all the hidden states from the encoder layers. Then index into
                # the tuple to access the hidden states from the desired layer.
                prompt_embeds = prompt_embeds[-1][-(clip_skip + 1)]
                # We also need to apply the final LayerNorm here to not mess with the
                # representations. The `last_hidden_states` that we typically use for
                # obtaining the final prompt representations passes through the LayerNorm
                # layer.
                prompt_embeds = self.text_encoder.text_model.final_layer_norm(prompt_embeds)

        # 确保 embeddings 与模型的 dtype 和 device 一致
        if self.text_encoder is not None:
            prompt_embeds_dtype = self.text_encoder.dtype
        elif self.unet is not None:
            prompt_embeds_dtype = self.unet.dtype
        else:
            prompt_embeds_dtype = prompt_embeds.dtype

        prompt_embeds = prompt_embeds.to(dtype=prompt_embeds_dtype, device=device)

        # batch 扩展，支持一个 prompt 生成多张图
        bs_embed, seq_len, _ = prompt_embeds.shape
        # duplicate text embeddings for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)

        # get unconditional embeddings for classifier free guidance
        # 获取无条件嵌入，用于实现 CFG
        if do_classifier_free_guidance and negative_prompt_embeds is None:
            uncond_tokens: List[str]
            # 没有提供 negative_prompt_embeds 和 negative_prompt
            if negative_prompt is None:
                uncond_tokens = [""] * batch_size
            elif prompt is not None and type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )
            else:
                # 正常情况：negative_prompt 是列表且长度匹配 batch_size
                uncond_tokens = negative_prompt

            # textual inversion: process multi-vector tokens if necessary
            if isinstance(self, TextualInversionLoaderMixin):
                uncond_tokens = self.maybe_convert_prompt(uncond_tokens, self.tokenizer)

            # 正向编码已经编码成了 prompt_embeds
            max_length = prompt_embeds.shape[1]
            uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )

            if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
                attention_mask = uncond_input.attention_mask.to(device)
            else:
                attention_mask = None

            negative_prompt_embeds = self.text_encoder(
                uncond_input.input_ids.to(device),
                attention_mask=attention_mask,
            )
            negative_prompt_embeds = negative_prompt_embeds[0]

        if do_classifier_free_guidance:
            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            seq_len = negative_prompt_embeds.shape[1]

            negative_prompt_embeds = negative_prompt_embeds.to(dtype=prompt_embeds_dtype, device=device)

            negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_images_per_prompt, 1)
            negative_prompt_embeds = negative_prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

        if self.text_encoder is not None:
            if isinstance(self, StableDiffusionLoraLoaderMixin) and USE_PEFT_BACKEND:
                # Retrieve the original scale by scaling back the LoRA layers
                unscale_lora_layers(self.text_encoder, lora_scale)

        return prompt_embeds, negative_prompt_embeds

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.encode_image
    def encode_image(self, image, device, num_images_per_prompt, output_hidden_states=None):
        # next() 返回 self.image_encoder 第一个参数张量，获取其数据类型
        dtype = next(self.image_encoder.parameters()).dtype

        # 如果不是 tensor, 使用 self.feature_extranctor 进行处理
        if not isinstance(image, torch.Tensor):
            image = self.feature_extractor(image, return_tensors="pt").pixel_values

        # 将预处理的像素值移动到指定设备和所需类型
        image = image.to(device=device, dtype=dtype)
        
        if output_hidden_states:
            # 正向编码，编码真实参考图像
            # hidden_states[-2] 倒数第二层（视觉特征丰富）
            image_enc_hidden_states = self.image_encoder(image, output_hidden_states=True).hidden_states[-2]
            image_enc_hidden_states = image_enc_hidden_states.repeat_interleave(num_images_per_prompt, dim=0)
            
            # 无条件编码：使用全零图像编码得到 negative 条件 
            uncond_image_enc_hidden_states = self.image_encoder(
                torch.zeros_like(image), output_hidden_states=True
            ).hidden_states[-2]
            uncond_image_enc_hidden_states = uncond_image_enc_hidden_states.repeat_interleave(
                num_images_per_prompt, dim=0
            )
            return image_enc_hidden_states, uncond_image_enc_hidden_states
        
        # 标准模式：返回全局池化输出（image_embeds）
        else:
            # 正向编码，直接取 image_encoder 的池化输出 image_embeds
            image_embeds = self.image_encoder(image).image_embeds
            image_embeds = image_embeds.repeat_interleave(num_images_per_prompt, dim=0)
            # 无条件编码
            uncond_image_embeds = torch.zeros_like(image_embeds)

            return image_embeds, uncond_image_embeds

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_ip_adapter_image_embeds
    def prepare_ip_adapter_image_embeds(
        self, ip_adapter_image, ip_adapter_image_embeds, device, num_images_per_prompt, do_classifier_free_guidance
    ):
        image_embeds = []

        if do_classifier_free_guidance:
            negative_image_embeds = []
        
        # 情况1：用户提供原始图像，需要现场编码
        if ip_adapter_image_embeds is None:
            if not isinstance(ip_adapter_image, list):
                ip_adapter_image = [ip_adapter_image]

            # 输入图像数量等于当前加载的 ip-adapter 数量
            if len(ip_adapter_image) != len(self.unet.encoder_hid_proj.image_projection_layers):
                raise ValueError(
                    f"`ip_adapter_image` must have same length as the number of IP Adapters. Got {len(ip_adapter_image)} images and {len(self.unet.encoder_hid_proj.image_projection_layers)} IP Adapters."
                )

            # 逐个 ip-adapter 处理对应的参考图像
            for single_ip_adapter_image, image_proj_layer in zip(
                ip_adapter_image, self.unet.encoder_hid_proj.image_projection_layers
            ):
                # 判断当前投影层是否需要返回完整的 hidden_states
                # ImageProjection 会直接输出投影后的嵌入
                output_hidden_state = not isinstance(image_proj_layer, ImageProjection)

                single_image_embeds, single_negative_image_embeds = self.encode_image(
                    single_ip_adapter_image, device, 1, output_hidden_state
                )
                # 添加一个维度[1, ...]
                image_embeds.append(single_image_embeds[None, :])
                if do_classifier_free_guidance:
                    negative_image_embeds.append(single_negative_image_embeds[None, :])
        
        # 情况2：用户直接提供预计算的 embeds
        else:
            # ip_adapter_image_embeds 通常是 list[tensor]
            for single_image_embeds in ip_adapter_image_embeds:
                if do_classifier_free_guidance:
                    # 这里假设预计算的 embeds 已经将 negative 和 positive 拼接到一起
                    # chunk(2) 方法分离
                    single_negative_image_embeds, single_image_embeds = single_image_embeds.chunk(2)
                    negative_image_embeds.append(single_negative_image_embeds)
                image_embeds.append(single_image_embeds)
        
        # 最终返回的列表
        ip_adapter_image_embeds = []
        for i, single_image_embeds in enumerate(image_embeds):
            # 根据 num_images_per_prompt 复制正向 embeds 
            single_image_embeds = torch.cat([single_image_embeds] * num_images_per_prompt, dim=0)
            if do_classifier_free_guidance:
                single_negative_image_embeds = torch.cat([negative_image_embeds[i]] * num_images_per_prompt, dim=0)
                single_image_embeds = torch.cat([single_negative_image_embeds, single_image_embeds], dim=0)

            single_image_embeds = single_image_embeds.to(device=device)
            ip_adapter_image_embeds.append(single_image_embeds)

        return ip_adapter_image_embeds

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.run_safety_checker
    def run_safety_checker(self, image, device, dtype):
        """
        在生成图像后运行安全检查器
        """

        # 如果禁用安全检查
        if self.safety_checker is None:
            has_nsfw_concept = None
        else:
            # 将图像转换为 pil 格式
            if torch.is_tensor(image):
                feature_extractor_input = self.image_processor.postprocess(image, output_type="pil")
            else:
                feature_extractor_input = self.image_processor.numpy_to_pil(image)
            
            # 提取 clip 图像特征
            safety_checker_input = self.feature_extractor(feature_extractor_input, return_tensors="pt").to(device)
            image, has_nsfw_concept = self.safety_checker(
                images=image, clip_input=safety_checker_input.pixel_values.to(dtype)
            )
        return image, has_nsfw_concept

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.decode_latents
    def decode_latents(self, latents):
        deprecation_message = "The decode_latents method is deprecated and will be removed in 1.0.0. Please use VaeImageProcessor.postprocess(...) instead"
        deprecate("decode_latents", "1.0.0", deprecation_message, standard_warn=False)

        latents = 1 / self.vae.config.scaling_factor * latents
        image = self.vae.decode(latents, return_dict=False)[0]
        image = (image / 2 + 0.5).clamp(0, 1)
        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()
        return image

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_extra_step_kwargs
    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://huggingface.co/papers/2010.02502
        # and should be between [0, 1]
        """
        准备额外参数，传递给 scheduler 的 step 方法
        """

        # 检查当前调度器的 step 方法时候接受 "eta" 的参数
        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        # 是否需要 generator
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs
    
    def check_inputs(
            self,
            prompt,
            image,
            callback_steps=None,
            negative_prompt=None,
            prompt_embeds=None,
            negative_prompt_embeds=None,
            ip_adapter_image=None,
            ip_adapter_image_embeds=None,
            controlnet_conditioning_scale=1.0,
            control_guidance_start=0.0,
            control_guidance_end=1.0,
            callback_on_step_end_tensor_inputs=None
    ):
        
        # 检查 callback_steps 是否为正整数
        if callback_steps is not None and (not isinstance(callback_steps, int) or callback_steps <= 0):
            raise ValueError(
                f"'callback_steps' must be a positive integer but is {callback_steps} of type"
                f"{type(callback_steps)}."
                )
        
        if callback_on_step_end_tensor_inputs is not None and not all(
            k in self._callback_tensor_inputs for k in callback_on_step_end_tensor_inputs
        ):
            raise ValueError(
                f"'callback_on_step_end_tensor_inputs' must be in {self._callback_tensor_inputs}, but found {[k for k in callback_on_step_end_tensor_inputs if k not in self._callback_tensor_inputs]}"
            )
        
        # 检查 prompt 和 prompt_embeds
        if prompt is not None and prompt_embeds is not None:
            raise ValueError(
                "Cannot forward both 'prompt' and 'prompt_embeds'. Please make sure to only forward one of the two."
            )
        elif prompt is None and prompt_embeds is None:
            raise ValueError(
                "Provide either 'prompt' or 'prompt_embeds'. Cannot leave both undefined."
            )
        elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            raise ValueError(
                f"'prompt must be of type 'str' or 'list' but is {type(prompt)}."
            )
        
        if negative_prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                "Cannot forward both 'negative_prompt' and 'negative_prompt_embeds'. Please make sure to only forward one of the two."
            )
        
        if prompt_embeds is not None and negative_prompt_embeds is not None:
            if prompt_embeds.shape != negative_prompt_embeds.shape:
                raise ValueError(
                    f"'prompt_embeds' and 'negative_prompt_embeds' must have the same shape when passed directly, but"
                    f"got: 'prompt_embeds': {prompt_embeds} != 'negative_prompt_embeds': {negative_prompt_embeds}."
                )
            
        # check image
        # 检查当前加载的 controlnet 模型是否被 torch compile 编译过
        is_compiled = hasattr(F, "scaled_dot_product_attention") and isinstance(
            self.controlnet, torch._dynamo.eval_frame.OptimizedModule
        )

        # 单个 controlnet 模型
        if (
            isinstance(self.controlnet, ControlNetModel)
            or is_compiled
            and isinstance(self.controlnet._orig_mod, ControlNetModel)
        ):
            self.check_image(image, prompt, prompt_embeds)
        else:
            assert False
        
        # 检查 controlnet_conditioning_scale
        if (
            isinstance(self.controlnet, ControlNetModel)
            or is_compiled
            and isinstance(self.controlnet._orig_mod, ControlNetModel)
        ):
            if not isinstance(controlnet_conditioning_scale, float):
                raise TypeError(
                    "For single controlnet: 'controlnet_conditioning_scale' must be type 'float'."
                )
        else:
            assert False

        if not isinstance(control_guidance_start, (tuple, list)):
            control_guidance_start = [control_guidance_start]
        
        if not isinstance(control_guidance_end, (tuple, list)):
            control_guidance_end = [control_guidance_end]

        if len(control_guidance_start) != len(control_guidance_end):
            raise ValueError(
                f"'control_guidance_start' has {len(control_guidance_start)} elements, but 'control_guidance_end' has {len(control_guidance_end)} elements. Make sure to provide the same number of elements to each list."
            )
        
        for start, end in zip(control_guidance_start, control_guidance_end):
            if start >= end:
                raise ValueError(
                    f"'control_guidance_start': {start} cannot be larger or equal to 'control_guidance_end': {end}."
                )
            if start < 0.0:
                raise ValueError(
                    f"'control_guidance_start': {start} cannot be smaller than 0."
                )
            if end > 1.0:
                raise ValueError(
                    f"'control_guidance_start': {end} cannot be larger than 1."
                )
            
        if ip_adapter_image is not None and ip_adapter_image_embeds is not None:
            raise ValueError(
                "Provide either 'ip_adapter_image' or 'ip_adapter_image_embeds'. Cannot leave both 'ip_adapter_image' and 'ip_adapter_image_embeds' defined."
            )
        
        if ip_adapter_image_embeds is not None:
            if not isinstance(ip_adapter_image_embeds, list):
                raise TypeError(
                    f"'ip_adapter_image_embeds' has to be of type 'list' but is {type(ip_adapter_image_embeds)}."
                )
            elif ip_adapter_image_embeds[0].ndim not in [3, 4]:
                raise ValueError(
                    f"'ip_adapter_image_embeds' has to be a list of 3D or 4D tensors but is{ip_adapter_image_embeds[0].ndim}D."
                )
        


    def check_image(self, image, prompt, prompt_embeds):
        """
        检查输入的 image 是否符合要求
        """

        # 判断是否是单张 PIL 图像
        image_is_pil = isinstance(image, PIL.Image.Image)
        # 判断是否是单张 torch 张量
        image_is_tensor = isinstance(image, torch.Tensor)
        # 判断是否是单张 numpy 数组
        image_is_np = isinstance(image, np.ndarray)

        # 判断是否是 PIL 图像列表
        image_is_pil_list = isinstance(image, list) and isinstance(image[0], PIL.Image.Image)
        # 判断是否是 torch 张量 列表
        image_is_tensor_list = isinstance(image, list) and isinstance(image[0], torch.Tensor)
        # 判断是否是 numpy 数组列表
        image_is_np_list = isinstance(image, list) and isinstance(image[0], np.ndarray)

        if (
            not image_is_pil
            and not image_is_tensor
            and not image_is_np
            and not image_is_pil_list
            and not image_is_tensor_list
            and not image_is_np_list
        ):
            raise TypeError(
                f"image must be passed and be one of pil image, torch tensor, numpy array, list of pil images, list of torch tensor or list of numpy array, but is {image.__class__}."
            )
        
        # 计算 image 的批次大小
        if image_is_pil or image_is_tensor or image_is_np:
            image_batch_size = 1
        else:
            image_batch_size = len(image)

        if prompt is not None and isinstance(prompt, str):
            prompt_batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            prompt_batch_size = len(prompt)
        elif prompt_embeds is not None:
            prompt_batch_size = prompt_embeds.shape[0]
        else:
            prompt_batch_size = 1

        if image_batch_size != 0 and image_batch_size != prompt_batch_size:
            raise ValueError(
                f"If image batch size is not 1, image batch size must be same as prompt batch size."
                f"image batch size: {image_batch_size}, prompt batch size: {prompt_batch_size}"
            ) 
        
    def prepare_image(
            self,
            image, 
            width,
            height,
            batch_size,                             # prompt 的批次大小
            num_image_per_prompt,
            device,
            dtype,
            do_classifier_free_guidance=False,      # 是否启用无分类器引导
            guess_mode=False                        # 猜测模式：某些 controlnet 使用，忽略提示词
    ):
        # 将图像转换为张量，resize 到（height，width），并进行归一化，转换到 float32 类型
        image = self.control_image_processor.preprocess(image, height=height, width=width).to(dtype=torch.float32)

        image_batch_size = image.shape[0]

        # 确定重复次数
        if image_batch_size == 1:
            repeat_by = batch_size
        else:
            repeat_by = num_image_per_prompt

        # 沿批次维度重复图像
        image = image.repeat_interleave(repeat_by, dim=0)

        if do_classifier_free_guidance and not guess_mode:
            image = torch.cat([image] * 2)

        return image
    
    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_latents
    def prepare_latents(self, batch_size, num_channels_latents, height, width, dtype, device, generator, latents=None):
        """
        准备扩散过程的初始潜在变量（latents）
        """

        # 计算潜在空间的实际形状
        # stable diffusion 的 vae 默认下采样因子为 8
        shape = (
            batch_size,
            num_channels_latents,
            int(height) // self.vae_scale_factor,
            int(width) // self.vae_scale_factor,
        )

        # 如果传入多个 generator（列表），则数量必须与 batch_size 保持一致
        # 多个 generator 保证生成多样性
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        # 噪声生成
        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device)

        # scale the initial noise by the standard deviation required by the scheduler
        # 缩放初始噪声的强度
        latents = latents * self.scheduler.init_noise_sigma
        return latents

    # Copied from diffusers.pipelines.latent_consistency_models.pipeline_latent_consistency_text2img.LatentConsistencyModelPipeline.get_guidance_scale_embedding
    def get_guidance_scale_embedding(
        self, w: torch.Tensor, embedding_dim: int = 512, dtype: torch.dtype = torch.float32
    ) -> torch.Tensor:
        """
        See https://github.com/google-research/vdm/blob/dc27b98a554f65cdc654b800da5aa1846545d41b/model_vdm.py#L298

        Args:
            w (`torch.Tensor`):
                Generate embedding vectors with a specified guidance scale to subsequently enrich timestep embeddings.
            embedding_dim (`int`, *optional*, defaults to 512):
                Dimension of the embeddings to generate.
            dtype (`torch.dtype`, *optional*, defaults to `torch.float32`):
                Data type of the generated embeddings.

        Returns:
            `torch.Tensor`: Embedding vectors with shape `(len(w), embedding_dim)`.
        """
        assert len(w.shape) == 1
        w = w * 1000.0

        half_dim = embedding_dim // 2
        emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=dtype) * -emb)
        emb = w.to(dtype)[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if embedding_dim % 2 == 1:  # zero pad
            emb = torch.nn.functional.pad(emb, (0, 1))
        assert emb.shape == (w.shape[0], embedding_dim)
        return emb
    
    @property
    def guidance_scale(self):
        return self._guidance_scale
    
    @property
    def clip_skip(self):
        return self._clip_skip
    
    @property
    def do_classifier_free_guidance(self):
        return self._guidance_scale > 1 and self.unet.config.time_cond_proj_dim is None
    
    @property
    def cross_attention_kwargs(self):
        return self._cross_attention_kwargs
        
    @property
    def num_timesteps(self):
        return self._num_timesteps
    
    @property
    def interrupt(self):
        return self._interrupt
    
    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        image: PipelineImageInput = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        timesteps: List[int] = None,
        sigmas: List[float] = None,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        ip_adapter_image: Optional[PipelineImageInput] = None,
        ip_adapter_image_embeds: Optional[List[torch.Tensor]] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        controlnet_conditioning_scale: Union[float, List[float]] = 1.0,
        guess_mode: bool = False,
        control_guidance_start: Union[float, List[float]] = 0.0,
        control_guidance_end: Union[float, List[float]] = 1.0,
        clip_skip: Optional[int] = None,
        callback_on_step_end: Union[Callable[[int, int, Dict], None], PipelineCallback] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        **kwargs
    ):
        
        # 兼容新版本 callback 参数
        callback = kwargs.pop("callback", None)
        callback_steps = kwargs.pop("callback_steps", None)

        if callback is not None:
            deprecate(
                "callback",
                "1.0.0",
                "Passing 'callback' as an input argument to '__call__' is deprecated, considering using 'callback_on_step_end'"
            )
        if callback_steps is not None:
            deprecate(
                "callback_steps",
                "1.0.0",
                "Passing 'callback_steps' as an input argument to '__call__' is deprecated, considering using 'callback_on_step_end'"
            )

        # 处理新的回调机制
        if isinstance(callback_on_step_end, PipelineCallback):
            callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs

        controlnet = self.controlnet._orig_mode if is_compiled_module(self.controlnet) else self.controlnet

        # 对齐 control_guidance_start 和 control_guidance_end 的格式
        if not isinstance(control_guidance_start, list) and isinstance(control_guidance_end, list):
            control_guidance_start = len(control_guidance_end) * [control_guidance_start]
        elif not isinstance(control_guidance_end, list) and isinstance(control_guidance_start, list):
            control_guidance_end = len(control_guidance_start) * [control_guidance_end]

        # 检查输入参数的有效性
        self.check_inputs(
            prompt,
            image,
            callback_steps,
            negative_prompt,
            prompt_embeds,
            negative_prompt_embeds,
            ip_adapter_image,
            ip_adapter_image_embeds,
            controlnet_conditioning_scale,
            control_guidance_start,
            control_guidance_end,
            callback_on_step_end_tensor_inputs
        )

        self._guidance_scale = guidance_scale
        self._clip_skip = clip_skip
        self._cross_attention_kwargs = cross_attention_kwargs
        self._interrput = False

        # 定义调用参数
        # 获取 batch size 大小
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        # 是否使用 global_pool_conditions
        global_pool_conditions = (
            controlnet.config.global_pool_conditions 
            if isinstance(controlnet, ControlNetModel)
            else controlnet.nets[0].config.global_pool_conditions
        )

        guess_mode = guess_mode or global_pool_conditions

        # 编码输入的文本提示
        text_encoder_lora_scale = (
            self.cross_attention_kwargs.get("scale", None) if self.cross_attention_kwargs is not None else None
        )

        # 调用 pipeline 的 encode_prompt 方法，将文本转为嵌入
        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            self.do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            lora_scale=text_encoder_lora_scale,
            clip_skip=self.clip_skip
        )

        if self.do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

        # 处理 IP-Adapter 相关的图像嵌入
        if ip_adapter_image is not None or ip_adapter_image_embeds is not None:
            image_embeds = self.prepare_ip_adapter_image_embeds(
                ip_adapter_image,
                ip_adapter_image_embeds,
                device,
                batch_size * num_images_per_prompt,
                self.do_classifier_free_guidance
            )
        
        # prepare image
        if isinstance(controlnet, ControlNetModel):
            # 调用 pipeline 的 prepare_image 方法，对输入的控制图像进行同意预处理
            image = self.prepare_image(
                image=image,
                width=width,
                height=height,
                batch_size=batch_size * num_images_per_prompt,
                device=device,
                dtype=controlnet.dtype,
                do_classifier_free_guidance=self.do_classifier_free_guidance,
                guess_mode=guess_mode
            )
            height, width = image.shape[-2:]

        else:
            assert False

        # prepare timesteps
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler,
            num_inference_steps,
            device,
            timesteps,
            sigmas
        )

        self.num_timesteps = len(timesteps)

        # prepare latent variables
        num_channels_latents = self.unet.config.in_channels

        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.detype,
            device,
            generator,
            latents
        )

        # 获取 guidance scale embedding
        timestep_cond = None

        if self.unet.time_cond_proj_dim is not None:
            guidance_scale_tensor = torch.tensor(self.guidance_scale - 1).repeat(batch_size * num_images_per_prompt)

            timestep_cond = self.get_guidance_scale_embedding(
                guidance_scale_tensor,
                embedding_dim=self.unet.time_cond_proj_dim
            ).to(device=device, dtype=latents.dtype)
        
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 准备采样器需要的额外参数
        added_cond_kwargs = {"image_embeds": image_embeds} if ip_adapter_image is not None or ip_adapter_image_embeds is not None else None

        controlnet_keep = []

        for i in range(len(timesteps)):
            keeps = [
                1.0 - float(i / len(timesteps) < s or (i + 1) / len(timesteps) > e for s, e in zip(control_guidance_start, control_guidance_end))
            ]
        
            controlnet_keep.append(
                keeps[0] if isinstance(controlnet, ControlNetModel) else keeps
            )

        # 核心去噪循环
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        # 检查 unet 和 controlnet 是否被 torch.compiled 编译
        is_unet_compiled = is_compiled_module(self.unet)
        is_controlnet_compiled = is_compiled_module(self.controlnet)
        # 检查 pytorch 版本
        is_torch_higher_equal_2_1 = is_torch_version(">=", "2.1")
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue
            
                if (is_unet_compiled and is_controlnet_compiled) and is_torch_higher_equal_2_1:
                    torch._inductor.cudagraph_mark_step_begin()

                latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                if guess_mode and self.do_classifier_free_guidance:
                    controlnet_model_input = latents
                    controlnet_model_input = self.scheduler.scale_model_input(controlnet_model_input, t)

                    controlnet_prompt_embeds = prompt_embeds.chunk(2)[1]
                else:
                    controlnet_model_input = latent_model_input
                    controlnet_prompt_embeds = prompt_embeds

                # 处理多 ControlNet 或动态控制强度（controlnet_keep[i] 允许每一步不同强度）
                if isinstance(controlnet_keep[i], list):
                # 多 ControlNet 情况：分别乘以对应比例和当前步的 keep 系数
                    cond_scale = [c * s for c, s in zip(controlnet_conditioning_scale, controlnet_keep[i])]
                else:
                # 单 ControlNet
                    controlnet_cond_scale = controlnet_conditioning_scale
                    if isinstance(controlnet_cond_scale, list):
                        controlnet_cond_scale = controlnet_cond_scale[0]  # 取列表第一个值（兼容性处理）
                    cond_scale = controlnet_cond_scale * controlnet_keep[i]  # 当前步的有效控制强度

                # 调用 controlnet 前向，得到各层残差
                down_block_res_sample, mid_block_res_sample = self.controlnet(
                    controlnet_model_input,
                    t,
                    encoder_hidden_states=controlnet_prompt_embeds,
                    controlnet_cond=image,
                    conditioning_scale=cond_scale,
                    guess_mode=guess_mode,
                    return_dict=False
                )

                if guess_mode and self.do_classifier_free_guidance:
                    down_block_res_sample = [torch.cat([torch.zeros_like(d), d] for d in down_block_res_sample)]
                    mid_block_res_sample = torch.cat([torch.zeros_like(mid_block_res_sample), mid_block_res_sample])

                # 噪声预测
                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    timestep_cond=timestep_cond,
                    cross_attention_kwargs=self.cross_attention_kwargs,
                    down_block_res_sample=down_block_res_sample,
                    mid_block_res_sample=mid_block_res_sample,
                    added_cond_kwargs=added_cond_kwargs,
                    return_dict=False
                )[0]

                # perform guidance
                if self.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs)

                if callback_on_step_end is not None:
                    callback_kwargs = []
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    # 用回调返回后的值更新变量
                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                    negative_prompt = callback_outputs.pop("negative_prompt_embeds", negative_prompt)
                    image = callback_outputs.pop("image", image)

                if i == len(timesteps - 1) or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        step_idx = i // getattr(self.scheduler, "order", 1)
                        callback(step_idx, t, latents)
                
                if XLA_AVAILABLE:
                    xm.mark_step()
        
        if hasattr(self, "final_offload_hook") and self.final_offload_hook is not None:
            self.unet.to("cpu")
            self.controlnet.to("cpu")
            empty_device_cache()

        # 如果用户请求的 output_type 不是 “latent”
        if not output_type == "latents":
            self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False, generator=generator)[
                0
            ]
            image, has_nsfw_concept = self.run_safety_checker(image, device, prompt_embeds.dtype)
        else:
            image = latents
            has_nsfw_concept = None
        
        if has_nsfw_concept is None:
            do_denormalize = [True] * image.shape[0]
        else:
            do_denormalize = [not has_nsfw for has_nsfw in has_nsfw_concept]

        image = self.image_processor.postprocess(image, output_type=output_type, do_denormalize=do_denormalize)

        # 卸载所有可能的模型钩子
        self.maybe_free_model_hooks()

        if not return_dict:
            return (image, has_nsfw_concept)

        return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept)
    