# from dataclasses import dataclass
# from PIL import Image
# import os

# import torch
# import torch.nn.functional as F
# from torchvision import transforms
# import matplotlib.pyplot as plt
# from datasets import load_dataset
# from diffusers import UNet2DModel, DDPMScheduler
# from diffusers.optimization import get_cosine_schedule_with_warmup
# from diffusers.utils import make_image_grid
# from accelerate import Accelerator
# from tqdm.auto import tqdm

# from diffusers import DDPMPipeline

# @dataclass
# class TrainingConfig:
#     image_size = 128
#     train_batch_size = 16
#     eval_batch_size = 16
#     num_epochs = 60
#     gradient_accumulation_steps = 1
#     learning_rate = 1e-4
#     lr_warmup_steps = 500
#     save_image_epochs = 5
#     save_model_epochs = 10
#     mixed_precision = "fp16"
#     output_dir = "ddpm-butterflies-128"

#     seed=0

# def evaluate(config, epoch, pipeline):
#     images = pipeline(
#         batch_size = config.eval_batch_size,
#         generator = torch.Generator(device="cpu").manual_seed(config.seed)
#     ).images

#     image_grid = make_image_grid(images, rows=4, cols=4)

#     test_dir = os.path.join(config.output_dir, "samples")
#     os.makedirs(test_dir, exist_ok=True)
#     image_grid.save(f"{test_dir}/{epoch:04d}.png")

# def train_loop(config, model, noise_scheduler, optimizer, train_dataloader, lr_scheduler):
#     accelerator = Accelerator(
#         mixed_precision=config.mixed_precision,
#         gradient_accumulation_steps=config.gradient_accumulation_steps,
#         log_with="tensorboard",
#         project_dir=os.path.join(config.output_dir, "logs")
#     )

#     if accelerator.is_main_process:
#         if config.output_dir is not None:
#             os.makedirs(config.output_dir, exist_ok=True)
#         accelerator.init_trackers("train_example")   

#     model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
#         model, optimizer, train_dataloader, lr_scheduler
#     ) 

#     global_step = 0

#     for epoch in range(config.num_epochs):
#         progress_bar = tqdm(total=len(train_dataloader), disable= not accelerator.is_local_main_process)
#         progress_bar.set_description(f"Epoch {epoch}")

#         for step, batch in enumerate(train_dataloader):
#             clean_images = batch["images"]
#             noise = torch.randn(clean_images.shape, device=clean_images.device)
#             bs = clean_images.shape[0]

#             timesteps = torch.randint(
#                 0, noise_scheduler.config.num_train_timesteps, (bs, ), device=clean_images.device,
#                 dtype=torch.int64
#             )
        
#             noise_images = noise_scheduler.add_noise(clean_images, noise, timesteps)

#             with accelerator.accumulate(model):
#                 noise_pred = model(noise_images, timesteps, return_dict=False)[0]\
            
#                 loss = F.mse_loss(noise_pred, noise)
#                 accelerator.backward(loss)

#                 if accelerator.sync_gradients:
#                     accelerator.clip_grad_norm_(model.parameters(), 1.0)
            
#                 optimizer.step()
#                 lr_scheduler.step()
#                 optimizer.zero_grad()

#             progress_bar.update(1)
#             logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "step": global_step}
#             progress_bar.set_postfix(logs)
#             accelerator.log(logs, step=global_step)
#             global_step += 1

#         if accelerator.is_main_process:
#             pipeline = DDPMPipeline(unet=accelerator.unwrap_model(model), scheduler=noise_scheduler)    

#             if (epoch + 1) % config.save_image_epochs == 0 or epoch == config.num_epochs - 1:
#                 evaluate(config, epoch, pipeline)

#             if (epoch + 1) % config.save_model_epochs == 0 or epoch == config.num_epochs - 1:
#                 pipeline.save_pretrained(config.output_dir)

# config = TrainingConfig()

# config.dataset_name = "huggan/smithsonian_butterflies_subset"
# dataset = load_dataset(config.dataset_name, split="train")

# def transform(examples):
#     preprocess = transforms.Compose([
#     transforms.Resize([config.image_size, config.image_size]),
#     transforms.RandomHorizontalFlip(),
#     transforms.ToTensor(),
#     transforms.Normalize([0.5], [0.5])
#     ])
#     images = [preprocess(image.convert("RGB")) for image in examples["image"]]
#     return {"images": images}

# dataset.set_transform(transform)

# # Dataloader
# train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=config.train_batch_size, shuffle=True)

# # UNet2DModel
# model = UNet2DModel(
#     sample_size=config.image_size,
#     in_channels=3,
#     out_channels=3,
#     layers_per_block=2,
#     block_out_channels=(128, 128, 256, 256, 512, 512),
#     down_block_types=(
#         "DownBlock2D",
#         "DownBlock2D",
#         "DownBlock2D",
#         "DownBlock2D",
#         "AttnDownBlock2D",
#         "DownBlock2D",
#     ),
#     up_block_types=(
#         "UpBlock2D",
#         "AttnUpBlock2D",
#         "UpBlock2D",
#         "UpBlock2D",
#         "UpBlock2D",
#         "UpBlock2D",
#     )
# )

# # 创建调度器
# noise_scheduler = DDPMScheduler(num_train_timesteps=1000)

# optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

# lr_scheduler = get_cosine_schedule_with_warmup(
#     optimizer=optimizer,
#     num_warmup_steps=config.lr_warmup_steps,
#     num_training_steps=(len(train_dataloader) * config.num_epochs)
# )

# train_loop(config, model, noise_scheduler, optimizer, train_dataloader, lr_scheduler)
import inspect
import torch
import sys

print(inspect.signature(isinstance).parameters)