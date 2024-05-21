#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@Created :   2024/05/15 17:29:21
@Desc    :   Cleaned up batch inference template
@Ref     :   
'''
import time
import random
from pathlib import Path

import numpy as np
import torch
from diffusers import AutoencoderKL, DDIMScheduler
from diffusers.utils.import_utils import is_xformers_available
from einops import rearrange
from omegaconf import OmegaConf
from tqdm import tqdm
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

from vid2vid_zero.models.unet_2d_condition import UNet2DConditionModel
from vid2vid_zero.data.dataset import VideoDataset
from vid2vid_zero.pipelines.pipeline_vid2vid_zero import Vid2VidZeroPipeline
from vid2vid_zero.p2p.null_text_w_ptp import NullInversion
from vid2vid_zero.util import save_videos_grid


negative_prompt="ugly, blurry, low res, unaesthetic"

data_root = '/data/trc/videdit-benchmark/DynEdit'
method_name = 'vid2vid-zero'

config = OmegaConf.create(dict(
    data_root=data_root,
    config_file=f'{data_root}/config.yaml',
    output_dir=f'{data_root}/outputs/{method_name}',
    seed=33,
    # TODO define arguments
    model_path='/data/trc/tmp-swh/models/stable-diffusion-v1-5',
    guidance_scale=7.5,
    null_normal_infer=True,
    null_inner_steps=1,
    null_base_lr=1e-2,
    num_inference_steps=50,
))


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    # load model
    print('Loading models ...')
    device = torch.device('cuda')
    # TODO define model
    tokenizer = CLIPTokenizer.from_pretrained(config.model_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(config.model_path, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(config.model_path, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(
        config.model_path, subfolder="unet", use_sc_attn=True, use_st_attn=True, st_attn_idx=0)
    # Freeze vae, text_encoder, and unet
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)
    vae.eval()
    text_encoder.eval()
    unet.eval()
    if is_xformers_available():
        unet.enable_xformers_memory_efficient_attention()
    unet.enable_gradient_checkpointing()
    vae.to(device)
    text_encoder.to(device)
    unet.to(device)
    pipeline = Vid2VidZeroPipeline(
        vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, unet=unet,
        scheduler=DDIMScheduler.from_pretrained(config.model_path, subfolder="scheduler"),
        safety_checker=None, feature_extractor=None,
    )

    data_config = OmegaConf.load(config.config_file)
    set_seed(config.seed)
    generator = torch.Generator(device="cuda")
    generator.manual_seed(config.seed)
    
    preprocess_elapsed_ls = []
    inference_elapsed_ls = []
    for row in tqdm(data_config['data']):
        output_dir = Path(f"{config.output_dir}/{row.video_id}")
        if output_dir.exists():
            print(f"Skip {row.video_id} ...")
            continue
        else:
            output_dir.mkdir(parents=True, exist_ok=True)

        # load video
        print(f"Processing {row.video_id} ...")
        video_path = f'{config.data_root}/videos/{row.video_id}.mp4'
        # TODO load video
        input_dataset = VideoDataset(video_path, row.prompt, n_sample_frames=24)
        input_dataset.prompt_ids = tokenizer(
            input_dataset.prompt, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
        ).input_ids[0]
        input_dataloader = torch.utils.data.DataLoader(input_dataset, batch_size=1)
        batch = next(iter(input_dataloader))
        pixel_values = batch["pixel_values"].to(device)
        video_length = pixel_values.shape[1]
        pixel_values = rearrange(pixel_values, "b f c h w -> (b f) c h w")

        # # Optional
        # inverse_path = Path(f"{config.output_dir}/{row.video_id}/.cache")
        # inverse_path.mkdir(parents=True, exist_ok=True)
        
        # preprocess
        start = time.perf_counter()
        # TODO preprocess video
        latents = vae.encode(pixel_values).latent_dist.sample()
        # take video as input
        latents = rearrange(latents, "(b f) c h w -> b c f h w", f=video_length)
        latents = latents * 0.18215
        ddim_inv_latent = None
        null_inversion = NullInversion(
            model=pipeline, guidance_scale=config.guidance_scale, null_inv_with_prompt=False,
            null_normal_infer=config.null_normal_infer,
        )
        ddim_inv_latent, uncond_embeddings = null_inversion.invert(
            latents, input_dataset.prompt, verbose=True, 
            null_inner_steps=config.null_inner_steps,
            null_base_lr=config.null_base_lr,
        )
        ddim_inv_latent = ddim_inv_latent.to(device)
        uncond_embeddings = [embed.to(device) for embed in uncond_embeddings]
        # ddim_inv_latent = ddim_inv_latent.repeat(2, 1, 1, 1, 1)
        preprocess_elapsed = time.perf_counter() - start
        preprocess_elapsed_ls.append(preprocess_elapsed)
        del null_inversion
        torch.cuda.empty_cache()

        # edit
        print(f'Editting {row.video_id} ...')
        start = time.perf_counter()
        for i, edit in tqdm(enumerate(row.edit)):
            # TODO edit
            # prompts=edit['prompt'],
            # negative_prompts=edit['src_words']+negative_prompt,
            # inversion_prompt=row['prompt'],
            # edit['tgt_words']

            # prompts = [input_dataset.prompt, edit['prompt']]  # a list of two prompts
            # cross_replace_steps, self_replace_steps = prepare_control(
            #     unet=unet, prompts=prompts, validation_data={'num_inference_steps': config.num_inference_steps})
            sample = pipeline(
                edit['prompt'], generator=generator, 
                latents=ddim_inv_latent, 
                uncond_embeddings=uncond_embeddings,
                video_length=24,
                height=512, width=512,
                num_inference_steps=config.num_inference_steps,
                guidance_scale=config.guidance_scale,
                null_uncond_ratio=-0.5,
            ).images

            # assert sample.shape[0] == 2
            # _, sample_gen = sample.chunk(2)
            sample_gen = sample[0].unsqueeze(0)
            save_videos_grid(sample_gen, f"{output_dir}/{i}.gif", fps=12)

        inference_elapsed = time.perf_counter() - start
        inference_elapsed_ls.append(inference_elapsed)

    with open(f'{config.output_dir}/time.log', 'a') as f:
        f.write(f'Preprocess: {sum(preprocess_elapsed_ls)/len(preprocess_elapsed_ls):.2f} sec/video\n')
        n_prompts = len(row.edit)
        f.write(f'Edit:       {sum(inference_elapsed_ls)/len(inference_elapsed_ls)/n_prompts:.2f} sec/edit\n')
        f.write('Preprocess:\n')
        f.writelines([f'{e:.1f} ' for e in preprocess_elapsed_ls])
        f.write('\nEdit:\n')
        f.writelines([f'{e:.1f} ' for e in inference_elapsed_ls])
        f.write('\n')
    print('Everything done!')


if __name__ == '__main__':
    main()