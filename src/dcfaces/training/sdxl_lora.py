"""SDXL component loading + LoRA helpers, shared by 05 (demographic LoRA) and
07 (per-identity DreamBooth-LoRA).

SDXL specifics handled here so the training scripts stay readable:
  * two text encoders (CLIP-L + CLIP-bigG) -> concatenated penultimate hidden
    states (2048-d) + pooled embeds from the second encoder;
  * micro-conditioning "added time ids" (original/crop/target sizes);
  * mandatory VAE swap to the fp16-fix VAE (default SDXL VAE NaNs in fp16).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from transformers import AutoTokenizer, CLIPTextModel, CLIPTextModelWithProjection


@dataclass
class SDXLComponents:
    tokenizers: list
    text_encoders: list
    vae: AutoencoderKL
    unet: UNet2DConditionModel
    noise_scheduler: DDPMScheduler


def load_sdxl_components(
    base: str,
    vae_id: str,
    device: str = "cuda",
    weight_dtype: torch.dtype = torch.bfloat16,
) -> SDXLComponents:
    """Load frozen SDXL pieces. VAE stays fp32 (robust encode); rest in weight_dtype."""
    tokenizer = AutoTokenizer.from_pretrained(base, subfolder="tokenizer", use_fast=False)
    tokenizer_2 = AutoTokenizer.from_pretrained(base, subfolder="tokenizer_2", use_fast=False)
    text_encoder = CLIPTextModel.from_pretrained(base, subfolder="text_encoder")
    text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(base, subfolder="text_encoder_2")
    vae = AutoencoderKL.from_pretrained(vae_id)  # fp16-fix VAE
    unet = UNet2DConditionModel.from_pretrained(base, subfolder="unet")
    noise_scheduler = DDPMScheduler.from_pretrained(base, subfolder="scheduler")

    for m in (text_encoder, text_encoder_2, vae, unet):
        m.requires_grad_(False)
    text_encoder.to(device, dtype=weight_dtype).eval()
    text_encoder_2.to(device, dtype=weight_dtype).eval()
    vae.to(device, dtype=torch.float32).eval()  # keep VAE fp32 for stable encode
    unet.to(device, dtype=weight_dtype)

    return SDXLComponents(
        tokenizers=[tokenizer, tokenizer_2],
        text_encoders=[text_encoder, text_encoder_2],
        vae=vae,
        unet=unet,
        noise_scheduler=noise_scheduler,
    )


@torch.no_grad()
def encode_prompt(captions, tokenizers, text_encoders, device):
    """SDXL dual-encoder prompt encoding -> (prompt_embeds[2048], pooled_embeds)."""
    embeds_list = []
    pooled = None
    for tok, te in zip(tokenizers, text_encoders):
        ids = tok(
            captions,
            padding="max_length",
            max_length=tok.model_max_length,
            truncation=True,
            return_tensors="pt",
        ).input_ids.to(device)
        out = te(ids, output_hidden_states=True, return_dict=True)
        pooled = out[0]  # pooled comes from the 2nd encoder (CLIPTextModelWithProjection)
        embeds_list.append(out.hidden_states[-2])  # penultimate hidden state
    return torch.cat(embeds_list, dim=-1), pooled


def build_time_ids(original_sizes, crop_top_lefts, target_sizes, device, dtype):
    """SDXL added time-ids: (orig_h, orig_w, crop_top, crop_left, tgt_h, tgt_w) per sample."""
    rows = [
        list(o) + list(c) + list(t)
        for o, c, t in zip(original_sizes, crop_top_lefts, target_sizes)
    ]
    return torch.tensor(rows, device=device, dtype=dtype)


def save_lora(unet, save_dir) -> None:
    """Save UNet LoRA in diffusers format (load with pipe.load_lora_weights)."""
    from diffusers import StableDiffusionXLPipeline
    from diffusers.utils import convert_state_dict_to_diffusers
    from peft.utils import get_peft_model_state_dict

    lora_sd = convert_state_dict_to_diffusers(get_peft_model_state_dict(unet))
    StableDiffusionXLPipeline.save_lora_weights(str(save_dir), unet_lora_layers=lora_sd)
