"""07: Train a per-identity DreamBooth-LoRA for each benchmark identity.

A comparison baseline: a separate rank-16 LoRA per identity on plain SDXL (no
demographic LoRA -- that stacking is an inference choice in 08, not training).
Token-based binding through an instance prompt; shared class images give prior
preservation against language drift. Each identity reloads a fresh UNet so the
adapters never share state.

Inputs:  data/identity_gallery/id_*/ref.jpg, configs/dreambooth_lora.yaml
Outputs: models/dreambooth_loras/id_*/pytorch_lora_weights.safetensors
         models/dreambooth_loras/_class_images/  (shared prior-preservation set)

Smoke-test, e.g.:
    python scripts/07_train_dreambooth_lora.py --limit-identities 1 --max-train-steps 10 --num-class-images 4
"""

from __future__ import annotations

import argparse
import gc
import json
import random

import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from dcfaces.paths import CONFIG_DIR, DREAMBOOTH_LORAS, IDENTITY_GALLERY, ensure_dirs

RESOLUTION = 1024
# Base SDXL defaults to white faces for an unspecified "a man/woman", so cycle
# races into the class prompt to keep the prior-preservation set demographically
# balanced (the prior-loss prompt during training stays the generic "a {gender}").
CLASS_RACES = ["white", "Black", "East Asian", "Southeast Asian", "South Asian", "Middle Eastern", "Latino"]


def to_tensor(path) -> torch.Tensor:
    tf = transforms.Compose(
        [
            transforms.Resize(RESOLUTION, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(RESOLUTION),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )
    return tf(Image.open(path).convert("RGB"))


def get_class_images(cfg, n: int, device: str) -> dict:
    """Generate (once) and cache per-gender prior-preservation class images.

    Returns {"man": [paths], "woman": [paths]}. Generated from base SDXL with a
    negative prompt so the prior set is clean (the bare prompt produced deformed,
    monochrome people).
    """
    tcfg = cfg["training"]
    steps = tcfg.get("class_inference_steps", 30)
    negative = tcfg.get("class_negative_prompt", "")
    root = DREAMBOOTH_LORAS / "_class_images"
    pipe = None
    sets = {}
    for gender in ("man", "woman"):
        cache = root / gender
        cache.mkdir(parents=True, exist_ok=True)
        have = sorted(cache.glob("*.png"))
        if len(have) < n:
            if pipe is None:
                from diffusers import AutoencoderKL, StableDiffusionXLPipeline

                pipe = StableDiffusionXLPipeline.from_pretrained(
                    cfg["model"]["base"],
                    vae=AutoencoderKL.from_pretrained(cfg["model"]["vae"], torch_dtype=torch.float16),
                    torch_dtype=torch.float16, use_safetensors=True,
                ).to(device)
                pipe.set_progress_bar_config(disable=True)
            print(f"generating {n - len(have)} '{gender}' class images (race-balanced)")
            for idx in range(len(have), n):
                race = CLASS_RACES[idx % len(CLASS_RACES)]
                prompt = tcfg["class_prompt"].format(gender=f"{race} {gender}")
                g = torch.Generator(device=device).manual_seed(1000 + idx)
                img = pipe(
                    prompt=prompt, negative_prompt=negative, num_inference_steps=steps,
                    guidance_scale=6.0, height=RESOLUTION, width=RESOLUTION, generator=g,
                ).images[0]
                img.save(cache / f"class_{idx:04d}.png")
        sets[gender] = sorted(cache.glob("*.png"))[:n]
    if pipe is not None:
        del pipe
        gc.collect()
        torch.cuda.empty_cache()
    return sets


def train_identity(id_dir, comp, class_images, cfg, device, weight_dtype, max_steps):
    import bitsandbytes as bnb
    from diffusers import UNet2DConditionModel
    from diffusers.optimization import get_scheduler
    from diffusers.training_utils import cast_training_params
    from peft import LoraConfig

    from dcfaces.training.sdxl_lora import build_time_ids, encode_prompt, save_lora

    tcfg = cfg["training"]
    unet = UNet2DConditionModel.from_pretrained(cfg["model"]["base"], subfolder="unet")
    unet.requires_grad_(False)
    unet.to(device, dtype=weight_dtype)
    unet.add_adapter(
        LoraConfig(
            r=cfg["lora"]["rank"], lora_alpha=cfg["lora"]["alpha"],
            init_lora_weights="gaussian", target_modules=cfg["lora"]["target_modules"],
        )
    )
    cast_training_params(unet, dtype=torch.float32)
    if tcfg.get("enable_xformers"):
        try:
            unet.enable_xformers_memory_efficient_attention()
        except Exception as e:  # noqa: BLE001
            print(f"xformers unavailable ({e})")
    if tcfg.get("gradient_checkpointing"):
        unet.enable_gradient_checkpointing()

    lora_params = [p for p in unet.parameters() if p.requires_grad]
    optimizer = bnb.optim.AdamW8bit(lora_params, lr=tcfg["learning_rate"])
    lr_scheduler = get_scheduler(
        tcfg["lr_scheduler"], optimizer,
        num_warmup_steps=tcfg["lr_warmup_steps"], num_training_steps=max_steps,
    )

    gender = json.loads((id_dir / "metadata.json").read_text())["fairface"]["gender"]
    instance_prompt = tcfg["instance_prompt"].format(gender=gender)
    class_prompt = tcfg["class_prompt"].format(gender=gender)
    class_set = class_images.get(gender, []) if class_images else []
    prior = bool(class_set)
    flip = tcfg.get("instance_flip", False)

    vae, scheduler = comp.vae, comp.noise_scheduler
    instance = to_tensor(id_dir / "ref.jpg")
    prior_weight = tcfg.get("prior_loss_weight", 1.0)
    rng = random.Random(tcfg["seed"])
    size = (RESOLUTION, RESOLUTION)

    unet.train()
    pbar = tqdm(range(max_steps), desc=f"{id_dir.name} ({gender})", leave=False)
    for step in pbar:
        inst = torch.flip(instance, dims=[2]) if (flip and rng.random() < 0.5) else instance
        pixels, prompts = [inst], [instance_prompt]
        if prior:
            pixels.append(to_tensor(class_set[rng.randrange(len(class_set))]))
            prompts.append(class_prompt)
        pixel_values = torch.stack(pixels).to(device, dtype=torch.float32)
        with torch.no_grad():
            latents = (vae.encode(pixel_values).latent_dist.sample() * vae.config.scaling_factor).to(weight_dtype)
            prompt_embeds, pooled = encode_prompt(prompts, comp.tokenizers, comp.text_encoders, device)
        bs = latents.shape[0]
        noise = torch.randn_like(latents)
        timesteps = torch.randint(0, scheduler.config.num_train_timesteps, (bs,), device=device).long()
        noisy = scheduler.add_noise(latents, noise, timesteps)
        added = {
            "text_embeds": pooled.to(weight_dtype),
            "time_ids": build_time_ids([size] * bs, [(0, 0)] * bs, [size] * bs, device, weight_dtype),
        }
        with torch.autocast("cuda", dtype=weight_dtype):
            model_pred = unet(noisy, timesteps, prompt_embeds.to(weight_dtype), added_cond_kwargs=added).sample
        target = noise if scheduler.config.prediction_type == "epsilon" else scheduler.get_velocity(latents, noise, timesteps)
        if prior:
            ip, cp = model_pred.chunk(2)
            it, ct = target.chunk(2)
            loss = F.mse_loss(ip.float(), it.float()) + prior_weight * F.mse_loss(cp.float(), ct.float())
        else:
            loss = F.mse_loss(model_pred.float(), target.float())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(lora_params, 1.0)
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        if step % 50 == 0:
            pbar.set_postfix(loss=f"{loss.item():.4f}")

    out_dir = DREAMBOOTH_LORAS / id_dir.name
    out_dir.mkdir(parents=True, exist_ok=True)
    save_lora(unet, out_dir)
    del unet, optimizer
    gc.collect()
    torch.cuda.empty_cache()
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--max-train-steps", type=int, default=None, help="Override config (smoke).")
    parser.add_argument("--limit-identities", type=int, default=0, help="Cap #identities (smoke).")
    parser.add_argument("--num-class-images", type=int, default=None, help="Override config (smoke).")
    parser.add_argument("--no-prior", action="store_true", help="Disable prior preservation (smoke).")
    args = parser.parse_args()

    ensure_dirs()
    cfg = yaml.safe_load((CONFIG_DIR / "dreambooth_lora.yaml").read_text())
    identities = sorted(d for d in IDENTITY_GALLERY.glob("id_*") if (d / "ref.jpg").exists())
    if not identities:
        raise SystemExit(f"No identities in {IDENTITY_GALLERY} — run 03_build_identity_gallery.py first.")
    if args.limit_identities:
        identities = identities[: args.limit_identities]

    device, weight_dtype = "cuda", torch.bfloat16
    torch.manual_seed(cfg["training"]["seed"])
    max_steps = args.max_train_steps or cfg["training"]["max_train_steps"]

    from dcfaces.training.sdxl_lora import load_sdxl_components

    comp = load_sdxl_components(cfg["model"]["base"], cfg["model"]["vae"], device=device, weight_dtype=weight_dtype)
    del comp.unet  # not used; a fresh UNet is reloaded per identity
    gc.collect()
    torch.cuda.empty_cache()

    class_images = {}
    if cfg["training"].get("with_prior_preservation") and not args.no_prior:
        n_class = args.num_class_images or cfg["training"]["num_class_images"]
        class_images = get_class_images(cfg, n_class, device)

    for id_dir in identities:
        print(f"training {id_dir.name}: {max_steps} steps, prior={bool(class_images)}")
        out = train_identity(id_dir, comp, class_images, cfg, device, weight_dtype, max_steps)
        print(f"  saved -> {out}")

    print(f"Done. {len(identities)} identities -> {DREAMBOOTH_LORAS}")


if __name__ == "__main__":
    main()
