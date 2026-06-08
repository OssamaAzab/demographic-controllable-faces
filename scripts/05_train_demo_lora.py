"""05: Train the demographic LoRA on FFHQ + hybrid captions.

Standard SDXL UNet-attention LoRA (PEFT) at the locked recipe: rank 32 / alpha 16,
BF16 weights + fp32 LoRA params under autocast, gradient checkpointing, AdamW8bit,
xformers, EMA 0.9999, 10k steps, LR 1e-4 cosine + 200 warmup, effective batch 4.

Every checkpointing_steps: save the EMA LoRA AND run a task-aware eval (generate
the tiny eval-prompt set, score age MAE + race acc with FairFace) -> eval_log.jsonl.
That curve, NOT val-MSE, is the selection signal (Part B lesson). 06 then picks a
checkpoint on the Pareto frontier.

Config: configs/demo_lora.yaml. Inputs: data/ffhq_train.jsonl + data/ffhq_metadata.jsonl.
Outputs: models/demo_lora_checkpoints/step_{1000..10000}/pytorch_lora_weights.safetensors (EMA)
         models/demo_lora_checkpoints/eval_log.jsonl

Smoke-test before the real run, e.g.:
    python scripts/05_train_demo_lora.py --max-train-steps 20 --checkpointing-steps 10
"""

from __future__ import annotations

import argparse
import json
import os

import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from dcfaces.paths import CONFIG_DIR, DEMO_LORA_CKPTS, FFHQ_METADATA, FFHQ_TRAIN, ensure_dirs

RACES = ["white", "Black", "East Asian", "Southeast Asian", "South Asian", "Middle Eastern", "Latino"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--max-train-steps", type=int, default=None, help="Override config (smoke test).")
    parser.add_argument("--checkpointing-steps", type=int, default=None, help="Override config.")
    parser.add_argument("--no-eval", action="store_true", help="Skip mid-training task eval.")
    parser.add_argument("--no-wandb", action="store_true", help="Disable wandb (local jsonl still written).")
    args = parser.parse_args()

    ensure_dirs()
    if not FFHQ_METADATA.exists():
        raise SystemExit(f"{FFHQ_METADATA} missing — run 04_caption_ffhq.py first.")

    cfg = yaml.safe_load((CONFIG_DIR / "demo_lora.yaml").read_text())
    tcfg = cfg["training"]
    device = "cuda"
    weight_dtype = torch.bfloat16
    torch.manual_seed(tcfg["seed"])

    max_steps = args.max_train_steps or tcfg["max_train_steps"]
    ckpt_steps = args.checkpointing_steps or cfg["checkpointing"]["steps"]
    grad_accum = tcfg["gradient_accumulation_steps"]

    # ---- Heavy imports after the cheap checks above ----
    import bitsandbytes as bnb
    from diffusers import EulerDiscreteScheduler, StableDiffusionXLPipeline
    from diffusers.optimization import get_scheduler
    from diffusers.training_utils import EMAModel, cast_training_params
    from peft import LoraConfig

    from dcfaces.training.dataset import FFHQCaptionDataset, collate_fn
    from dcfaces.training.sdxl_lora import (
        build_time_ids,
        encode_prompt,
        load_sdxl_components,
        save_lora,
    )

    base = cfg["model"]["base"]
    comp = load_sdxl_components(base, cfg["model"]["vae"], device=device, weight_dtype=weight_dtype)
    unet, vae, scheduler = comp.unet, comp.vae, comp.noise_scheduler

    # LoRA on UNet attention; LoRA params kept fp32 for stable optimization.
    unet.add_adapter(
        LoraConfig(
            r=cfg["lora"]["rank"],
            lora_alpha=cfg["lora"]["alpha"],
            init_lora_weights="gaussian",
            target_modules=cfg["lora"]["target_modules"],
        )
    )
    cast_training_params(unet, dtype=torch.float32)
    if tcfg.get("enable_xformers"):
        try:
            unet.enable_xformers_memory_efficient_attention()
        except Exception as e:  # noqa: BLE001
            print(f"xformers unavailable ({e}); continuing without it.")
    if tcfg.get("gradient_checkpointing"):
        unet.enable_gradient_checkpointing()

    lora_params = [p for p in unet.parameters() if p.requires_grad]
    print(f"trainable LoRA params: {sum(p.numel() for p in lora_params)/1e6:.1f}M")

    optimizer = bnb.optim.AdamW8bit(lora_params, lr=tcfg["learning_rate"])
    lr_scheduler = get_scheduler(
        tcfg["lr_scheduler"], optimizer,
        num_warmup_steps=tcfg["lr_warmup_steps"], num_training_steps=max_steps,
    )
    ema = EMAModel(lora_params, decay=cfg["ema"]["decay"]) if cfg["ema"]["enabled"] else None

    dataset = FFHQCaptionDataset(FFHQ_TRAIN, FFHQ_METADATA, resolution=cfg["data"]["resolution"])
    print(f"dataset: {len(dataset)} captioned train images")
    loader = DataLoader(
        dataset, batch_size=tcfg["train_batch_size"], shuffle=True,
        collate_fn=collate_fn, num_workers=4, pin_memory=True, drop_last=True,
    )

    # Eval setup (pipeline reuses the in-training components; FairFace scores).
    eval_enabled = cfg["mid_training_eval"]["enabled"] and not args.no_eval
    pipe = classifier = None
    if eval_enabled:
        from dcfaces.demographics import FairFaceClassifier
        pipe = StableDiffusionXLPipeline(
            vae=vae, text_encoder=comp.text_encoders[0], text_encoder_2=comp.text_encoders[1],
            tokenizer=comp.tokenizers[0], tokenizer_2=comp.tokenizers[1], unet=unet,
            scheduler=EulerDiscreteScheduler.from_pretrained(base, subfolder="scheduler"),
        )
        pipe.set_progress_bar_config(disable=True)
        classifier = FairFaceClassifier()

    use_wandb = not args.no_wandb
    if use_wandb:
        try:
            import wandb
            wandb.init(project=cfg["logging"]["wandb_project"],
                       mode=os.environ.get("WANDB_MODE", "offline"), config=cfg)
        except Exception as e:  # noqa: BLE001
            print(f"wandb disabled ({e})")
            use_wandb = False

    eval_log_path = DEMO_LORA_CKPTS / "eval_log.jsonl"

    def compute_loss(batch) -> torch.Tensor:
        pixel_values = batch["pixel_values"].to(device, dtype=torch.float32)
        with torch.no_grad():
            latents = vae.encode(pixel_values).latent_dist.sample() * vae.config.scaling_factor
            latents = latents.to(weight_dtype)
            prompt_embeds, pooled = encode_prompt(batch["captions"], comp.tokenizers, comp.text_encoders, device)
        noise = torch.randn_like(latents)
        timesteps = torch.randint(0, scheduler.config.num_train_timesteps, (latents.shape[0],), device=device).long()
        noisy = scheduler.add_noise(latents, noise, timesteps)
        added = {
            "text_embeds": pooled.to(weight_dtype),
            "time_ids": build_time_ids(batch["original_sizes"], batch["crop_top_lefts"], batch["target_sizes"], device, weight_dtype),
        }
        with torch.autocast("cuda", dtype=weight_dtype):
            model_pred = unet(noisy, timesteps, prompt_embeds.to(weight_dtype), added_cond_kwargs=added).sample
        target = noise if scheduler.config.prediction_type == "epsilon" else scheduler.get_velocity(latents, noise, timesteps)
        return F.mse_loss(model_pred.float(), target.float(), reduction="mean")

    def checkpoint(step: int) -> None:
        from dcfaces.training.eval import run_task_eval
        step_dir = DEMO_LORA_CKPTS / f"step_{step}"
        step_dir.mkdir(parents=True, exist_ok=True)
        if ema:
            ema.store(lora_params); ema.copy_to(lora_params)  # save/eval EMA weights
        save_lora(unet, step_dir)
        metrics = {}
        if eval_enabled:
            unet.eval()
            try:
                metrics = run_task_eval(
                    pipe, cfg["mid_training_eval"]["prompts"], cfg["mid_training_eval"]["seeds"],
                    classifier, RACES,
                    steps=cfg["mid_training_eval"]["inference_steps"],
                    guidance=cfg["mid_training_eval"]["guidance_scale"],
                )
            except Exception as e:  # noqa: BLE001  (never let eval kill a 7h run)
                print(f"mid-training eval failed at step {step}: {e}")
            unet.train()
        if ema:
            ema.restore(lora_params)
        rec = {"step": step, **metrics}
        with open(eval_log_path, "a") as f:  # append-and-close: durable per checkpoint
            f.write(json.dumps(rec) + "\n")
        print(f"[ckpt {step}] saved -> {step_dir} | {metrics}")
        if use_wandb and metrics:
            import wandb
            wandb.log({f"eval/{k}": v for k, v in metrics.items()}, step=step)

    # ---- Training loop ----
    unet.train()
    global_step = 0
    accum = 0
    optimizer.zero_grad()
    progress = tqdm(total=max_steps, desc="training")
    while global_step < max_steps:
        for batch in loader:
            loss = compute_loss(batch) / grad_accum
            loss.backward()
            accum += 1
            if accum % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(lora_params, 1.0)
                optimizer.step(); lr_scheduler.step(); optimizer.zero_grad()
                if ema:
                    ema.step(lora_params)
                global_step += 1
                progress.update(1)
                if global_step % cfg["logging"]["log_every_n_steps"] == 0:
                    lr = lr_scheduler.get_last_lr()[0]
                    progress.set_postfix(loss=f"{loss.item()*grad_accum:.4f}", lr=f"{lr:.2e}")
                    if use_wandb:
                        import wandb
                        wandb.log({"train/loss": loss.item() * grad_accum, "train/lr": lr}, step=global_step)
                if global_step % ckpt_steps == 0 or global_step >= max_steps:
                    checkpoint(global_step)
                if global_step >= max_steps:
                    break
        if global_step >= max_steps:
            break

    print(f"Done. {global_step} steps. Checkpoints -> {DEMO_LORA_CKPTS}")


if __name__ == "__main__":
    main()
