"""Download the pretrained weights the benchmark needs that are available as a clean
HuggingFace pull. Run once, after scripts/setup_external.sh, with the main .venv
active. Idempotent (hf downloads are cached).

Three weights are NOT here because they need a license click-through or live off
HuggingFace; they are listed at the end with their sources and target paths. The
demographic LoRA and the per-identity DreamBooth LoRAs are trained, not downloaded
(scripts 05 and 07).
"""

from __future__ import annotations

from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download

from dcfaces.paths import HF_CACHE_DIR, MODELS_DIR, PROJECT_ROOT


def ip_adapter() -> None:
    out = MODELS_DIR / "ip_adapter"
    out.mkdir(parents=True, exist_ok=True)
    print("IP-Adapter FaceID Plus v2 (SDXL)...")
    hf_hub_download("h94/IP-Adapter-FaceID", "ip-adapter-faceid-plusv2_sdxl.bin", local_dir=str(out))
    hf_hub_download("h94/IP-Adapter-FaceID", "ip-adapter-faceid-plusv2_sdxl_lora.safetensors", local_dir=str(out))
    print("IP-Adapter image encoder...")
    snapshot_download("h94/IP-Adapter", local_dir=str(out / "_hub"), allow_patterns=["models/image_encoder/*"])


def photomaker() -> None:
    out = MODELS_DIR / "photomaker"
    out.mkdir(parents=True, exist_ok=True)
    print("PhotoMaker v2...")
    hf_hub_download("TencentARC/PhotoMaker-V2", "photomaker-v2.bin", local_dir=str(out))


def pickscore() -> None:
    out = MODELS_DIR / "pickscore"
    out.mkdir(parents=True, exist_ok=True)
    print("PickScore v1...")
    snapshot_download("yuvalkirstain/PickScore_v1", local_dir=str(out))


def hyperlora() -> None:
    out = MODELS_DIR / "hyperlora"
    out.mkdir(parents=True, exist_ok=True)
    print("HyperLoRA (bytedance-research/HyperLoRA, CC-BY-NC)...")
    snapshot_download(
        "bytedance-research/HyperLoRA",
        local_dir=str(out),
        allow_patterns=["sdxl_hyper_id_lora_v1_fidelity/*", "sdxl_hyper_id_lora_v1_edit/*", "*.md", "*.json"],
    )

    # HyperLoRA expects a specific on-disk layout (CLIP encoder + the LoRAs +
    # insightface antelopev2) under a single root; build it with symlinks.
    root = MODELS_DIR / "hyperlora_root"
    hl = root / "hyper_lora"
    clip_proc = hl / "clip_processor" / "clip_vit_large_14_processor"
    clip_vit = hl / "clip_vit" / "clip_vit_large_14"
    hyper = hl / "hyper_lora"
    iface = root / "insightface" / "models"
    for d in (clip_proc, clip_vit, hyper, iface):
        d.mkdir(parents=True, exist_ok=True)

    print("CLIP ViT-L/14 (encoder)...")
    snapshot_download("openai/clip-vit-large-patch14", local_dir=str(clip_vit),
                      allow_patterns=["config.json", "model.safetensors"])
    print("CLIP ViT-L/14 (processor)...")
    snapshot_download("openai/clip-vit-large-patch14", local_dir=str(clip_proc),
                      allow_patterns=["preprocessor_config.json", "tokenizer.json", "tokenizer_config.json",
                                      "vocab.json", "merges.txt", "special_tokens_map.json"])

    def link(src: Path, dst: Path) -> None:
        if not dst.exists():
            dst.symlink_to(src.resolve())
            print(f"linked {dst.name} -> {src}")

    for v in ("sdxl_hyper_id_lora_v1_fidelity", "sdxl_hyper_id_lora_v1_edit"):
        link(out / v, hyper / v)
    # antelopev2 is fetched by setup_external.sh / PuLID; see the manual notes below.
    link(PROJECT_ROOT / "external" / "PuLID" / "models" / "antelopev2", iface / "antelopev2")


def realvisxl() -> None:
    print("RealVisXL v4.0 (HyperLoRA base)...")
    snapshot_download("SG161222/RealVisXL_V4.0", cache_dir=str(HF_CACHE_DIR / "hub"),
                      allow_patterns=["*.json", "*.txt", "*fp16*", "*.safetensors"],
                      ignore_patterns=["*non-ema*", "*.ckpt", "*.bin"])


MANUAL = """
Manual downloads (license click-through or non-HuggingFace host):

  AdaFace IR-50 (MS1MV2)   -> models/adaface/adaface_ir50_ms1mv2.ckpt
    From the AdaFace repo "Pretrained Models" table (Google Drive):
    https://github.com/mk-minchul/AdaFace

  MiVOLO age model         -> models/mivolo/mivolo_imdb.pth.tar
  MiVOLO person+face yolo  -> models/mivolo/detector/yolov8x_person_face.pt
    From the MiVOLO repo model table:
    https://github.com/WildChlamydia/MiVOLO

  insightface antelopev2   -> external/PuLID/models/antelopev2/
    PuLID downloads this on first run; or fetch the antelopev2 pack manually.
    HyperLoRA's symlink layout (above) points at this directory.

SDXL 1.0 base + madebyollin/sdxl-vae-fp16-fix download automatically on first use.
"""


def main() -> None:
    ip_adapter()
    photomaker()
    pickscore()
    hyperlora()
    realvisxl()
    print(MANUAL)
    print("WEIGHTS_OK")


if __name__ == "__main__":
    main()
