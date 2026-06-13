# Sample gallery

A curated subset of the benchmark generations (the full ~18k images are not
shipped; regenerate them with the pipeline). Selection is fixed in
`scripts/make_sample_gallery.py`. All on SDXL 1.0 + fp16-fix VAE, seed 42
(HyperLoRA on RealVisXL v4.0).

## references

The reference photo for each identity used below (FFHQ, public domain). Identity preservation in the other panels is judged against these.

## trade-off

One identity (id_002), the prompt "depicted as East Asian", every method, same seed. PuLID locks the reference and ignores the race request; HyperLoRA follows the prompt but drifts off the reference.

## rescue

pulid_only vs pulid_with_demo_lora vs pulid_delayed on the same race prompt. Stacking the demographic LoRA does not shift race; holding identity injection off for the first 13 steps (delayed) does.

## age-limit

The rescue is attribute-dependent. At age_20 and age_80, pulid_delayed cannot move the structural age (it stays identity-locked), while a per-identity DreamBooth LoRA can.

## accessory

Accessory control (round glasses, baseball cap) differs by method: PhotoMaker renders the accessory readily, PuLID resists it in favour of the locked reference.

