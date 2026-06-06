"""04: Hybrid BLIP-2 + FairFace captioning of FFHQ training subset.

Inputs:  data/ffhq_train.jsonl, data/ffhq_val.jsonl
Outputs: data/ffhq_metadata.jsonl  (one line per image with caption + tags)

Caption format (the contribution of this script):
    "a photo of a {age_bucket} year old {race} {gender}, {blip2_description}"

Why hybrid:
    Pure BLIP-2 → captions are generic, no demographic anchor.
    Pure FairFace tags → no visual variation, template memorization.
    Hybrid → demographic responsiveness + visual variety.

TODO:
- Load BLIP-2 (Salesforce/blip2-opt-2.7b)
- Load FairFace classifier (dchen236/FairFace)
- For each image: get FairFace tags + BLIP-2 caption, combine via template
- Write to ffhq_metadata.jsonl with fields:
    {image_path, image_id, caption, fairface_age, fairface_race, fairface_gender, blip2_desc}

Expected runtime: ~10 hours on 20 GB GPU for 30k images.
"""

from dcfaces.paths import FFHQ_METADATA, FFHQ_TRAIN, FFHQ_VAL, ensure_dirs


def main() -> None:
    ensure_dirs()
    print(f"TODO: caption {FFHQ_TRAIN} + {FFHQ_VAL} → {FFHQ_METADATA}")


if __name__ == "__main__":
    main()
