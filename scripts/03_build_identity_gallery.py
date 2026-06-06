"""03: Curate 10 FFHQ images as benchmark identities, covering FairFace's 7 race
classes × both genders × age diversity.

Inputs:  data/ffhq_test.jsonl  (we pick identities from the test split to avoid
                                 any chance of training-time leakage)
Outputs: data/identity_gallery/id_{001..010}/ref.jpg
         data/identity_gallery/id_{001..010}/metadata.json

Each metadata.json contains:
    {
        "id": "id_001",
        "ffhq_image_id": "12345",
        "fairface": {"age_bucket": "30-39", "race": "East Asian", "gender": "Female"},
        "selection_notes": "neutral expression, frontal pose, soft lighting"
    }

Curation strategy:
    1. Run FairFace on all test images
    2. Group by (race, gender, age_bucket)
    3. Within each target demographic, rank by neutrality (eyes open, frontal pose)
       — use MediaPipe FaceMesh landmarks for a yaw/pitch score
    4. Pick the top candidate per target slot
    5. Allow manual override via configs/identity_gallery_overrides.yaml

TODO: implement.
"""

from dcfaces.paths import FFHQ_TEST, IDENTITY_GALLERY, ensure_dirs


def main() -> None:
    ensure_dirs()
    print(f"TODO: build identity gallery from {FFHQ_TEST} → {IDENTITY_GALLERY}")


if __name__ == "__main__":
    main()
