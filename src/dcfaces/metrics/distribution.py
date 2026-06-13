"""CLIP-FID and KID with face-crop preprocessing.

Both generations and the FFHQ reference set are cropped to a face-centred square
before scoring, so the distance reflects face quality rather than background — the
v1 lesson that plain FID just rewards "looks FFHQ-like". CLIP-FID (clean-fid with a
CLIP backbone) is more robust to that than Inception FID.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

from dcfaces.paths import INSIGHTFACE_DIR


class FaceCropper:
    def __init__(self, size: int = 224, pad_frac: float = 0.25, margin: float = 1.4):
        from insightface.app import FaceAnalysis

        self.size = size
        self.pad_frac = pad_frac
        self.margin = margin
        self.app = FaceAnalysis(name="buffalo_l", root=str(INSIGHTFACE_DIR), providers=["CPUExecutionProvider"])
        self.app.prepare(ctx_id=-1, det_size=(640, 640))

    def crop(self, image: Image.Image) -> Image.Image | None:
        arr = np.array(image.convert("RGB"))
        p = int(arr.shape[0] * self.pad_frac)
        arr = np.pad(arr, ((p, p), (p, p), (0, 0)), mode="edge")
        faces = self.app.get(arr[:, :, ::-1])
        if not faces:
            return None
        f = max(faces, key=lambda x: x.det_score)
        x1, y1, x2, y2 = f.bbox
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        half = max(x2 - x1, y2 - y1) * self.margin / 2
        h, w = arr.shape[:2]
        x1, y1 = max(0, int(cx - half)), max(0, int(cy - half))
        x2, y2 = min(w, int(cx + half)), min(h, int(cy + half))
        return Image.fromarray(arr[y1:y2, x1:x2]).resize((self.size, self.size))


def compute_fid_kid(gen_dir, ref_dir) -> tuple[float, float]:
    """CLIP-FID and KID (x1000) between two folders of face crops."""
    from cleanfid import fid

    clip_fid = fid.compute_fid(str(gen_dir), str(ref_dir), mode="clean", model_name="clip_vit_b_32", verbose=False)
    kid = fid.compute_kid(str(gen_dir), str(ref_dir), mode="clean", verbose=False)
    return float(clip_fid), float(kid) * 1000.0
