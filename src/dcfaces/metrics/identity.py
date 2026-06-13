"""ArcFace identity embeddings (insightface buffalo_l) for identity preservation.

Returns the normalized recognition embedding of the most-confident face so two
faces can be compared by cosine similarity. The same 25% edge-padding as the
gallery builder is applied, since FFHQ-style reference crops otherwise dodge the
detector.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

from dcfaces.paths import INSIGHTFACE_DIR


class IdentityScorer:
    def __init__(self, pad_frac: float = 0.25, det_size: int = 640, use_gpu: bool = False):
        from insightface.app import FaceAnalysis

        self.pad_frac = pad_frac
        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"] if use_gpu else ["CPUExecutionProvider"]
        )
        self.app = FaceAnalysis(name="buffalo_l", root=str(INSIGHTFACE_DIR), providers=providers)
        self.app.prepare(ctx_id=0 if use_gpu else -1, det_size=(det_size, det_size))

    def embed(self, image: Image.Image) -> np.ndarray | None:
        """Normalized ArcFace embedding of the most-confident face, or None."""
        out = self.detect(image)
        return None if out is None else out[0]

    def detect(self, image: Image.Image):
        """Return (ArcFace embedding, 112x112 BGR aligned crop) of the best face, or None.

        The crop feeds AdaFace, so both identity metrics share one detection.
        """
        from insightface.utils import face_align

        arr = np.array(image.convert("RGB"))
        if self.pad_frac > 0:
            p = int(arr.shape[0] * self.pad_frac)
            arr = np.pad(arr, ((p, p), (p, p), (0, 0)), mode="edge")
        bgr = arr[:, :, ::-1]
        faces = self.app.get(bgr)
        if not faces:
            return None
        f = max(faces, key=lambda x: x.det_score)
        crop = face_align.norm_crop(bgr, f.kps, image_size=112)
        return f.normed_embedding.astype(np.float32), crop

    @staticmethod
    def cosine(a: np.ndarray | None, b: np.ndarray | None) -> float:
        """Cosine similarity of two normalized embeddings (NaN if either missing)."""
        if a is None or b is None:
            return float("nan")
        return float(np.dot(a, b))
