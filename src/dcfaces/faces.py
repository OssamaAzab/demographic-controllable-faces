"""insightface wrapper for face detection + head pose, configured for this repo.

Two non-obvious fixes are baked in (both discovered empirically on FFHQ):

  1. insightface IGNORES the INSIGHTFACE_HOME env var and hardcodes
     ~/.insightface. We pass root=INSIGHTFACE_DIR so the buffalo_l models live
     in-repo (.cache/insightface) instead of the user's near-full home dir.

  2. FFHQ aligned crops have the face filling the whole frame, which RetinaFace
     detects poorly (det score ~0.57, or misses entirely). Edge-padding the
     image before detection so the face is no longer full-frame recovers this
     (det ~0.85, no misses). Default pad_frac=0.25.

Note: onnxruntime-gpu here is a CUDA-11 build but the env is CUDA-12, so the GPU
provider fails to load; we default to CPU. Fine for one-off curation (~130 ms/
image). Revisit (CUDA-12 onnxruntime build) before GPU-bound inference scripts.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image

from dcfaces.paths import INSIGHTFACE_DIR


@dataclass
class FaceDet:
    det_score: float
    yaw: float
    pitch: float
    roll: float
    n_faces: int

    @property
    def frontality(self) -> float:
        """Higher (closer to 0) = more frontal/neutral. Used to rank references."""
        return -(abs(self.yaw) + abs(self.pitch) + 0.5 * abs(self.roll))


class FaceAnalyzer:
    def __init__(self, pad_frac: float = 0.25, det_size: int = 640, use_gpu: bool = False):
        from insightface.app import FaceAnalysis

        self.pad_frac = pad_frac
        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"] if use_gpu else ["CPUExecutionProvider"]
        )
        self.app = FaceAnalysis(name="buffalo_l", root=str(INSIGHTFACE_DIR), providers=providers)
        self.app.prepare(ctx_id=0 if use_gpu else -1, det_size=(det_size, det_size))

    def analyze(self, image: Image.Image) -> FaceDet | None:
        """Detect the most-confident face; return its score + pose, or None."""
        arr = np.array(image.convert("RGB"))
        if self.pad_frac > 0:
            p = int(arr.shape[0] * self.pad_frac)
            arr = np.pad(arr, ((p, p), (p, p), (0, 0)), mode="edge")
        faces = self.app.get(arr[:, :, ::-1])  # insightface expects BGR
        if not faces:
            return None
        f = max(faces, key=lambda x: x.det_score)
        pose = getattr(f, "pose", None)
        pitch, yaw, roll = (float(pose[0]), float(pose[1]), float(pose[2])) if pose is not None else (0.0, 0.0, 0.0)
        return FaceDet(det_score=float(f.det_score), yaw=yaw, pitch=pitch, roll=roll, n_faces=len(faces))
