"""AdaFace IR-50 (MS1MV2) identity embeddings.

The non-circular cross-check to ArcFace: methods that inject an ArcFace/antelopev2
embedding (IP-Adapter, HyperLoRA, PuLID) get an inflated ArcFace cosine, so AdaFace
gives an independent identity read. Consumes the 112x112 BGR crop that insightface's
norm_crop produces (same crop used for ArcFace), so no extra face detection.
"""

from __future__ import annotations

import sys

import numpy as np
import torch

from dcfaces.paths import MODELS_DIR, PROJECT_ROOT

ADAFACE_REPO = PROJECT_ROOT / "external" / "AdaFace"
ADAFACE_CKPT = MODELS_DIR / "adaface" / "adaface_ir50_ms1mv2.ckpt"


class AdaFaceScorer:
    def __init__(self, device: str = "cpu"):
        if str(ADAFACE_REPO) not in sys.path:
            sys.path.insert(0, str(ADAFACE_REPO))
        import net

        self.device = device
        self.model = net.build_model("ir_50")
        state = torch.load(str(ADAFACE_CKPT), map_location="cpu")["state_dict"]
        self.model.load_state_dict({k[6:]: v for k, v in state.items() if k.startswith("model.")})
        self.model.eval().to(device)

    @torch.no_grad()
    def embed(self, bgr_crop_112: np.ndarray) -> np.ndarray:
        """Normalized AdaFace embedding from a 112x112 BGR aligned crop."""
        x = ((bgr_crop_112.astype(np.float32) / 255.0) - 0.5) / 0.5
        t = torch.from_numpy(x.transpose(2, 0, 1)[None]).float().to(self.device)
        feat, _ = self.model(t)
        e = feat[0].cpu().numpy()
        return e / (np.linalg.norm(e) + 1e-9)

    @staticmethod
    def cosine(a: np.ndarray | None, b: np.ndarray | None) -> float:
        if a is None or b is None:
            return float("nan")
        return float(np.dot(a, b))
