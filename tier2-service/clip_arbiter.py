import io
import time
from typing import Dict, List

import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

CLASS_PROMPTS: Dict[str, List[str]] = {
    "Recycle": [
        "a photo of recyclable cardboard box or paper packaging",
        "a photo of a recyclable plastic bottle, container, or jug",
        "a photo of a recyclable aluminum metal can or glass bottle",
    ],
    "Compost": [
        "a photo of organic biodegradable food scraps or fruit peel",
        "a photo of compostable biological vegetable or plant waste",
        "a photo of decaying organic kitchen compost waste",
    ],
    "Landfill": [
        "a photo of non-recyclable general trash or mixed garbage",
        "a photo of discarded old clothes, fabric, or worn shoes",
        "a photo of a used household battery or dirty landfill waste",
    ],
}
CLASS_ORDER: List[str] = ["Recycle", "Compost", "Landfill"]


def _extract_tensor(out) -> torch.Tensor:
    return out.pooler_output if hasattr(out, "pooler_output") else out


class CLIPArbiter:
    """Zero-shot OpenAI CLIP (ViT-B/32) arbiter with cached prompt-ensemble text embeddings."""

    def __init__(self, model_id: str = "openai/clip-vit-base-patch32") -> None:
        self.model_id = model_id
        self.device = torch.device("cpu")
        self.processor = CLIPProcessor.from_pretrained(model_id)
        self.model = CLIPModel.from_pretrained(model_id).to(self.device)
        self.model.eval()
        self.class_text_embeds = self._precompute_text_embeddings()

    @torch.no_grad()
    def _precompute_text_embeddings(self) -> torch.Tensor:
        """Average normalized prompt embeddings per EcoSort class once at boot."""
        per_class = []
        for cls in CLASS_ORDER:
            inputs = self.processor(
                text=CLASS_PROMPTS[cls],
                return_tensors="pt",
                padding=True,
                truncation=True,
            )
            text_feats = _extract_tensor(self.model.get_text_features(**inputs))
            text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)
            mean_feat = text_feats.mean(dim=0)
            mean_feat = mean_feat / mean_feat.norm()
            per_class.append(mean_feat)
        return torch.stack(per_class, dim=0)  # (3, D)

    @torch.no_grad()
    def predict(self, raw_bytes: bytes) -> Dict:
        t0 = time.perf_counter()
        img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
        inputs = self.processor(images=img, return_tensors="pt")
        img_feats = _extract_tensor(self.model.get_image_features(**inputs))
        img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)

        logit_scale = self.model.logit_scale.exp()
        logits = (logit_scale * (img_feats @ self.class_text_embeds.T))[0]
        probs = torch.softmax(logits, dim=-1).cpu().numpy()

        pred_idx = int(probs.argmax())
        latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)

        return {
            "category": CLASS_ORDER[pred_idx],
            "confidence": round(float(probs[pred_idx]), 4),
            "probabilities": {
                CLASS_ORDER[i]: round(float(probs[i]), 4) for i in range(3)
            },
            "latency_ms": latency_ms,
            "model_variant": "clip-vit-b32-zeroshot",
        }
