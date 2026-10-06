"""Optional measurements for knowledge-base records, the same code the maintainer runs: the paper's distances from
the unbent baseline, and the two CLIP checks behind the knowledge base's broken-render rule.

Needs PyTorch and friends (not needed by the rest of the skill):
  pip install torch lpips transformers
  python kb_run.py measure my_run              # or: python kb_measure.py my_run


- cosine_distance: 1 - cosine of the final latents (ComfyUI .latent files)
- lpips_distance: LPIPS (AlexNet) at 256 px                       (needs `lpips`)
- dinov2_distance: 1 - cosine of DINOv2 ViT-B/14 CLS embeddings  (torch.hub, downloaded on first use)
- clip_distance: 1 - cosine of CLIP ViT-B/32 image embeddings     (transformers, openai/clip-vit-base-patch32)

- prompt_retention: CLIP image-prompt cosine of the render divided by the baseline's (1 = shows the prompt as well)
- clip_degenerate: CLIP zero-shot probability mass on degenerate-image prompts (noise, static, blobs...)

Each model loads lazily; one that cannot load is skipped and its metric left out, never guessed.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from PIL import Image

log = logging.getLogger(__name__)
CLIP_ID = "openai/clip-vit-base-patch32"
# the knowledge base's degeneracy prompts (its producers use the same lists)
DEGENERATE_PROMPTS = [
    "random noise", "tv static", "a blank image", "a plain solid color background",
    "a corrupted glitched image with digital artifacts", "a completely black image", "a completely white image",
    "a blurry field of colored blobs",
]
CONTENT_PROMPTS = [
    "a painting", "a photograph", "an illustration", "an artwork", "a drawing", "an abstract artwork",
    "a textured abstract composition",
]


def clip_degenerate_mass(clip_img: np.ndarray, deg_text: np.ndarray, content_text: np.ndarray,
                         scale: float = 100.0) -> np.ndarray:
    """Softmax probability mass on degenerate prompts vs content prompts, per image."""
    if len(clip_img) == 0:
        return np.zeros(0, np.float32)
    logits = scale * clip_img @ np.vstack([deg_text, content_text]).T
    logits -= logits.max(axis=1, keepdims=True)
    e = np.exp(logits)
    probs = e / e.sum(axis=1, keepdims=True)
    return probs[:, : len(deg_text)].sum(axis=1).astype(np.float32)


def load_latent(path: Path | str) -> np.ndarray:
    from safetensors.numpy import load_file

    d = load_file(str(path))
    return np.asarray(d.get("latent_tensor", next(iter(d.values()))), np.float32)


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.ravel(a).astype(np.float64), np.ravel(b).astype(np.float64)
    return float(1.0 - a @ b / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-12))


class Distances:
    def __init__(self, device: str | None = None, use: tuple[str, ...] = ("lpips", "dinov2", "clip")):
        import torch

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.use = use
        self._lpips = self._dino = self._clip = None
        self._failed: set[str] = set()
        self._base_cache: dict[tuple[str, int], object] = {}

    def _load(self, name: str):
        if name in self._failed or name not in self.use:
            return None
        try:
            if name == "lpips" and self._lpips is None:
                import lpips

                self._lpips = lpips.LPIPS(net="alex", verbose=False).eval().to(self.device)
            elif name == "dinov2" and self._dino is None:
                self._dino = self.torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14", pretrained=True,
                                                 verbose=False).eval().to(self.device)
            elif name == "clip" and self._clip is None:
                from transformers import CLIPModel, CLIPProcessor

                self._clip = (CLIPModel.from_pretrained(CLIP_ID).eval().to(self.device),
                              CLIPProcessor.from_pretrained(CLIP_ID))
        except Exception as e:  # noqa: BLE001 - a missing optional model only drops its metric
            log.warning("distance model %s unavailable: %s", name, e)
            self._failed.add(name)
            return None
        return {"lpips": self._lpips, "dinov2": self._dino, "clip": self._clip}[name]

    # Preprocessing follows the paper's compute_metrics.py: full-resolution RGB, resized with torch bilinear.
    def _rgb01(self, im: Image.Image):
        t = self.torch.from_numpy(np.asarray(im.convert("RGB"), np.float32) / 255.0)
        return t.permute(2, 0, 1).unsqueeze(0).to(self.device)

    def _resize(self, t, side: int):
        return self.torch.nn.functional.interpolate(t, size=(side, side), mode="bilinear", align_corners=False)

    def _lpips_t(self, im: Image.Image):
        return self._resize(self._rgb01(im) * 2.0 - 1.0, 256)

    def _dino_t(self, im: Image.Image):
        t = self._resize(self._rgb01(im), 224)
        mean = self.torch.tensor((0.485, 0.456, 0.406), device=self.device).view(1, 3, 1, 1)
        std = self.torch.tensor((0.229, 0.224, 0.225), device=self.device).view(1, 3, 1, 1)
        return (t - mean) / std

    def _feat(self, name: str, im: Image.Image):
        m = self._load(name)
        if m is None:
            return None
        with self.torch.no_grad():
            if name == "lpips":
                return self._lpips_t(im)
            if name == "dinov2":
                return m(self._dino_t(im)).float().cpu().numpy()
            model, proc = m
            px = proc(images=im.convert("RGB"), return_tensors="pt")["pixel_values"].to(self.device)
            f = model.get_image_features(pixel_values=px)
            f = getattr(f, "pooler_output", f)
            return f.float().cpu().numpy()

    def _text(self, texts: list[str]) -> np.ndarray | None:
        m = self._load("clip")
        if m is None:
            return None
        model, proc = m
        with self.torch.no_grad():
            t = proc(text=texts, return_tensors="pt", padding=True, truncation=True).to(self.device)
            f = model.get_text_features(input_ids=t["input_ids"], attention_mask=t["attention_mask"])
            f = getattr(f, "pooler_output", f)
        f = f.float().cpu().numpy()
        return f / np.linalg.norm(f, axis=1, keepdims=True)

    def subject(self, img: Image.Image, base: Image.Image, prompt: str | None, base_key: str | None = None) -> dict:
        """prompt_retention (CLIP image-prompt cosine, output / baseline) and clip_degenerate (zero-shot mass on
        degenerate-image prompts), for kb.degenerate_reasons. Empty when CLIP cannot load."""
        if "clip" not in self.use or self._load("clip") is None:
            return {}
        if not hasattr(self, "_zs"):
            t = self._text(DEGENERATE_PROMPTS + CONTENT_PROMPTS)
            self._zs = (t[: len(DEGENERATE_PROMPTS)], t[len(DEGENERATE_PROMPTS):])
            self._ptext: dict[str, np.ndarray] = {}
        unit = lambda f: f / np.linalg.norm(f, axis=1, keepdims=True)  # noqa: E731
        f = unit(self._feat("clip", img))
        out = {"clip_degenerate": round(float(clip_degenerate_mass(f, *self._zs)[0]), 4)}
        if prompt:
            p = self._ptext.get(prompt)
            if p is None:
                p = self._ptext[prompt] = self._text([prompt])[0]
            ck = (base_key, hash("clip")) if base_key else None
            bf = self._base_cache.get(ck) if ck else None
            if bf is None:
                bf = self._feat("clip", base)
                if ck:
                    self._base_cache[ck] = bf
            base_sim = float(unit(bf)[0] @ p)
            if base_sim > 0:
                out["prompt_retention"] = round(float(f[0] @ p) / base_sim, 4)
        return out

    def compare(self, img: Image.Image, base: Image.Image, base_key: str | None = None,
                latent: Path | None = None, base_latent: Path | None = None) -> dict:
        """name -> value for every metric that could be computed. base_key caches the baseline's features."""
        out: dict[str, float] = {}
        if latent and base_latent and Path(latent).exists() and Path(base_latent).exists():
            out["cosine_distance"] = round(cosine_distance(load_latent(latent), load_latent(base_latent)), 6)
        for name, key in (("lpips", "lpips_distance"), ("dinov2", "dinov2_distance"), ("clip", "clip_distance")):
            if name not in self.use or name in self._failed:
                continue
            ck = (base_key, hash(name)) if base_key else None
            bf = self._base_cache.get(ck) if ck else None
            if bf is None:
                bf = self._feat(name, base)
                if ck and bf is not None:
                    self._base_cache[ck] = bf
            f = self._feat(name, img)
            if f is None or bf is None:
                continue
            if name == "lpips":
                with self.torch.no_grad():
                    out[key] = round(float(self._lpips(bf, f).item()), 6)
            else:
                out[key] = round(cosine_distance(f, bf), 6)
        return out


def measure_dir(root: Path | str, device: str | None = None, redo: bool = False, log_every: int = 50) -> dict:
    """Add the distances and the CLIP checks to every record of a run (or a knowledge-base working copy) that lacks
    them, and re-decide `degenerate` with the knowledge base's rule. Only measurements.json changes."""
    import json
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import kb

    root = Path(root)
    dist, n = Distances(device=device), {"measured": 0, "already": 0, "no_images": 0}
    for f in sorted((root / "records").glob("*/*/*/record.json")):
        d, rec = f.parent, json.loads(f.read_text(encoding="utf-8"))
        m = kb.load_measurements(d) or {"record": rec["id"], "values": {}}
        vals = m.setdefault("values", {})
        if "lpips_distance" in vals and "clip_degenerate" in vals and not redo:
            n["already"] += 1
            continue
        bid = (rec.get("outputs") or {}).get("baseline_id")
        base = next((p for p in (root / "baselines" / f"{bid}.webp", root / "baselines" / f"{bid}.png") if p.exists()),
                    None)
        img = d / (rec.get("outputs") or {}).get("image", "output.webp")
        if base is None or not img.exists():
            n["no_images"] += 1
            continue
        a, b = Image.open(img).convert("RGB"), Image.open(base).convert("RGB")
        new = {**dist.compare(a, b, base_key=bid), **dist.subject(a, b, rec["setup"].get("prompt"), base_key=bid)}
        for k, v in new.items():
            vals[k] = kb.measure(k, v)
        flags = (vals.get("pixel_flags") or {}).get("value", [])
        reasons = kb.degenerate_reasons(flags, new.get("prompt_retention"), new.get("clip_degenerate"))
        vals["degenerate"] = kb.measure("degenerate", bool(reasons), reasons=reasons)
        (d / "measurements.json").write_text(json.dumps(m, indent=1), encoding="utf-8")
        n["measured"] += 1
        if n["measured"] % log_every == 0:
            print(f"  {n['measured']} measured", flush=True)
    return n


if __name__ == "__main__":
    import argparse
    import json

    ap = argparse.ArgumentParser(description="Measure a run's records (distances and CLIP checks)")
    ap.add_argument("dir")
    ap.add_argument("--device", default=None)
    ap.add_argument("--redo", action="store_true")
    a = ap.parse_args()
    print(json.dumps(measure_dir(a.dir, a.device, a.redo)))
