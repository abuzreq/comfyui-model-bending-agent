"""Sharing a render by hand: scripts/contribute.py turns two ComfyUI pictures into a checked contribution folder.

    uv run --no-project --with pytest pytest tests/test_contribute.py
"""

from __future__ import annotations

import json
import struct
import sys
import zlib
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "skills" / "comfyui-model-bending" / "scripts"))
import contribute  # noqa: E402
import kb  # noqa: E402

BENDS = {"bends": [{"path": "output_blocks.6.0", "module_type": "scale", "module_args": {"scale_factor": 0.7}}],
         "steps_min": 4, "steps_max": 5}


def workflow(bent: bool = True, seed: int = 1234) -> dict:
    model = ["5", 0] if bent else ["1", 0]
    wf = {"1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "LCM_Dreamshaper_v7_4k.safetensors"}},
          "6": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["1", 1], "text": "portrait of an old fisherman"}},
          "7": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["1", 1], "text": "lowres, watermark"}},
          "8": {"class_type": "EmptyLatentImage", "inputs": {"width": 512, "height": 512, "batch_size": 1}},
          "9": {"class_type": "KSampler", "inputs": {"model": model, "positive": ["6", 0], "negative": ["7", 0],
                "latent_image": ["8", 0], "seed": seed, "steps": 6, "cfg": 1.5, "sampler_name": "lcm",
                "scheduler": "sgm_uniform", "denoise": 1.0}}}
    if bent:
        wf["5"] = {"class_type": "ApplyBendsFromJSON", "inputs": {"model": ["1", 0], "bends_json": json.dumps(BENDS)}}
    return wf


def png(path: Path, prompt: dict | None) -> Path:
    """A 1×1 PNG with ComfyUI's tEXt "prompt" chunk, the way Save Image writes it."""
    def chunk(tag: bytes, body: bytes) -> bytes:
        return struct.pack(">I", len(body)) + tag + body + struct.pack(">I", zlib.crc32(tag + body) & 0xFFFFFFFF)
    data = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0))
    if prompt is not None:
        data += chunk(b"tEXt", b"prompt\0" + json.dumps(prompt).encode("latin-1"))
    data += chunk(b"IDAT", zlib.compress(b"\0\0\0\0")) + chunk(b"IEND", b"")
    path.write_bytes(data)
    return path


def test_two_comfyui_pictures_become_a_checked_contribution(tmp_path):
    d = contribute.prepare(png(tmp_path / "bent.png", workflow()), png(tmp_path / "unbent.png", workflow(bent=False)),
                           tmp_path / "out", name="Ada", agree=True, describe="The hat turns to rock.")
    rec = json.loads((d / "record.json").read_text(encoding="utf-8"))
    assert d.parent.name == "artist" and d.name == rec["id"]
    assert rec["setup"]["prompt"] == "portrait of an old fisherman" and rec["setup"]["negative"] == "lowres, watermark"
    assert rec["bends"][0]["window"] == "late"  # the document's steps_min / steps_max reach the bend
    assert rec["contribution"] == {"by": {"type": "human", "name": "Ada"}, "via": "hand",
                                   "agreed": {"license": "CC0-1.0", "at": rec["contribution"]["agreed"]["at"]}}
    assert json.loads((d / "workflow.json").read_text(encoding="utf-8"))["5"]["class_type"] == "ApplyBendsFromJSON"
    assert (tmp_path / "out" / "baselines" / f"{rec['outputs']['baseline_id']}.png").exists()
    line = json.loads((d / "interpretations.jsonl").read_text(encoding="utf-8"))
    assert line["author"] == {"type": "human", "name": "Ada"} and line["kind"] == "change"
    assert contribute.check(tmp_path / "out") == []
    assert contribute.check(d) == []


def test_not_named_leaves_the_name_out(tmp_path):
    d = contribute.prepare(png(tmp_path / "b.png", workflow()), png(tmp_path / "u.png", workflow(bent=False)),
                           tmp_path / "out", name=None, agree=True, describe="Rocky.")
    rec = json.loads((d / "record.json").read_text(encoding="utf-8"))
    assert rec["contribution"]["by"] == {"type": "human", "name": None}
    assert "name" not in json.loads((d / "interpretations.jsonl").read_text(encoding="utf-8"))["author"]


@pytest.mark.parametrize("case, message", [
    ("no-cc0", "CC0"), ("jpeg", "JPEG"), ("no-workflow", "no ComfyUI workflow"), ("unbent-is-bent", "bent too"),
    ("other-seed", "seed"), ("no-bend", "no bend")])
def test_what_is_refused_and_why(tmp_path, case, message):
    bent, unbent = png(tmp_path / "b.png", workflow()), png(tmp_path / "u.png", workflow(bent=False))
    if case == "jpeg":
        bent.write_bytes(b"\xff\xd8\xff\xe0" + b"0" * 20)
    if case == "no-workflow":
        png(bent, None)
    if case == "unbent-is-bent":
        png(unbent, workflow())
    if case == "other-seed":
        png(unbent, workflow(bent=False, seed=7))
    if case == "no-bend":
        png(bent, workflow(bent=False))
    with pytest.raises(contribute.ContributionError, match=message):
        contribute.prepare(bent, unbent, tmp_path / "out", name="Ada", agree=case != "no-cc0")


def test_check_finds_a_wrong_id_and_a_missing_baseline(tmp_path):
    d = contribute.prepare(png(tmp_path / "b.png", workflow()), png(tmp_path / "u.png", workflow(bent=False)),
                           tmp_path / "out", name="Ada", agree=True)
    rec = json.loads((d / "record.json").read_text(encoding="utf-8"))
    rec["setup"]["seed"] = 99
    (d / "record.json").write_text(json.dumps(rec), encoding="utf-8")
    for p in (tmp_path / "out" / "baselines").iterdir():
        p.unlink()
    problems = " | ".join(contribute.check(tmp_path / "out"))
    assert "the id should be" in problems and "unbent picture" in problems


def test_comfyui_webp_metadata_is_read(tmp_path):
    exif = b"Exif\0\0" + b"prompt:" + json.dumps(workflow()).encode() + b"\0"
    vp8l = b"VP8L" + struct.pack("<I", 5) + b"\x2f\0\0\0\0\0"
    body = b"WEBP" + vp8l + b"EXIF" + struct.pack("<I", len(exif)) + exif + (b"\0" if len(exif) & 1 else b"")
    p = tmp_path / "b.webp"
    p.write_bytes(b"RIFF" + struct.pack("<I", len(body)) + body)
    data, kind, prompt = contribute.read_picture(p)
    assert kind == "webp-lossless" and prompt["5"]["class_type"] == "ApplyBendsFromJSON"
