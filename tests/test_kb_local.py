"""The user's own knowledge base (scripts/kb_local.py): log_round against a stub ComfyUI, source choice, find.

    uv run --no-project --with pytest --with pillow --with numpy pytest tests/test_kb_local.py
"""

from __future__ import annotations

import http.server
import importlib
import io
import json
import sys
import threading
from pathlib import Path

import pytest
from PIL import Image

SCRIPTS = Path(__file__).resolve().parents[1] / "skills" / "comfyui-model-bending" / "scripts"
sys.path.insert(0, str(SCRIPTS))

PROMPT = {
    "4": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "v1-5-pruned-emaonly.safetensors"}},
    "6": {"class_type": "CLIPTextEncode", "inputs": {"text": "my secret prompt", "clip": ["4", 1]}},
    "5": {"class_type": "EmptyLatentImage", "inputs": {"width": 64, "height": 64, "batch_size": 1}},
    "10": {"class_type": "ApplyBendsFromJSON", "inputs": {"model": ["4", 0], "bends_json": json.dumps(
        {"bends": [{"path": "middle_block.1", "module_type": "rotate", "module_args": {"angle_degrees": 90},
                    "t": [1.0, 0.7]}]})}},
    "3": {"class_type": "KSampler", "inputs": {"model": ["10", 0], "seed": 7, "steps": 20, "cfg": 7.0,
                                               "sampler_name": "euler", "scheduler": "normal", "denoise": 1.0,
                                               "positive": ["6", 0], "negative": ["6", 0], "latent_image": ["5", 0]}},
    "9": {"class_type": "SaveImage", "inputs": {"images": ["8", 0], "filename_prefix": "x"}},
}


def entry(prompt: dict, image: str) -> dict:
    return {"prompt": [1, "pid", prompt, {}, []], "status": {"status_str": "success", "completed": True},
            "outputs": {"9": {"images": [{"filename": image, "subfolder": "", "type": "output"}]}}}


@pytest.fixture()
def comfy(tmp_path, monkeypatch):
    base = {k: v for k, v in PROMPT.items() if k != "10"}
    base["3"] = {**PROMPT["3"], "inputs": {**PROMPT["3"]["inputs"], "model": ["4", 0]}}
    hist = {"bent": entry(PROMPT, "bent.png"), "base": entry(base, "base.png"),
            "failed": {**entry(PROMPT, "bent.png"), "status": {"status_str": "error"}}}
    pngs = {}
    for name, color in (("bent.png", (200, 20, 20)), ("base.png", (90, 90, 90))):
        buf = io.BytesIO()
        Image.new("RGB", (64, 64), color).save(buf, "PNG")
        pngs[name] = buf.getvalue()

    class H(http.server.BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):  # noqa: N802
            if self.path.startswith("/api/history/"):
                pid = self.path.rsplit("/", 1)[1]
                body, ctype = json.dumps({pid: hist[pid]} if pid in hist else {}).encode(), "application/json"
            elif self.path.startswith("/api/view"):
                name = self.path.split("filename=")[1].split("&")[0]
                body, ctype = pngs[name], "image/png"
            elif self.path.startswith("/api/object_info"):
                body, ctype = b"{}", "application/json"
            else:
                self.send_response(404)
                self.end_headers()
                return
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.end_headers()
            self.wfile.write(body)

    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    monkeypatch.setenv("COMFYUI_URL", f"http://127.0.0.1:{srv.server_port}")
    monkeypatch.setenv("COMFY_BENDING_WORKDIR", str(tmp_path / "work"))
    import comfy_canvas
    import kb_local
    importlib.reload(comfy_canvas)
    importlib.reload(kb_local)
    yield kb_local
    srv.shutdown()


def test_log_round_keeps_facts_private_and_labels_authors(comfy):
    kl = comfy
    r = kl.log_round("bent", "s1", verdict="kept", words="a new coastline", caption="rocks turned into arches",
                     effect_tags=["new_composition", "bogus"], agent_model="claude-opus-5-5",
                     baseline_prompt_id="base")
    d = Path(r["folder"])
    rec = json.loads((d / "record.json").read_text())
    assert rec["setup"]["seed"] == 7 and "prompt" not in rec["setup"]
    assert json.loads((d / "private.json").read_text())["prompt"] == "my secret prompt"
    assert json.loads((d / "sharing.json").read_text())["share_ok"] is False
    assert rec["bends"][0]["window"] == "early" and r["unknown_tags_dropped"] == ["bogus"]
    ints = {e["kind"]: e["author"] for e in kl.kb.load_interpretations(d)}
    assert ints["verdict"]["type"] == "human" and ints["note"]["type"] == "human"
    assert ints["caption"] == ints["effect_tags"] == {"type": "ai", "model": "claude-opus-5-5",
                                                      "prompt_version": "skill-log-round-v1",
                                                      "tool": "comfyui-model-bending skill"}
    m = kl.kb.load_measurements(d)["values"]
    assert m["mae_vs_baseline"]["value"] > 50
    with pytest.raises(ValueError, match="agent_model"):
        kl.log_round("bent", "s1", caption="x")
    with pytest.raises(kl.cc.CanvasError, match="failed"):
        kl.log_round("failed", "s1")
    with pytest.raises(ValueError, match="no bends"):
        kl.log_round("base", "s1")


def test_sources_and_find_mine(comfy):
    kl = comfy
    assert kl.session_sources("s2") is None
    assert kl.session_sources("s2", ["mine"]) == ["mine"]
    kl.log_round("bent", "s2", caption="arches", effect_tags=["new_composition"], agent_model="claude-opus-5-5",
                 baseline_prompt_id="base")
    out = kl.find("new composition please", "sd15", ["mine"])
    assert out["results"] and out["results"][0]["source"] == "mine"
    assert out["results"][0]["effects"]["new_composition"]["by"] == ["ai:claude-opus-5-5"]
    assert kl.find("anything", "sd15", [])["results"] == []
    assert kl.status("s2")["mine"]["records"] == 1
