"""Full runs for the community knowledge base (scripts/kb_run.py): plan, render (against a stub), check, describe,
submit (dry run), and the schema's contribution rules.

    uv run --no-project --with pytest --with pillow --with numpy pytest tests/test_kb_run.py
"""

from __future__ import annotations

import hashlib
import io
import json
import shutil
import sys
from pathlib import Path

import pytest
from PIL import Image

SCRIPTS = Path(__file__).resolve().parents[1] / "skills" / "comfyui-model-bending" / "scripts"
sys.path.insert(0, str(SCRIPTS))

import kb  # noqa: E402
import kb_run  # noqa: E402


def small_plan(**over) -> dict:
    p = kb_run.init_plan("sd15", "v1-5-pruned-emaonly.safetensors", "Ada")
    p["contributor"]["agree_cc0"] = True
    p["sampling"].update(width=64, height=64, steps=6)
    p["ops"] = {"multiply": [0.0, 0.5, 2.0], "rotate": [30, 90, 180], "add_noise": [0.2, 0.8, 1.5]}
    p.update(over)
    return p


@pytest.fixture()
def stub_render(monkeypatch):
    """Each workflow renders a flat colour derived from it; the unbent one is grey."""
    calls = []

    def fake(wf, timeout=900):
        calls.append(wf)
        bent = any(n["class_type"] == "ApplyBendsFromJSON" for n in wf.values())
        h = hashlib.sha256(json.dumps(wf, sort_keys=True).encode()).digest()
        return Image.new("RGB", (64, 64), tuple(h[:3]) if bent else (90, 90, 90))

    monkeypatch.setattr(kb_run, "_render", fake)
    return calls


@pytest.fixture()
def run(tmp_path, stub_render) -> Path:
    out = tmp_path / "run"
    kb_run.render(small_plan(), out, log=lambda *a: None)
    return out


def test_the_default_plan_is_a_full_run_and_needs_cc0(tmp_path, stub_render):
    p = kb_run.init_plan("sd15", "x.safetensors", None)
    assert kb_run.check_plan(p) == [] and len(kb_run.bends_of(p)) == 7 * 4 * 3
    with pytest.raises(kb_run.RunError, match="agree_cc0"):
        kb_run.render(p, tmp_path / "r", log=lambda *a: None)
    assert any("missing in.hi" in e for e in kb_run.check_plan({**p, "layers": {**p["layers"], "in.hi": []}}))
    assert any("at least 3 ops" in e for e in kb_run.check_plan({**p, "ops": {"multiply": [0, 1, 2], "rotate": [1, 2, 3]}}))
    assert any("not middle_block" in e or "region" in e for e in
               kb_run.check_plan({**p, "layers": {**p["layers"], "in.hi": ["middle_block.1"]}}))
    with pytest.raises(kb_run.RunError, match="U-Net"):
        kb_run.init_plan("flux", "flux.safetensors", None)


def test_render_writes_checked_records_with_workflows_and_resumes(run, stub_render, tmp_path):
    rep = kb_run.check(run)
    assert rep["ok"], rep["problems"]
    assert rep["records"] == 7 * 3 * 3 and rep["broken"] == 0
    assert set(rep["coverage"]) == {"multiply", "rotate", "add_noise"}
    d, rec = next(kb_run._records(run))
    assert rec["source"] == "sweep" and rec["consent"]["prompt"] and rec["setup"]["negative"] == ""
    assert rec["contribution"]["by"] == {"type": "human", "name": "Ada"} and rec["contribution"]["via"] == "script"
    assert rec["contribution"]["agreed"]["license"] == "CC0-1.0" and rec["license"] == "CC0-1.0"
    assert rec["provenance"]["producer"] == "kb_run.py" and rec["outputs"]["workflow"] == "workflow.json"
    wf = json.loads((d / "workflow.json").read_text(encoding="utf-8"))
    bend = next(n for n in wf.values() if n["class_type"] == "ApplyBendsFromJSON")["inputs"]
    assert bend["strict"] is True and bend["clamp"] == "none"
    exif = Image.open(d / "output.webp").getexif().get(0x0110)
    assert exif.startswith("prompt:") and json.loads(exif[7:]) == wf  # the picture opens in ComfyUI
    assert Image.open(d / "output.webp").info.get("lossless", True)
    vals = kb.load_measurements(d)["values"]
    assert {"mae_vs_baseline", "std", "hf_ratio", "pixel_flags", "degenerate"} <= set(vals)
    n = len(stub_render)
    kb_run.render(small_plan(), run, log=lambda *a: None)  # resumes: nothing rendered again
    assert len(stub_render) == n


def test_check_names_what_stops_a_review(run):
    d, rec = next(kb_run._records(run))
    (d / "workflow.json").unlink()
    rec2 = dict(rec, setup={k: v for k, v in rec["setup"].items() if k != "negative"})
    (d / "record.json").write_text(json.dumps(rec2), encoding="utf-8")
    probs = " | ".join(kb_run.check(run)["problems"])
    assert "workflow.json is missing" in probs and "negative is missing" in probs and "id does not match" in probs


def test_check_enforces_the_full_run_rule(run):
    for d, rec in list(kb_run._records(run)):  # drop a region and an op
        if rec["bends"][0]["group"] == "out.hi" or rec["bends"][0]["op"] == "add_noise":
            shutil.rmtree(d)
    probs = " | ".join(kb_run.check(run)["problems"])
    assert "missing out.hi" in probs and "at least 3 ops" in probs


def test_failed_renders_are_logged_not_saved(tmp_path, monkeypatch, stub_render):
    real = kb_run._render

    def flaky(wf, timeout=900):
        if any("output_blocks.10.1" in json.dumps(n["inputs"]) for n in wf.values() if n["class_type"] == "ApplyBendsFromJSON"):
            raise kb_run.RunError("ApplyBendsFromJSON: path not found")
        return real(wf, timeout)

    monkeypatch.setattr(kb_run, "_render", flaky)
    out = tmp_path / "run"
    n = kb_run.render(small_plan(), out, log=lambda *a: None)
    assert n["failed"] == 9 and (out / "failures.jsonl").exists()
    assert "missing out.hi" in " | ".join(kb_run.check(out)["problems"])


def test_descriptions_are_saved_as_ai_interpretations_naming_the_model(run):
    r = kb_run.sheets(run)
    index = json.loads((run / "sheets" / "index.json").read_text(encoding="utf-8"))
    assert r["pairs"] == 7 * 3 * 3 and len(index) == r["pairs"]
    key = next(iter(index))
    (run / "d.json").write_text(json.dumps({key: {"caption": "a red field", "change": "it turns red",
                                                  "keywords": ["Red", " "], "effect_tags": ["colour_shift", "nope"]},
                                            "sheet_999:Z": {"caption": "x"}}), encoding="utf-8")
    with pytest.raises(kb_run.RunError, match="--model"):
        kb_run.add_descriptions(run, run / "d.json", "")
    out = kb_run.add_descriptions(run, run / "d.json", "claude-opus-5-5")
    assert out == {"described": 1, "unknown": ["sheet_999:Z"]}
    d = next(d for d, rec in kb_run._records(run) if rec["id"] == index[key])
    ints = {e["kind"]: e for e in kb.load_interpretations(d)}
    assert ints["effect_tags"]["value"] == ["colour_shift"] and ints["keywords"]["value"] == ["red"]
    assert ints["caption"]["author"] == {"type": "ai", "model": "claude-opus-5-5", "prompt_version": "kb-caption-v1",
                                         "tool": "kb_run.py"}


def test_submit_dry_run_sends_only_records_and_baselines(run):
    (run / "notes.txt").write_text("private", encoding="utf-8")
    r = kb_run.submit(run, dry_run=True)
    assert r["would_upload"] > 0 and all(f.startswith(("records/", "baselines/")) for f in r["files"])
    assert "Full run" in r["title"]


def test_the_schema_takes_runs_not_single_hand_made_shares():
    rec = small_plan() and kb.make_record(source="sweep", model={"checkpoint": "x", "arch": "sd15"},
                                          setup={"route": "txt2img", "seed": 1, "sampler": "euler",
                                                 "scheduler": "normal", "steps": 6, "cfg": 7.0, "width": 64,
                                                 "height": 64, "prompt": "p", "negative": ""},
                                          bends=[{"path": "middle_block.1", "op": "rotate",
                                                  "args": {"angle_degrees": 90}}], consent={"prompt": True})
    rec["contribution"] = {"by": {"type": "human", "name": None}, "via": "script", "run": "r",
                           "agreed": {"license": "CC0-1.0", "at": "now"}}
    assert kb.validate(rec, "record") == []
    assert kb.validate({**rec, "contribution": {**rec["contribution"], "via": "hand"}}, "record")
    assert kb.validate({**rec, "source": "artist"}, "record")
    assert "artist" not in kb.SOURCES
