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
    """Each workflow renders a flat colour derived from it; the unbent one is grey. Video workflows (an animated WebP
    save) give 5 frames. The stub ComfyUI has every node."""
    calls = []

    def fake(wf, timeout=900):
        calls.append(wf)
        bent = any(n["class_type"] in ("ApplyBendsFromJSON", "DiT Block Bending") for n in wf.values())
        h = hashlib.sha256(json.dumps(wf, sort_keys=True).encode()).digest()
        img = Image.new("RGB", (64, 64), tuple(h[:3]) if bent else (90, 90, 90))
        if any(n["class_type"] == "SaveAnimatedWEBP" for n in wf.values()):
            return [Image.new("RGB", (64, 48), tuple((c + 9 * i) % 256 for c in img.getpixel((0, 0))))
                    for i in range(5)]
        return img

    monkeypatch.setattr(kb_run, "_render", fake)
    monkeypatch.setattr(kb_run, "_missing_nodes", lambda wf: [])
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
    with pytest.raises(kb_run.RunError, match="U-Net models .* or transformer models"):
        kb_run.init_plan("pixart", "pixart.safetensors", None)


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


# --------------------------------------------------------------------------- transformer (DiT) models
def dit_plan(arch: str, model: str, clips: list[str], **over) -> dict:
    p = kb_run.init_plan(arch, model, "Ada", vae="ae.safetensors", clips=clips, loader="unet")
    p["contributor"]["agree_cc0"] = True
    p["sampling"].update(width=64, height=64, steps=4)
    p["ops"] = {"multiply": [0.0, 0.5, 2.0], "rotate": [30, 90, 180], "add_noise": [0.2, 0.8, 1.5]}
    p.update(over)
    return p


def nodes(wf: dict, ct: str) -> list[dict]:
    return [n for n in wf.values() if n["class_type"] == ct]


def test_flux_regions_are_each_stack_in_thirds():
    p = dit_plan("flux", "flux1-dev.safetensors", ["t5xxl_fp8.safetensors", "clip_l.safetensors"])
    assert kb_run.check_plan(p) == []
    assert kb_run.regions_of(p) == ["double.early", "double.mid", "double.late",
                                    "single.early", "single.mid", "single.late"]
    assert p["layers"] == {"double.early": ["double_blocks.3"], "double.mid": ["double_blocks.9"],
                           "double.late": ["double_blocks.15"], "single.early": ["single_blocks.6"],
                           "single.mid": ["single_blocks.19"], "single.late": ["single_blocks.31"]}
    assert [kb.third_range(19, t) for t in kb.THIRDS] == [(0, 6), (7, 12), (13, 18)]
    assert [kb.third_range(38, t) for t in kb.THIRDS] == [(0, 12), (13, 25), (26, 37)]
    assert p["sampling"]["cfg"] == 1.0 and p["sampling"]["guidance"] == 3.5 and p["model"]["clip_type"] == "flux"
    assert len(kb_run.bends_of(p)) == 6 * 3 * 3
    both = {**p, "streams": ["img", "txt"]}  # spatial ops (rotate) have no grid on the text stream
    assert len(kb_run.bends_of(both)) == 6 * 3 * 3 + 6 * 2 * 3
    assert any("img is required" in e for e in kb_run.check_plan({**p, "streams": ["txt"]}))
    assert any("not a whole transformer block" in e for e in
               kb_run.check_plan({**p, "layers": {**p["layers"], "double.early": ["double_blocks.3.img_mlp"]}}))
    assert any("in region double.mid, not double.early" in e for e in
               kb_run.check_plan({**p, "layers": {**p["layers"], "double.early": ["double_blocks.9"]}}))
    schnell = kb_run.init_plan("flux", "flux1-schnell.safetensors", None, clips=["t5", "clip_l"], vae="ae")
    assert schnell["sampling"]["steps"] == 4 and "guidance" not in schnell["sampling"]


def test_a_flux_run_renders_dit_block_bends_and_passes_the_check(tmp_path, stub_render):
    out = tmp_path / "flux"
    kb_run.render(dit_plan("flux", "flux1-dev.safetensors", ["t5xxl_fp8.safetensors", "clip_l.safetensors"]), out,
                  log=lambda *a: None)
    rep = kb_run.check(out)
    assert rep["ok"], rep["problems"]
    assert rep["records"] == 6 * 3 * 3 and rep["regions"][0] == "double.early" and rep["family"] == "flux"
    assert "double.early" in kb_run.coverage_text(rep) and "image stream" in kb_run.coverage_text(rep)
    d, rec = next((d, r) for d, r in kb_run._records(out) if r["bends"][0]["path"] == "single_blocks.31")
    b = rec["bends"][0]
    assert (b["group"], b["kind"], b["node"], b["stream"], b["spatial"]) == ("single.late", "img_stream", "dit_block",
                                                                             "img", True)
    assert rec["model"]["loader"] == "unet" and rec["model"]["blocks"] == {"double_blocks": 19, "single_blocks": 38}
    wf = json.loads((d / "workflow.json").read_text(encoding="utf-8"))
    assert nodes(wf, "UNETLoader")[0]["inputs"]["unet_name"] == "flux1-dev.safetensors"
    assert nodes(wf, "DualCLIPLoader")[0]["inputs"]["type"] == "flux"
    assert nodes(wf, "FluxGuidance")[0]["inputs"]["guidance"] == 3.5
    assert nodes(wf, "EmptySD3LatentImage") and not nodes(wf, "ApplyBendsFromJSON")
    (dit,) = nodes(wf, "DiT Block Bending")
    assert dit["inputs"]["blocks"] == "single:31" and dit["inputs"]["stream"] == "img" and dit["inputs"]["strict"]
    module = wf[dit["inputs"]["bending_module"][0]]
    assert module["class_type"].endswith("Module (Bending)")
    assert json.loads(Image.open(d / "output.webp").getexif()[0x0110][7:]) == wf
    # the workflow read back gives the same record (the session log path, log_round, reads it this way)
    model, setup, bends = kb.setup_from_api_prompt(wf)
    again = kb.make_record(source="sweep", model={**model, "arch": "flux"}, setup={**setup, "prompt": rec["setup"]["prompt"]},
                           bends=bends, consent={"prompt": True})
    assert again["id"] == rec["id"]


def test_the_full_run_rule_on_a_transformer(tmp_path, stub_render):
    out = tmp_path / "flux"
    kb_run.render(dit_plan("flux", "flux1-dev.safetensors", ["t5", "clip_l"], streams=["img", "txt"]), out,
                  log=lambda *a: None)
    assert kb_run.check(out)["ok"]
    for d, rec in list(kb_run._records(out)):  # single.late keeps only its text-stream renders
        if rec["bends"][0]["group"] == "single.late" and rec["bends"][0]["stream"] == "img":
            shutil.rmtree(d)
    probs = " | ".join(kb_run.check(out)["problems"])
    assert "transformer block stacks" in probs and "missing single.late" in probs and "image stream" in probs


def test_sd3_flux2_and_wan_follow_their_own_stacks(tmp_path, stub_render):
    sd3 = dit_plan("sd3", "sd3.5_medium.safetensors", ["clip_g", "clip_l", "t5xxl"])
    assert kb_run.regions_of(sd3) == ["joint.early", "joint.mid", "joint.late"] and kb_run.check_plan(sd3) == []
    kb_run.render(sd3, tmp_path / "sd3", log=lambda *a: None)
    assert kb_run.check(tmp_path / "sd3")["ok"]
    d, rec = next(kb_run._records(tmp_path / "sd3"))
    wf = json.loads((d / "workflow.json").read_text(encoding="utf-8"))
    assert nodes(wf, "TripleCLIPLoader") and nodes(wf, "ModelSamplingSD3")[0]["inputs"]["shift"] == 3.0
    assert nodes(wf, "DiT Block Bending")[0]["inputs"]["blocks"].startswith("double:")  # joint blocks are "double"
    # Flux.2: the real counts come from the model (the catalogue), not from the usual ones
    f2 = dit_plan("flux2", "flux2-dev.safetensors", ["mistral.safetensors"])
    kb_run.read_blocks(f2, {"block_lists": {"double_blocks": {"count": 5}, "single_blocks": {"count": 20},
                                            "joint_blocks": {"count": None}}})
    assert f2["model"]["blocks"] == {"double_blocks": 5, "single_blocks": 20}
    assert f2["layers"]["double.early"] == ["double_blocks.0"] and f2["layers"]["single.late"] == ["single_blocks.16"]  # 14-19
    assert kb_run.check_plan(f2) == []
    kb_run.render(f2, tmp_path / "f2", log=lambda *a: None)
    rep = kb_run.check(tmp_path / "f2")
    assert rep["ok"], rep["problems"]
    d, rec = next(kb_run._records(tmp_path / "f2"))
    wf = json.loads((d / "workflow.json").read_text(encoding="utf-8"))
    assert nodes(wf, "EmptyFlux2LatentImage") and nodes(wf, "CLIPLoader")[0]["inputs"]["type"] == "flux2"
    # WAN: a video model; the run's pictures are animated WebPs
    wan = dit_plan("wan21", "wan2.1_t2v_1.3B_fp16.safetensors", ["umt5_xxl_fp8.safetensors"])
    wan["sampling"].update(frames=5)
    assert kb_run.regions_of(wan) == ["blocks.early", "blocks.mid", "blocks.late"]
    kb_run.render(wan, tmp_path / "wan", log=lambda *a: None)
    rep = kb_run.check(tmp_path / "wan")
    assert rep["ok"], rep["problems"]
    d, rec = next(kb_run._records(tmp_path / "wan"))
    assert rec["setup"]["route"] == "txt2video" and rec["setup"]["frames"] == 5
    im = Image.open(d / "output.webp")
    assert im.n_frames == 5 and im.getexif().get(0x0110, "").startswith("prompt:")
    wf = json.loads((d / "workflow.json").read_text(encoding="utf-8"))
    assert nodes(wf, "EmptyHunyuanLatentVideo")[0]["inputs"]["length"] == 5 and nodes(wf, "SaveAnimatedWEBP")
    assert nodes(wf, "CLIPLoader")[0]["inputs"]["type"] == "wan"
    assert "mae_vs_baseline" in kb.load_measurements(d)["values"]


def test_the_index_says_which_diagram_each_family_needs(tmp_path, stub_render):
    out = tmp_path / "flux"
    kb_run.render(dit_plan("flux", "flux1-dev.safetensors", ["t5", "clip_l"]), out, log=lambda *a: None)
    kb_run.render(small_plan(), out, log=lambda *a: None)
    meta = kb.build_index(out)
    arch = meta["architectures"]
    assert arch["sd1"]["diagram"] == "unet" and arch["sd1"]["regions"][0] == "in.hi"
    flux = arch["flux"]
    assert flux["diagram"] == "stacks" and [s["name"] for s in flux["stacks"]] == ["double_blocks", "single_blocks"]
    assert flux["stacks"][1]["regions"][2] == {"group": "single.late", "from": 26, "to": 37}
    cells = kb.read_jsonl(out / "index" / "cells.jsonl")
    c = next(c for c in cells if c["family"] == "flux" and c["op"] == "rotate")
    assert c["kind"] == "img_stream" and c["recipe"]["node"] == "dit_block"
    frag = c["recipe"]["fragment"]
    assert frag["out"] == ["bend", 0] and frag["nodes"]["bend"]["inputs"]["model"] == ["@model", 0]


def test_agents_get_a_fragment_for_a_transformer_recipe():
    import bendjson
    bend = {"path": "double_blocks.3", "module_type": "rotate", "module_args": {"angle_degrees": 30},
            "node": "dit_block", "stream": "img", "kb": {"cell": "flux|double.early|img_stream|*|rotate|slight|all|txt2img"}}
    r = bendjson.check(bend, "flux")
    assert r["ok"] and not r["errors"] and r["document"]["bends"] == []
    frag = r["dit_bends"][0]["fragment"]
    assert frag["nodes"]["bend"]["inputs"]["blocks"] == "double:3" and frag["nodes"]["bend"]["inputs"]["stream"] == "img"
    assert r["bends"][0]["group"] == "double.early" and "transformer block" in r["summary"][0].lower()
    assert r["bends"][0]["link"].startswith(kb.NAVIGATOR)
    (item,) = bendjson.open_tray(kb.tray_link([bend]), "flux")["bends"]
    assert item["ok"] and item["fragment"] == frag
