"""The bend knowledge base library (scripts/kb.py): records, privacy, validation, cells, ranking, prompt parsing.

    uv run --no-project --with pytest pytest tests/test_kb.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "skills" / "comfyui-model-bending" / "scripts"))
import kb  # noqa: E402

MODEL = {"arch": "sd15", "checkpoint": "realisticVisionV51_v51VAE.safetensors"}
SETUP = {"route": "txt2img", "seed": 42, "sampler": "dpmpp_2m", "scheduler": "karras", "steps": 20, "cfg": 7.0,
         "width": 512, "height": 512, "prompt": "Analog style portrait of a person", "negative": "blurry"}


def rec(seed=42, prompt="Analog style portrait of a person", op="multiply", args=None, path="input_blocks.4.0.skip_connection",
        consent=True, **kw):
    s = {**SETUP, "seed": seed, "prompt": prompt}
    return kb.make_record(source="paper", model=MODEL, setup=s,
                          bends=[{"path": path, "op": op, "args": args or {"scalar": 0}}],
                          consent={"prompt": consent}, **kw)


def test_ids_are_stable_and_depend_on_facts_only():
    a, b = rec(), rec(created="2020-01-01T00:00:00Z")
    assert a["id"] == b["id"]
    assert rec(seed=1)["id"] != a["id"]
    assert rec(args={"scalar": 2})["id"] != a["id"]


def test_prompt_and_input_dropped_without_consent():
    r = kb.make_record(source="session", model=MODEL, setup={**SETUP, "input_image": "me.png"},
                       bends=[{"path": "middle_block.1", "op": "rotate", "args": {"angle_degrees": 90}}],
                       consent={"prompt": False, "input_image": False}, salt="secret")
    assert "prompt" not in r["setup"] and "negative" not in r["setup"] and "input_image" not in r["setup"]
    assert r["setup"]["prompt_key"] != kb.private_key(SETUP["prompt"], "")  # salted
    assert kb.validate(r, "record") == []
    shared = rec(consent=True)
    assert shared["setup"]["prompt"] == SETUP["prompt"]
    assert shared["setup"]["prompt_key"] == kb.private_key(SETUP["prompt"], "")


def test_validation_catches_missing_fields_and_unconsented_prompt():
    r = rec(consent=False)
    r["setup"]["prompt"] = "leaked"
    errs = kb.validate(r, "record")
    assert any("without consent" in e for e in errs)
    bad = dict(rec())
    bad.pop("model")
    assert any("missing 'model'" in e for e in kb.validate(bad, "record"))


def test_ai_interpretations_must_name_their_model():
    ok = kb.make_interpretation({"record": "x"}, "caption", "a face dissolving", kb.ai("claude-opus-5-5", "kb-caption-v1"))
    assert kb.validate(ok, "interpretation") == []
    bad = kb.make_interpretation({"record": "x"}, "caption", "a face", {"type": "ai"})
    assert any("must name its model" in e for e in kb.validate(bad, "interpretation"))
    assert kb.validate(kb.make_interpretation({"record": "x"}, "note", "lovely", kb.human("Ahmed")), "interpretation") == []


@pytest.mark.parametrize("path,family,group,kind", [
    ("input_blocks.4.0.skip_connection", "sd1", "in.mid", "skip"),
    ("input_blocks.1.0.in_layers.2", "sd1", "in.hi", "res_branch"),
    ("middle_block.1.transformer_blocks.0.attn2.to_out.0", "sd1", "mid", "cross_attn"),
    ("output_blocks.4.1.transformer_blocks.0.norm1", "sd1", "out.lo", "norm"),
    ("output_blocks.10.1", "sd1", "out.hi", "attn_block"),
    ("input_blocks.3.0.op", "sd1", "in.hi", "downsample"),
    ("input_blocks.3.0", "sd1", "in.hi", "downsample"),
    ("middle_block.1", "sd1", "mid", "attn_block"),
    ("time_embed.2", "sd1", "embed", "time_embed"),
    ("output_blocks.7.0", "sdxl", "out.hi", "res_block"),
    ("input_blocks.5.1.proj_in", "sd1", "in.mid", "proj"),
])
def test_where_a_path_sits(path, family, group, kind):
    assert kb.unet_group(path, family) == group
    assert kb.unet_kind(path, family) == kind


def test_buckets_and_windows():
    assert kb.amount_bucket("multiply", {"scalar": 0}) == "zero"
    assert kb.amount_bucket("multiply", {"scalar": 1.5}) == "boost"
    assert kb.amount_bucket("rotate", {"angle_degrees": 270}) == "quarter"
    assert kb.amount_bucket("add_noise", {"noise_std": 2}) == "extreme"
    assert kb.is_neutral("multiply", {"scalar": 1}) and kb.is_neutral("rotate", {"angle_degrees": 360})
    assert kb.window_of(*kb.step_fraction({"steps_min": 0, "steps_max": 4}, 20)) == "early"
    assert kb.window_of(*kb.step_fraction({"steps_min": 14, "steps_max": None}, 20)) == "late"
    assert kb.window_of(*kb.step_fraction({"steps_min": 10, "steps_max": None}, 20)) == "mid_to_end"
    assert kb.window_of(*kb.step_fraction({"t": [1.0, 0.7]}, None)) == "early"
    assert kb.window_of(*kb.step_fraction({"steps": "5-"}, 20)) == "mid_to_end"
    assert kb.window_of(*kb.step_fraction({}, 20)) == "all"


def _kb_with(tmp_path: Path, records: list[tuple[dict, list[str]]], findings=()) -> dict:
    for r, tags in records:
        ints = [kb.make_interpretation({"record": r["id"]}, "effect_tags", tags, kb.ai("claude-opus-5-5", "v1"))]
        kb.save_record(tmp_path, r, {"output.webp": b"img"}, kb.make_measurements(r["id"], {"lpips_distance": 0.3}), ints)
    for f in findings:
        (tmp_path / "findings").mkdir(exist_ok=True)
        (tmp_path / "findings" / f"{f['id']}.json").write_text(json.dumps(f), encoding="utf-8")
    return kb.build_index(tmp_path)


def test_index_cells_grades_and_ranking(tmp_path):
    rot = lambda seed, prompt: kb.make_record(  # noqa: E731
        source="author_experiment", model={"arch": "sd14", "checkpoint": "sd-v1-4.ckpt"},
        setup={**SETUP, "seed": seed, "prompt": prompt}, consent={"prompt": True},
        bends=[{"path": "middle_block.1", "op": "rotate", "args": {"angle_degrees": 90}}])
    finding = kb.make_finding("Rotating the core recomposes the scene.", scope={"family": ["sd1"], "op": ["rotate"]},
                              author=kb.human("Abuzuraiq & Pasquier"), citation={"arxiv": "2607.22428"})
    out = _kb_with(tmp_path, [(rot(1, "a"), ["abstract", "new_composition"]), (rot(2, "b"), ["abstract"]),
                              (rec(), ["washed_out"])], [finding])
    assert out["errors"] == [] and out["records"] == 3 and out["cells"] == 2
    cells = kb.read_jsonl(tmp_path / "index" / "cells.jsonl")
    c = next(c for c in cells if c["op"] == "rotate")
    assert c["grade"] == "replicated" and c["study_backed"] and c["effect_tags"]["abstract"]["count"] == 2
    assert c["effect_tags"]["abstract"]["by"] == ["ai:claude-opus-5-5"]
    assert c["recipe"] == {"path": "middle_block.1", "module_type": "rotate", "module_args": {"angle_degrees": 90}}
    res = kb.rank_cells(cells, "make it more abstract", arch="sd14",
                        findings=kb.read_jsonl(tmp_path / "index" / "findings.jsonl"))
    assert res["effect_tags"] == ["abstract"]
    assert [r["op"] for r in res["results"]] == ["rotate"]
    assert res["results"][0]["findings"][0]["author"]["type"] == "human"
    res15 = kb.rank_cells(cells, "more abstract", arch="sd15")
    assert res15["note"] and "sd15" in res15["note"]
    assert res["note"] is None  # sd14 records exist for the sd14 query


def test_record_json_never_rewritten_and_interpretations_dedupe(tmp_path):
    r = rec()
    i = kb.make_interpretation({"record": r["id"]}, "caption", "x", kb.human("A"))
    d = kb.save_record(tmp_path, r, {}, None, [i])
    kb.save_record(tmp_path, {**r, "license": "CC-BY-4.0"}, {}, None, [i])
    assert json.loads((d / "record.json").read_text())["license"] == "CC0-1.0"
    assert len(kb.load_interpretations(d)) == 1


def test_goal_tags_from_plain_words():
    assert set(kb.goal_tags("more abstract but keep the subject")) == {"abstract", "subject_kept"}
    assert kb.goal_tags("posterized, flat color") == ["poster"]


def test_setup_from_api_prompt():
    prompt = {
        "4": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "v1-5-pruned-emaonly.safetensors"}},
        "6": {"class_type": "CLIPTextEncode", "inputs": {"text": "a lighthouse", "clip": ["4", 1]}},
        "7": {"class_type": "CLIPTextEncode", "inputs": {"text": "blurry", "clip": ["4", 1]}},
        "5": {"class_type": "EmptyLatentImage", "inputs": {"width": 512, "height": 512, "batch_size": 1}},
        "10": {"class_type": "ApplyBendsFromJSON", "inputs": {"model": ["4", 0], "bends_json": json.dumps({"bends": [
            {"path": "middle_block.1", "module_type": "rotate", "module_args": {"angle_degrees": "{{a}}"}, "t": [1.0, 0.7]}]})}},
        "3": {"class_type": "KSampler", "inputs": {"model": ["10", 0], "seed": 7, "steps": 20, "cfg": 7.5,
                                                   "sampler_name": "euler", "scheduler": "normal", "denoise": 1.0,
                                                   "positive": ["6", 0], "negative": ["7", 0], "latent_image": ["5", 0]}},
        "11": {"class_type": "Rotate Module (Bending)", "inputs": {"angle_degrees": 45}},
        "12": {"class_type": "Timestep Gated Bending", "inputs": {"bending_module": ["11", 0], "t_start": 0.5, "t_end": 0.0}},
        "13": {"class_type": "Model Bending", "inputs": {"model": ["4", 0], "bending_module": ["12", 0],
                                                         "path": "output_blocks.3.1, output_blocks.4.1"}},
    }
    resolved = json.dumps({"bends": [{"path": "middle_block.1", "module_type": "rotate",
                                      "module_args": {"angle_degrees": 90}, "t": [1.0, 0.7]}]})
    model, setup, bends = kb.setup_from_api_prompt(prompt, resolved)
    assert model == {"checkpoint": "v1-5-pruned-emaonly.safetensors", "arch": "sd15"}
    assert setup == {"seed": 7, "steps": 20, "cfg": 7.5, "sampler": "euler", "scheduler": "normal",
                     "prompt": "a lighthouse", "negative": "blurry", "width": 512, "height": 512, "route": "txt2img"}
    assert bends[0]["args"] == {"angle_degrees": 90}
    assert [b["path"] for b in bends[1:]] == ["output_blocks.3.1", "output_blocks.4.1"]
    assert bends[1]["t"] == [0.5, 0.0] and bends[1]["op"] == "rotate"
    r = kb.make_record(source="session", model=model, setup=setup, bends=bends, salt="s")
    assert kb.validate(r, "record") == [] and r["bends"][0]["window"] == "early" and r["bends"][1]["window"] == "mid_to_end"
