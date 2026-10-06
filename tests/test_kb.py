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


def test_broken_renders_leave_statistics_but_describe_the_risk(tmp_path):
    def rot(seed, lp, deg, ret, broken, change):
        r = kb.make_record(source="sweep", model={"arch": "sd15", "checkpoint": "rv.safetensors"},
                           setup={**SETUP, "seed": seed, "prompt": "p"}, consent={"prompt": True},
                           bends=[{"path": "output_blocks.10.1", "op": "rotate", "args": {"angle_degrees": 90}}])
        m = kb.make_measurements(r["id"], {"lpips_distance": lp, "clip_degenerate": deg, "prompt_retention": ret,
                                           "degenerate": kb.measure("degenerate", broken,
                                                                    reasons=["clip_degenerate"] if broken else [])})
        a = kb.ai("claude-opus-5-5", "v1")
        ints = [kb.make_interpretation({"record": r["id"]}, "effect_tags", ["degenerate"] if broken else ["abstract"], a),
                kb.make_interpretation({"record": r["id"]}, "change", change, a)]
        kb.save_record(tmp_path, r, {"output.webp": b"img"}, m, ints)

    rot(1, 0.4, 0.1, 0.95, False, "the scene turns into a new coastline")
    rot(2, 0.5, 0.7, 0.70, False, "the subject fades into blobs")
    rot(3, 0.9, 0.95, 0.5, True, "dissolves into grey static")
    kb.build_index(tmp_path)
    c = kb.read_jsonl(tmp_path / "index" / "cells.jsonl")[0]
    assert c["n"] == 3 and c["n_intact"] == 2 and c["degenerate_rate"] == 0.333
    assert c["measurements"]["lpips_distance"]["mean"] == 0.45  # the broken render's 0.9 is left out
    assert "degenerate" not in c["effect_tags"] and c["effect_tags"]["abstract"]["count"] == 2
    assert c["broken_reasons"] == {"clip_degenerate": 1}
    assert c["signals"]["subject_fades"] == 0.667 and c["signals"]["noise_or_blob_look"] == 0.667
    assert [n["value"] for n in c["failure_notes"]] == ["dissolves into grey static", "the subject fades into blobs"]
    assert c["failure_notes"][0]["broken"] and c["failure_notes"][0]["author"]["model"] == "claude-opus-5-5"
    assert len(c["broken_examples"]) == 1 and len(c["examples"]) == 2
    r = kb.rank_cells([c], "more abstract")["results"][0]
    assert r["risks"]["summary"].startswith("33% of renders broke (clip_degenerate)")


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


def test_degenerate_rule_is_strict():
    assert kb.degenerate_reasons([]) == []
    # smooth blobs, flat textures and lost subjects were often judged fine: recorded, not broken
    assert kb.degenerate_reasons(["blob", "flat", "clipped"], prompt_retention=0.5, clip_degenerate=0.3) == []
    assert kb.degenerate_reasons(["noise"]) == ["noise"]
    assert kb.degenerate_reasons(["extreme", "blob"]) == ["extreme"]
    assert kb.degenerate_reasons([], prompt_retention=0.8, clip_degenerate=0.92) == ["clip_degenerate"]
    assert kb.degenerate_reasons([], prompt_retention=0.9, clip_degenerate=0.95) == []  # subject kept
    assert kb.degenerate_reasons([], prompt_retention=0.6, clip_degenerate=0.89) == []  # below the mass cut
    assert kb.degenerate_reasons([], clip_degenerate=0.93) == []                         # no prompt: stricter cut
    assert kb.degenerate_reasons([], clip_degenerate=0.96) == ["clip_degenerate"]
    assert "calibrated" in kb.METRICS["degenerate"]["method"]


def test_ids_are_frozen_to_a_fixed_field_list():
    r = rec()
    assert r["id_scheme"] == kb.ID_SCHEME
    # new fields outside the frozen list (loaders, notes...) never change a published id
    same = kb.record_id({**r["model"], "clip": "c.safetensors", "vae": "v.safetensors"}, {**r["setup"], "note": "x"},
                        r["bends"])
    assert same == r["id"]
    # a fact inside the list does
    assert kb.record_id(r["model"], {**r["setup"], "negative": ""}, r["bends"]) != r["id"]
    assert kb.baseline_id(r) == kb.record_id(r["model"], r["setup"], [])


def test_transformer_facts_join_the_id_only_when_present():
    r = rec()
    # scheme-1 ids hash these bend keys with None when absent: the newer keys must not be hashed at all then
    old = kb.digest(kb._id_value({"model": {k: r["model"].get(k) for k in kb.ID_MODEL_KEYS},
                                  "setup": {k: r["setup"][k] for k in kb.ID_SETUP_KEYS if k in r["setup"]},
                                  "bends": [{k: b.get(k) for k in kb.ID_BEND_KEYS} for b in r["bends"]]}))
    assert old == r["id"]
    dit = [{**r["bends"][0], "node": "dit_block", "stream": "img", "spatial": True}]
    assert kb.record_id(r["model"], r["setup"], dit) != r["id"]
    assert kb.record_id(r["model"], {**r["setup"], "guidance": 3.5}, r["bends"]) != r["id"]


@pytest.mark.parametrize("path,family,blocks,group,kind", [
    ("double_blocks.3", "flux", None, "double.early", "block"),
    ("double_blocks.7.img_attn.qkv", "flux", None, "double.mid", "img_attn"),
    ("single_blocks.37.linear2", "flux", None, "single.late", "linear2"),
    ("double:13-18", "flux", None, "double.late", "block"),
    ("single:0-6", "flux", None, "single.early", "block"),
    ("joint_blocks.20.context_block.mlp.fc1", "sd3", None, "joint.late", "ctx_mlp"),
    ("double:12", "sd3", {"joint_blocks": 38}, "joint.early", "block"),  # SD3.5 Large: 38 joint blocks
    ("blocks.15.cross_attn.q", "wan", None, "blocks.mid", "cross_attn"),
    ("img_in", "flux", None, "embed", "embed"),
    ("final_layer.linear", "flux", None, "out", "out"),
])
def test_where_a_transformer_path_sits(path, family, blocks, group, kind):
    assert kb.dit_group(path, family, blocks) == group
    assert kb.dit_kind(path) == kind
    assert kb.dit_kind(path, "txt") == "txt_stream"


def test_architectures_and_regions():
    assert kb.regions("sd1") == list(kb.UNET_REGIONS) and kb.architecture("sdxl")["diagram"] == "unet"
    assert kb.regions("sd3") == ["joint.early", "joint.mid", "joint.late"]
    assert kb.regions("flux2", {"double_blocks": 8, "single_blocks": 48})[-1] == "single.late"
    a = kb.architecture("wan", {"blocks": 40})
    assert a["stacks"] == [{"name": "blocks", "label": "blocks", "count": 40, "selector": "double", "regions": [
        {"group": "blocks.early", "from": 0, "to": 13}, {"group": "blocks.mid", "from": 14, "to": 26},
        {"group": "blocks.late", "from": 27, "to": 39}]}]
    assert kb.dit_selector("joint_blocks.7", "sd3") == "double:7" and kb.dit_selector("single_blocks.2", "flux") == "single:2"
    assert kb.dit_selector("double_blocks.2.img_mlp", "flux") is None
    assert kb.guess_arch("flux2-dev.safetensors") == "flux2" and kb.guess_arch("flux1-schnell.safetensors") == "flux"


def test_comfy_workflow_renders_the_record():
    r = kb.make_record(source="sweep", model={"checkpoint": "LCM.safetensors", "arch": "sd15", "clip": "clip.st",
                                              "vae": "vae.st", "model_sampling": "lcm"},
                       setup={**SETUP, "sampler": "lcm", "steps": 6, "cfg": 1.5}, consent={"prompt": True},
                       bends=[{"path": "middle_block.1", "op": "rotate", "args": {"angle_degrees": 90},
                               "steps_min": 0, "steps_max": 1}])
    wf = kb.comfy_workflow(r)
    types = {n["class_type"] for n in wf.values()}
    assert {"CLIPLoader", "VAELoader", "ModelSamplingDiscrete", "ApplyBendsFromJSON", "KSampler"} <= types
    bend = next(n for n in wf.values() if n["class_type"] == "ApplyBendsFromJSON")["inputs"]
    doc = json.loads(bend["bends_json"])
    assert bend["clamp"] == "none" and doc["steps_min"] == 0 and doc["steps_max"] == 1
    assert doc["bends"] == [{"path": "middle_block.1", "module_type": "rotate", "module_args": {"angle_degrees": 90}}]
    ks = next(n for n in wf.values() if n["class_type"] == "KSampler")["inputs"]
    assert (ks["seed"], ks["steps"], ks["cfg"], ks["sampler_name"]) == (42, 6, 1.5, "lcm")
    assert ks["control_after_generate"] == "fixed"
    texts = [n["inputs"]["text"] for n in wf.values() if n["class_type"] == "CLIPTextEncode"]
    assert texts == ["Analog style portrait of a person", "blurry"]
    # private prompt -> placeholder; no bends (a baseline) -> no bend node
    private = kb.comfy_workflow(rec(consent=False))
    assert kb.PROMPT_PLACEHOLDER in [n["inputs"].get("text") for n in private.values()]
    assert "ApplyBendsFromJSON" not in {n["class_type"] for n in kb.comfy_workflow({**r, "bends": []}).values()}


def test_id_aliases_chain_across_migrations(tmp_path):
    (tmp_path / "migrations").mkdir()
    (tmp_path / "migrations" / "id_scheme_1.json").write_text(json.dumps({"records": {"a": "b"}, "baselines": {}}))
    (tmp_path / "migrations" / "id_scheme_2.json").write_text(json.dumps({"records": {"b": "c"}, "baselines": {}}))
    al = kb.id_aliases(tmp_path)
    assert kb.resolve_id("a", al) == "c" and kb.resolve_id("b", al) == "c" and kb.resolve_id("z", al) == "z"


def test_ids_normalise_numbers():
    r = rec()
    floaty = {**r["setup"], "cfg": 7, "steps": 20.0}
    assert kb.record_id(r["model"], floaty, r["bends"]) == kb.record_id(r["model"], {**r["setup"], "cfg": 7.0}, r["bends"])
    b = [{**r["bends"][0], "args": {"scalar": 0.0}}]
    assert kb.record_id(r["model"], r["setup"], b) == kb.record_id(r["model"], r["setup"], [{**b[0], "args": {"scalar": 0}}])
    assert kb.record_id(r["model"], {**r["setup"], "cfg": 7.5}, r["bends"]) != kb.record_id(r["model"], floaty, r["bends"])


def test_cells_sum_community_likes(tmp_path):
    r1, r2 = rec(seed=1), rec(seed=2)
    for r in (r1, r2):
        kb.save_record(tmp_path, r, {"output.webp": b"img"}, kb.make_measurements(r["id"], {"lpips_distance": 0.3}))
    kb.build_index(tmp_path)
    assert "likes" not in kb.read_jsonl(tmp_path / "index" / "cells.jsonl")[0]  # no community file: no field
    (tmp_path / "community").mkdir()
    (tmp_path / "community" / "likes.jsonl").write_text(
        json.dumps({"record": r1["id"], "likes": 3}) + "\n" + json.dumps({"record": r2["id"], "likes": 2}) + "\n")
    kb.build_index(tmp_path)
    assert kb.read_jsonl(tmp_path / "index" / "cells.jsonl")[0]["likes"] == 5
