"""The first minutes of a session: the surprise draw, the intro examples, and bending a copy of the user's own run,
against a stub HTTP server standing in for ComfyUI.

    uv run --no-project --with "mcp>=2.2" --with pytest --with pillow --with numpy pytest tests
"""

from __future__ import annotations

import copy
import http.server
import json
import re
import sys
import threading
import unittest.mock as um
from urllib.parse import unquote, urlsplit

import pytest

from test_board import SCRIPTS, data, run, text

sys.path.insert(0, str(SCRIPTS))
import bendjson  # noqa: E402
import comfy_canvas as cc  # noqa: E402
import kb  # noqa: E402
import kb_local  # noqa: E402
import media  # noqa: E402
import splice  # noqa: E402
import surprise  # noqa: E402


def node(required: dict, outputs: list[str], optional: dict | None = None, names: list[str] | None = None) -> dict:
    return {"input": {"required": required, "optional": optional or {}},
            "input_order": {"required": list(required), "optional": list(optional or {})},
            "output": outputs, "output_name": names or outputs}


# what ComfyUI's /api/object_info says about the node classes these tests use
OI = {
    "CheckpointLoaderSimple": node({"ckpt_name": [["a.safetensors"]]}, ["MODEL", "CLIP", "VAE"]),
    "LoraLoader": node({"model": ["MODEL"], "clip": ["CLIP"], "lora_name": [["mine.safetensors"]],
                        "strength_model": ["FLOAT", {"default": 1.0}], "strength_clip": ["FLOAT", {"default": 1.0}]},
                       ["MODEL", "CLIP"]),
    "ModelSamplingDiscrete": node({"model": ["MODEL"], "sampling": [["eps", "lcm"]], "zsnr": ["BOOLEAN", {}]},
                                  ["MODEL"]),
    "CLIPTextEncode": node({"text": ["STRING", {"multiline": True}], "clip": ["CLIP"]}, ["CONDITIONING"]),
    "EmptyLatentImage": node({"width": ["INT", {"default": 512}], "height": ["INT", {"default": 512}],
                              "batch_size": ["INT", {"default": 1}]}, ["LATENT"]),
    "KSampler": node({"model": ["MODEL"], "seed": ["INT", {"default": 0, "control_after_generate": True}],
                      "steps": ["INT", {"default": 20}], "cfg": ["FLOAT", {"default": 8.0}],
                      "sampler_name": [["euler", "lcm"]], "scheduler": [["normal", "karras"]],
                      "positive": ["CONDITIONING"], "negative": ["CONDITIONING"], "latent_image": ["LATENT"],
                      "denoise": ["FLOAT", {"default": 1.0}]}, ["LATENT"]),
    "VAEDecode": node({"samples": ["LATENT"], "vae": ["VAE"]}, ["IMAGE"]),
    "SaveImage": node({"images": ["IMAGE"], "filename_prefix": ["STRING", {"default": "ComfyUI"}]}, []),
    "PreviewAny": node({"source": ["*"]}, []),
    "ApplyBendsFromJSON": node({"model": ["MODEL"], "bends_json": ["STRING", {"multiline": True}]},
                               ["MODEL", "STRING", "STRING"],
                               {"strict": ["BOOLEAN", {"default": False}], "clamp": [["none", "hard", "safe"]],
                                "safe_ranges": ["STRING", {"multiline": True}]},
                               ["MODEL", "report", "resolved_json"]),
    "BasicGuider": node({"model": ["MODEL"], "conditioning": ["CONDITIONING"]}, ["GUIDER"]),
    "BasicScheduler": node({"model": ["MODEL"], "scheduler": [["normal"]], "steps": ["INT", {}],
                            "denoise": ["FLOAT", {}]}, ["SIGMAS"]),
    "SamplerCustomAdvanced": node({"guider": ["GUIDER"], "sigmas": ["SIGMAS"], "latent_image": ["LATENT"]},
                                  ["LATENT", "LATENT"]),
    "Rotate Module (Bending)": node({"angle_degrees": ["FLOAT", {"default": 0.0}]}, ["BENDING_MODULE"]),
    "DiT Block Bending": node({"model": ["MODEL"], "bending_module": ["BENDING_MODULE"],
                               "blocks": ["STRING", {"default": "double:0-3"}], "stream": [["img", "txt", "both"]],
                               "spatial": ["BOOLEAN", {"default": True}]}, ["MODEL", "STRING"],
                              {"strict": ["BOOLEAN", {"default": False}]}, ["MODEL", "report"]),
}

USER = {"groups": [{"id": "mine", "title": "My project"}], "nodes": {
    "ckpt": {"class_type": "CheckpointLoaderSimple", "group": "mine", "inputs": {"ckpt_name": "dreamshaper_8.safetensors"}},
    "lora": {"class_type": "LoraLoader", "group": "mine",
             "inputs": {"model": ["ckpt", 0], "clip": ["ckpt", 1], "lora_name": "mine.safetensors",
                        "strength_model": 0.8, "strength_clip": 1.0}},
    "msd": {"class_type": "ModelSamplingDiscrete", "group": "mine",
            "inputs": {"model": ["lora", 0], "sampling": "eps", "zsnr": False}},
    "pos": {"class_type": "CLIPTextEncode", "group": "mine", "inputs": {"clip": ["lora", 1], "text": "a harbour at night"}},
    "lat": {"class_type": "EmptyLatentImage", "group": "mine", "inputs": {"width": 512, "height": 512, "batch_size": 1}},
    "ks": {"class_type": "KSampler", "group": "mine", "title": "My sampler",
           "inputs": {"model": ["msd", 0], "positive": ["pos", 0], "negative": ["pos", 0], "latent_image": ["lat", 0],
                      "seed": 1234, "steps": 30, "cfg": 7.0, "sampler_name": "euler", "scheduler": "karras",
                      "denoise": 1.0}},
    "dec": {"class_type": "VAEDecode", "group": "mine", "inputs": {"samples": ["ks", 0], "vae": ["ckpt", 2]}},
    "save": {"class_type": "SaveImage", "group": "mine", "inputs": {"images": ["dec", 0], "filename_prefix": "harbour"}},
    "note": {"class_type": "Note", "group": "mine", "inputs": {"text": "my own note"}}}}
BEND = {"path": "middle_block.1", "module_type": "rotate", "module_args": {"angle_degrees": 60}, "t": [1.0, 0.7]}


@pytest.fixture(autouse=True)
def _object_info():
    with um.patch.object(cc, "object_info", return_value=OI):
        yield


def user_run(spec: dict = USER) -> tuple[dict, dict]:
    """(API prompt, canvas graph) of a run the user made, with "randomize" on the seed as the canvas saves it."""
    ui, api = cc.build(spec)
    for n in ui["nodes"]:
        if n["type"] == "KSampler":
            n["widgets_values"][1] = "randomize"
    return api, ui


def check_graph(g: dict) -> None:
    """Every link is listed by the output it leaves and named by the input it enters; ids are unique."""
    nodes = {n["id"]: n for n in g["nodes"]}
    assert len(nodes) == len(g["nodes"])
    lids = [row[0] for row in g["links"]]
    assert len(lids) == len(set(lids))
    for lid, s, ss, d, ds, _ in g["links"]:
        assert lid in (nodes[s]["outputs"][ss]["links"] or [])
        assert nodes[d]["inputs"][ds]["link"] == lid
    for n in g["nodes"]:
        assert all(i["link"] is None or i["link"] in lids for i in n["inputs"])
        assert all(lid in lids for o in n["outputs"] for lid in o["links"] or [])
    assert g["last_node_id"] >= max(nodes) and g["last_link_id"] >= max(lids)


def by_class(api: dict, ct: str) -> list[str]:
    return [i for i, n in api.items() if n["class_type"] == ct]


# ---------------------------------------------------------------- the surprise round
@pytest.mark.parametrize("arch", ["sd15", "sd14", "sdxl"])
def test_surprise_draws_stay_in_the_safe_ranges_and_cover_the_model(arch):
    family, ranges = surprise.table(arch)
    for seed in range(40):
        d = surprise.draw(arch, seed=seed)
        cands = d["candidates"]
        assert [c["label"] for c in cands] == ["A", "B", "C"] and d["seed"] == seed
        assert {c["region"] for c in cands} == {"early", "core", "late"}  # one per part of the model
        assert [c["wild"] for c in cands] == [False, False, True]
        for c in cands:
            bend = json.loads(c["bends_json"])["bends"][0]
            assert bend == c["bend"] and kb.unet_group(bend["path"], family) == c["group"]
            assert c["what_was_bent"] and c["key"] == f"{bend['path']}|{bend['module_type']}"
            op = bend["module_type"]
            if not c["wild"]:
                arg, neutral = surprise.NEUTRAL[op]
                (lo, hi), _ = surprise._range(ranges, bend["path"], op)
                assert lo <= bend["module_args"][arg] <= hi and bend["module_args"][arg] != neutral
                assert c["clamp"] == "safe" and bend["label"] == c["label"]
            else:
                assert c["clamp"] == "none" and "wild" in bend["label"] and "wild card" in c["what_was_bent"]
                if op in surprise.NEUTRAL:  # past the limit, on the same side of "no change"
                    arg, neutral = surprise.NEUTRAL[op]
                    (lo, hi), _ = surprise._range(ranges, bend["path"], op)
                    assert not lo <= bend["module_args"][arg] <= hi or op == "rotate"
                else:
                    assert op in surprise.OFF_TABLE


def test_surprise_repeats_with_a_seed_and_avoids_earlier_draws():
    a, b = surprise.draw("sd15", seed=11), surprise.draw("sd15", seed=11)
    assert a == b and surprise.draw("sd15", seed=12) != a
    keys = [c["key"] for c in a["candidates"]]
    again = surprise.draw("sd15", seed=11, avoid=keys)
    assert not set(keys) & {c["key"] for c in again["candidates"]}
    plain = surprise.draw("sdxl", n=4, wild=False, seed=3)["candidates"]
    assert len(plain) == 4 and not any(c["wild"] for c in plain)
    assert isinstance(surprise.draw("sd15")["seed"], int)  # a seed is always reported, so a round can be repeated


def test_surprise_has_nothing_to_draw_for_models_without_a_table():
    for arch in ("flux", "sd3", "wan21", ""):
        with pytest.raises(ValueError, match="no safe-range table"):
            surprise.draw(arch)
    with pytest.raises(ValueError, match="between 1 and 6"):
        surprise.draw("sd15", n=0)


def test_surprise_skips_ranges_that_barely_leave_no_change():
    pool = surprise.options("sd15")
    early = {(o["group"], o["op"]) for o in pool["early"]}
    assert ("in.hi", "rotate") not in early and ("in.hi", "multiply") in early  # rotate 0-5° there shows nothing
    assert ("in.mid", "rotate") in early and all(o["path"] != "input_blocks.0" for o in pool["early"])


# ---------------------------------------------------------------- intro examples
@pytest.fixture
def snapshot(monkeypatch):
    """The bundled community snapshot, with no network: a record names its baseline."""
    monkeypatch.setattr(kb_local, "community_index", lambda refresh=False: (kb_local.SNAPSHOT, "bundled snapshot"))
    asked = []

    def get(url):
        asked.append(url)
        return json.dumps({"outputs": {"baseline_id": "0d80844c7d529a3cfc9e"}}).encode()

    monkeypatch.setattr(kb_local, "_get", get)
    return asked


def test_intro_examples_are_the_hand_picked_ones_that_are_published(snapshot):
    r = kb_local.intro_examples("sd15", 4)
    ex = r["examples"]
    assert [e["part"] for e in ex[:3]] == ["early", "core", "late"] and len({e["record"] for e in ex}) == 4
    assert r["family"] == "sd1" and "note" not in r and snapshot == []  # nothing fetched: the picks name their baselines
    pins = json.loads(kb_local.INTRO_PINS.read_text(encoding="utf-8"))["sd1"]
    by_record = {p["record"]: p for p in pins}
    for e in ex:
        pin = by_record[e["record"]]
        assert pin["source"] in ("paper", "author_experiment", "sweep")  # every pick's source is published
        assert media.source_kind(e["image_url"]) == "kb" and media.source_kind(e["unbent_url"]) == "kb"
        assert e["image_url"].endswith(f"/records/sd1/{pin['source']}/{e['record']}/output.webp")
        assert e["unbent_url"].endswith(f"/baselines/{pin['baseline']}.webp")
        assert e["where"] and e["how"] and e["when"] and e["caption"] and e["caption_by"].startswith("ai:")
        assert e["recipe"]["path"] and e["tested_on"] == [pin["checkpoint"]]
    core = next(p for p in pins if p["part"] == "core" and not p.get("fallback"))
    assert ex[1]["record"] == core["record"]  # the core's first choice, now that its source is published
    assert bendjson.check([e["recipe"] for e in ex], "sd15")["errors"] == []  # every recipe is a usable bend
    assert "kb_sources" in r["say"]
    assert len(kb_local.intro_examples("sd15", 6)["examples"]) == 6  # past the usable picks, the index fills in


def test_hand_picked_examples_from_a_new_source_appear_once_the_index_has_it(snapshot, monkeypatch, tmp_path):
    cells = kb.read_jsonl(kb_local.SNAPSHOT / "cells.jsonl")
    cells[0]["sources"] = {**cells[0]["sources"], "sweep": 1}  # as after the sweep's records are published
    (tmp_path / "cells.jsonl").write_text("".join(json.dumps(c) + "\n" for c in cells), encoding="utf-8")
    monkeypatch.setattr(kb_local, "community_index", lambda refresh=False: (tmp_path, "an index with the sweep"))
    ex = kb_local.intro_examples("sd15", 4)["examples"]
    pins = json.loads(kb_local.INTRO_PINS.read_text(encoding="utf-8"))["sd1"]
    core = next(p for p in pins if p["part"] == "core" and not p.get("fallback"))
    assert core["source"] == "sweep" and ex[1]["record"] == core["record"]  # the first choice, not the stand-in
    assert all(e["record"] != next(p["record"] for p in pins if p.get("fallback")) for e in ex)
    assert ex[3]["record"] == next(p["record"] for p in pins if p["record"] not in {e["record"] for e in ex[:3]}
                                   and not p.get("fallback"))


def test_intro_examples_without_picks_come_from_the_index(snapshot, monkeypatch, tmp_path):
    monkeypatch.setattr(kb_local, "INTRO_PINS", tmp_path / "none.json")
    r = kb_local.intro_examples("sd15", 4)
    ex = r["examples"]
    assert len(ex) == 4 and [e["part"] for e in ex[:3]] == ["early", "core", "late"]
    assert len({e["cell"] for e in ex}) == 4
    for e in ex:
        assert media.source_kind(e["image_url"]) == "kb" and e["image_url"].endswith("/output.webp")
        assert re.search(r"/baselines/[0-9a-f]{20}\.webp$", e["unbent_url"])  # named by the index's example_baselines
        assert e["where"] and e["how"] and e["when"] and e["evidence"] and e["recipe"]["path"] and e["tested_on"]
        assert all(t["by"] and t["means"] for t in e["looked_like"])  # interpretations name their author
    assert snapshot == []  # the index names each example's baseline: no record.json fetched


def test_intro_examples_say_when_they_are_from_another_model_type(snapshot):
    r = kb_local.intro_examples("flux", 3)
    assert "SD1.5-type" in r["note"] and len(r["examples"]) == 3 and r["family"] == "sd1"


def test_intro_examples_survive_a_missing_baseline_and_take_pinned_cells(monkeypatch, tmp_path):
    monkeypatch.setattr(kb_local, "community_index", lambda refresh=False: (kb_local.SNAPSHOT, "bundled snapshot"))

    def offline(url):
        raise OSError("offline")

    monkeypatch.setattr(kb_local, "_get", offline)
    monkeypatch.setattr(kb_local, "INTRO_PINS", tmp_path / "none.json")
    first = kb_local.intro_examples("sd15", 3)["examples"]
    # offline: the index still names each example's baseline, and nothing breaks without the network
    assert all(e["image_url"] and (e["unbent_url"] is None or e["unbent_url"].endswith(".webp")) for e in first)
    cells = [c for c in kb.read_jsonl(kb_local.SNAPSHOT / "cells.jsonl") if c["group"] == "out.hi" and c["example_paths"]]
    pins = tmp_path / "intro.json"
    pins.write_text(json.dumps({"sd1": [cells[0]["key"], "no|such|cell", {"record": "../../etc", "source": "paper",
                                                                           "group": "mid", "part": "core"}]}),
                    encoding="utf-8")
    monkeypatch.setattr(kb_local, "INTRO_PINS", pins)
    pinned = kb_local.intro_examples("sd15", 3)["examples"]
    assert [e.get("cell") for e in pinned].count(cells[0]["key"]) == 1 and len(pinned) == 3
    assert not any("etc" in e["image_url"] for e in pinned)  # an entry that is not a record id is dropped


# ---------------------------------------------------------------- a bend handed over as JSON
PASTED = '''```json
{"bends": [
  {"path": "middle_block.1", "module_type": "rotate", "module_args": {"angle_degrees": 90}, "t": [1.0, 0.7],
   "label": "recompose"},
  {"path": "output_blocks.10.1.transformer_blocks.0.attn1", "module_type": "rotate",
   "module_args": {"angle_degrees": 40, "speed": 2}},
  {"path": "input_blocks.4", "module_type": "multipy", "module_args": {"scalar": 3}}],
 "steps_min": 0, "steps_max": 3, "selected_part": "diffusion_model", "version": 1.1, "mood": "stormy"}
```'''


def test_a_pasted_bend_is_checked_like_the_node_would_and_said_in_plain_words():
    r = bendjson.check(PASTED, "sd15")
    assert r["ok"] and len(r["bends"]) == 2
    assert r["errors"] == ["bend #2 (input_blocks.4): unknown module_type 'multipy'; did you mean 'multiply'?"]
    assert any("'mood'" in p for p in r["problems"]) and any("'speed'" in p and "angle_degrees" in p for p in r["problems"])
    doc = json.loads(r["bends_json"])
    assert doc == r["document"] and [b["path"] for b in doc["bends"]] == ["middle_block.1",
                                                                        "output_blocks.10.1.transformer_blocks.0.attn1"]
    assert doc["bends"][1]["module_args"] == {"angle_degrees": 40}  # what the node ignores is left out
    assert (doc["steps_min"], doc["steps_max"], doc["selected_part"]) == (0, 3, "diffusion_model") and "mood" not in doc
    first, second = r["bends"]
    assert first["group"] == "mid" and first["inside_safe_range"] is True and first["notes"] == []
    assert r["summary"][0] == ("Recompose: the model's core (what the scene is), turned by 90°, early in the painting, "
                               "while the layout forms")
    assert second["inside_safe_range"] is False and any("0 to 5" in n for n in second["notes"])
    assert any("not image-shaped" in n and "output_blocks.10" in n for n in second["notes"])


def test_bend_json_in_its_other_shapes():
    one = {"path": "output_blocks.7", "module_type": "subset", "module_args": {"percentage": "0.3", "dim": "channel"},
           "inner": {"module_type": "add_noise", "module_args": {"noise_std": 0.8}}, "steps": "3-", "blend": 0.5,
           "guard": {"nan": "zero", "max_std_ratio": 8, "x": 1}}
    r = bendjson.check(one, "sdxl")
    (b,) = r["document"]["bends"]
    assert b["module_args"] == {"percentage": 0.3, "dim": "channel"} and b["inner"]["module_type"] == "add_noise"
    assert b["steps"] == "3-" and b["blend"] == 0.5 and b["guard"] == {"nan": "zero", "max_std_ratio": 8}
    assert r["bends"][0]["when"] == "on steps 3-" and any("guard.x" in p for p in r["problems"])
    old = bendjson.check('{"bend": [{"path": "input_blocks.8.0", "angle": 45}]}')  # the oldest clipboard format
    assert old["document"] == {"bends": [{"path": "input_blocks.8.0", "module_type": "rotate",
                                          "module_args": {"angle_degrees": 45}}]}
    assert old["bends"][0]["where"] == "the layer input_blocks.8.0" and "arch" in old["note"]  # no model type given
    video = bendjson.check([{"path": "blocks.14", "module_type": "temporal_blur", "module_args": {"sigma": 2}}], "sd15")
    assert any("video frames" in n for n in video["bends"][0]["notes"]) and any("0.3" in n for n in video["bends"][0]["notes"])
    att = bendjson.check({"bends": [], "attention_bends": [{"module_type": "rotate", "attention": "cross_text",
                                                            "module_args": {"angle_degrees": 12}}]}, "wan21")
    assert att["ok"] and att["document"]["attention_bends"] and "attention maps" in att["summary"][0]
    assert bendjson.check({"bends": [{"module_type": "rotate"}, {"path": "mid", "module_type": "subset"}, 7]})["errors"] == [
        "bend #0 has no 'path' (the layer to bend, e.g. \"middle_block.1\")",
        "bend #1 (mid): subset needs an 'inner' bend with a module_type", "bend #2 is not an object"]
    empty = bendjson.check({"bends": []})
    assert not empty["ok"] and empty["errors"] == ["there are no bends in it"]
    for bad, why in (("not json {", "not valid JSON"), ("", "no bend JSON"), ("42", "a bend document is")):
        with pytest.raises(bendjson.BendJSONError, match=why):
            bendjson.check(bad)


# ---------------------------------------------------------------- bending a copy of the user's run
def test_inspect_says_what_the_run_is_made_of():
    api, ui = user_run()
    info = splice.inspect(api, ui)
    assert info["model"] == {"checkpoint": "dreamshaper_8.safetensors", "arch": "sd15"}
    assert info["loras"] == [{"name": "mine.safetensors", "strength": 0.8}]
    assert info["setup"]["steps"] == 30 and info["setup"]["prompt"] == "a harbour at night"
    (s,) = info["samplers"]
    assert s["title"].endswith("My sampler") and "ModelSamplingDiscrete" in s["model_from"]
    assert info["can_bend"] and info["copy"] == "full" and info["notes"] == [] and info["already_bent"] == []
    assert info["bending_nodes"] == [] and "batch_size" not in info["setup"]
    assert splice.inspect(api, None)["copy"] == "pictures_only"


def test_inspect_notices_bends_and_batches_already_in_the_run():
    spec = copy.deepcopy(USER)
    spec["nodes"]["lat"]["inputs"]["batch_size"] = 4
    spec["nodes"]["turn"] = {"class_type": "Rotate Module (Bending)", "inputs": {"angle_degrees": 30}}
    spec["nodes"]["theirs"] = {"class_type": "DiT Block Bending", "title": "My own bend",
                               "inputs": {"model": ["msd", 0], "bending_module": ["turn", 0], "blocks": "double:0-3",
                                          "stream": "img", "spatial": True}}
    spec["nodes"]["ks"]["inputs"]["model"] = ["theirs", 0]
    info = splice.inspect(*user_run(spec))
    assert [t.split(" ", 1)[1] for t in info["bending_nodes"]] == ["My own bend"]  # the module node bends nothing itself
    assert info["setup"]["batch_size"] == 4
    assert any("already bends" in n for n in info["notes"]) and any("renders 4 pictures" in n for n in info["notes"])
    bent, _, _ = splice.bend_copy(*user_run(spec), BEND)  # the new bend goes after theirs
    (new,), (theirs,) = by_class(bent, "ApplyBendsFromJSON"), by_class(bent, "DiT Block Bending")
    assert bent[new]["inputs"]["model"] == [theirs, 0]


def test_the_base_preset_is_one_unbent_run_that_bend_copy_can_take():
    preset = json.loads((SCRIPTS.parent / "presets" / "base_sd15.spec.json").read_text(encoding="utf-8"))
    oi = {**OI, "CLIPLoader": node({"clip_name": [["c.safetensors"]], "type": [["stable_diffusion"]]}, ["CLIP"]),
          "VAELoader": node({"vae_name": [["v.safetensors"]]}, ["VAE"])}
    with um.patch.object(cc, "object_info", return_value=oi):
        ui, api = cc.build(preset)
        info = splice.inspect(api, ui)
        assert info["can_bend"] and info["copy"] == "full" and info["setup"]["prompt"].startswith("a lighthouse")
        bent, bent_ui, _ = splice.bend_copy(api, ui, BEND, name="surprise_A")
    assert not by_class(api, "ApplyBendsFromJSON") and len(by_class(bent, "ApplyBendsFromJSON")) == 1
    assert bent[by_class(bent, "SaveImage")[0]]["inputs"]["filename_prefix"] == "agent_bending/surprise_A"
    check_graph(bent_ui)


def test_bend_copy_keeps_everything_of_theirs_and_adds_the_bend_before_the_sampler():
    api, ui = user_run()
    before = copy.deepcopy((api, ui))
    bent, bent_ui, info = splice.bend_copy(api, ui, BEND, name="harbour_A", meta={"bent_copy_of": "pid"})
    assert (api, ui) == before  # their run is left alone
    (bend,), (ks,), (msd,), (report,) = (by_class(bent, c) for c in ("ApplyBendsFromJSON", "KSampler",
                                                                    "ModelSamplingDiscrete", "PreviewAny"))
    assert bent[ks]["inputs"]["model"] == [bend, 0] and bent[bend]["inputs"]["model"] == [msd, 0]
    assert bent[report]["inputs"]["source"] == [bend, 1]
    ins = bent[bend]["inputs"]
    assert json.loads(ins["bends_json"]) == {"bends": [BEND]} and ins["strict"] is True and ins["clamp"] == "safe"
    assert json.loads(ins["safe_ranges"])["rotate"] == {"angle_degrees": [0.0, 180.0]}  # the sd15 table
    assert {k: v for k, v in bent.items() if k in api and k != ks and v["class_type"] != "SaveImage"} == \
        {k: v for k, v in api.items() if k != ks and v["class_type"] != "SaveImage"}
    assert bent[by_class(bent, "SaveImage")[0]]["inputs"]["filename_prefix"] == "agent_bending/harbour_A"
    assert info["copy"] == "full" and info["spliced"][0]["into"] == [f"#{ks} My sampler"]

    check_graph(bent_ui)
    theirs = {n["id"]: n for n in ui["nodes"]}
    now = {n["id"]: n for n in bent_ui["nodes"]}
    assert all(now[i]["pos"] == n["pos"] and now[i]["type"] == n["type"] for i, n in theirs.items())
    assert len(now) == len(theirs) + 2 and {str(i) for i in set(now) - set(theirs)} == {bend, report}
    assert [g["title"] for g in bent_ui["groups"]] == ["My project", splice.FRAGMENT_TITLE]
    sampler = now[int(ks)]
    assert sampler["widgets_values"][:2] == [1234, "fixed"]  # the copy repeats their seed
    assert next(n for n in bent_ui["nodes"] if n["type"] == "SaveImage")["widgets_values"] == ["agent_bending/harbour_A"]
    assert bent_ui["extra"]["agent"] == {"bent_copy_of": "pid", "bent_copy": True}
    link = {row[0]: row for row in bent_ui["links"]}[sampler["inputs"][0]["link"]]
    assert str(link[1]) == bend and link[5] == "MODEL"
    box = bent_ui["groups"][-1]["bounding"]
    for n in ui["nodes"]:  # the new group sits clear of their nodes
        x, y = n["pos"]
        w, h = n["size"]
        assert not (box[0] < x + w and x < box[0] + box[2] and box[1] < y + h and y < box[1] + box[3])


def test_bend_copy_handles_shared_and_separate_models():
    two = copy.deepcopy(USER)
    two["nodes"]["ks2"] = {**copy.deepcopy(USER["nodes"]["ks"]), "title": "High-res pass"}
    two["nodes"]["ks2"]["inputs"]["latent_image"] = ["ks", 0]
    api, ui = user_run(two)
    bent, bent_ui, info = splice.bend_copy(api, ui, [BEND])
    (bend,) = by_class(bent, "ApplyBendsFromJSON")  # one bend feeds both samplers
    assert all(bent[k]["inputs"]["model"] == [bend, 0] for k in by_class(bent, "KSampler"))
    assert len(info["spliced"]) == 1 and len(info["spliced"][0]["into"]) == 2
    check_graph(bent_ui)

    bent, bent_ui, info = splice.bend_copy(api, ui, [BEND], only=["High-res pass"])
    first, second = sorted(by_class(bent, "KSampler"), key=int)
    assert bent[first]["inputs"]["model"] == api[first]["inputs"]["model"]
    assert bent[bent[second]["inputs"]["model"][0]]["class_type"] == "ApplyBendsFromJSON"
    check_graph(bent_ui)
    with pytest.raises(cc.CanvasError, match="samplers:"):
        splice.bend_copy(api, ui, [BEND], only=["nope"])

    two["nodes"]["ks2"]["inputs"]["model"] = ["ckpt", 0]  # a second sampler on another model
    api, ui = user_run(two)
    assert any("different models" in n for n in splice.inspect(api, ui)["notes"])
    bent, bent_ui, info = splice.bend_copy(api, ui, {"bends": [BEND]})
    assert len(by_class(bent, "ApplyBendsFromJSON")) == 2 and len(info["spliced"]) == 2
    assert len(bent_ui["nodes"]) == len(ui["nodes"]) + 4
    check_graph(bent_ui)


def test_bend_copy_finds_guiders_and_leaves_schedulers_alone():
    spec = copy.deepcopy(USER)
    n = spec["nodes"]
    del n["ks"]
    n["guider"] = {"class_type": "BasicGuider", "inputs": {"model": ["msd", 0], "conditioning": ["pos", 0]}}
    n["sched"] = {"class_type": "BasicScheduler", "inputs": {"model": ["msd", 0], "scheduler": "normal", "steps": 20,
                                                             "denoise": 1.0}}
    n["ks"] = {"class_type": "SamplerCustomAdvanced", "inputs": {"guider": ["guider", 0], "sigmas": ["sched", 0],
                                                                 "latent_image": ["lat", 0]}}
    api, ui = user_run(spec)
    assert [s["class"] for s in splice.inspect(api, ui)["samplers"]] == ["BasicGuider"]
    bent, bent_ui, _ = splice.bend_copy(api, ui, BEND)
    (bend,), (guider,), (sched,) = (by_class(bent, c) for c in ("ApplyBendsFromJSON", "BasicGuider", "BasicScheduler"))
    assert bent[guider]["inputs"]["model"] == [bend, 0] and bent[sched]["inputs"]["model"] == api[sched]["inputs"]["model"]
    check_graph(bent_ui)


def test_runs_without_a_usable_canvas_graph_give_pictures_only():
    api, ui = user_run()
    bent, bent_ui, info = splice.bend_copy(api, None, BEND)  # queued through the API: no graph at all
    assert bent_ui is None and info["copy"] == "pictures_only" and len(by_class(bent, "ApplyBendsFromJSON")) == 1
    assert "Apply Bends from JSON box" in info["notes"][-1] and "My sampler" in info["notes"][-1]

    ks = by_class(api, "KSampler")[0]
    nested = {(f"57:{k}" if k == ks else k): v for k, v in api.items()}  # the sampler sits inside a subgraph
    ui_outer = {**ui, "nodes": [n for n in ui["nodes"] if str(n["id"]) != ks]}
    info = splice.inspect(nested, ui_outer)
    assert info["copy"] == "pictures_only" and any("subgraph" in n for n in info["notes"])
    bent, bent_ui, info = splice.bend_copy(nested, ui_outer, BEND)
    assert bent_ui is None and bent[f"57:{ks}"]["inputs"]["model"][0] in by_class(bent, "ApplyBendsFromJSON")

    del api[ks]
    assert splice.inspect(api, ui)["can_bend"] is False
    with pytest.raises(cc.CanvasError, match="nothing to bend"):
        splice.bend_copy(api, ui, BEND)


def test_models_without_a_table_and_custom_fragments():
    spec = copy.deepcopy(USER)
    spec["nodes"]["ckpt"]["inputs"]["ckpt_name"] = "flux1-dev.safetensors"
    api, ui = user_run(spec)
    assert any("no safe-range table" in n for n in splice.inspect(api, ui)["notes"])
    bent, _, info = splice.bend_copy(api, ui, BEND)
    ins = bent[by_class(bent, "ApplyBendsFromJSON")[0]]["inputs"]
    assert ins["clamp"] == "hard" and "safe_ranges" not in ins and any("gentle" in n for n in info["notes"])
    wild, _, _ = splice.bend_copy(api, ui, BEND, clamp="none")
    assert wild[by_class(wild, "ApplyBendsFromJSON")[0]]["inputs"]["clamp"] == "none"

    fragment = {"nodes": {"turn": {"class_type": "Rotate Module (Bending)", "inputs": {"angle_degrees": 12}},
                          "bend": {"class_type": "DiT Block Bending",
                                   "inputs": {"model": ["@model", 0], "bending_module": ["turn", 0],
                                              "blocks": "double:0-6", "stream": "img", "spatial": True}},
                          "report": {"class_type": "PreviewAny", "inputs": {"source": ["bend", 1]}}},
                "out": ["bend", 0]}
    bent, bent_ui, _ = splice.bend_copy(api, ui, fragment=fragment)
    (dit,), (ks,), (turn,), (msd,) = (by_class(bent, c) for c in ("DiT Block Bending", "KSampler",
                                                                 "Rotate Module (Bending)", "ModelSamplingDiscrete"))
    assert bent[ks]["inputs"]["model"] == [dit, 0] and bent[dit]["inputs"]["model"] == [msd, 0]
    assert bent[dit]["inputs"]["bending_module"] == [turn, 0] and not by_class(bent, "ApplyBendsFromJSON")
    check_graph(bent_ui)
    with pytest.raises(cc.CanvasError, match="not both"):
        splice.bend_copy(api, ui, BEND, fragment=fragment)
    with pytest.raises(cc.CanvasError, match="@model"):
        splice.bend_copy(api, ui, fragment={"nodes": {"p": {"class_type": "PreviewAny", "inputs": {}}}, "out": ["p", 0]})
    with pytest.raises(cc.CanvasError, match="there are no bends in it"):
        splice.bend_copy(api, ui, {"bends": []})
    with pytest.raises(cc.CanvasError, match="no usable bends given"):
        splice.bend_copy(api, ui)


def test_bend_copy_takes_a_pasted_bend_and_refuses_one_the_node_would_refuse():
    api, ui = user_run()
    with pytest.raises(cc.CanvasError, match="did you mean 'multiply'"):
        splice.bend_copy(api, ui, PASTED)  # nothing is rendered with a bend silently dropped
    fixed = PASTED.replace("multipy", "multiply")
    bent, bent_ui, info = splice.bend_copy(api, ui, fixed, clamp="none")
    doc = json.loads(bent[by_class(bent, "ApplyBendsFromJSON")[0]]["inputs"]["bends_json"])
    assert len(doc["bends"]) == 3 and doc["steps_max"] == 3 and "mood" not in doc
    assert len(info["bends"]) == 3 and info["bends"][0].startswith("Recompose: the model's core")
    assert any("'speed'" in n for n in info["notes"]) and any("not image-shaped" in n for n in info["notes"])
    assert any("may fall apart" in n for n in info["notes"])  # with clamp none the amount is used as given
    safe = splice.bend_copy(api, ui, fixed)[2]
    assert not any("may fall apart" in n for n in safe["notes"])  # clamp safe pulls it back, so nothing to warn of
    check_graph(bent_ui)


def test_replace_bends_puts_the_bend_in_place_of_the_runs_own():
    spec = copy.deepcopy(USER)
    spec["nodes"]["turn"] = {"class_type": "Rotate Module (Bending)", "inputs": {"angle_degrees": 30}}
    spec["nodes"]["theirs"] = {"class_type": "DiT Block Bending", "title": "My own bend",
                               "inputs": {"model": ["msd", 0], "bending_module": ["turn", 0], "blocks": "double:0-3",
                                          "stream": "img", "spatial": True}}
    spec["nodes"]["ks"]["inputs"]["model"] = ["theirs", 0]
    api, ui = user_run(spec)
    assert any("replace_bends" in n for n in splice.inspect(api, ui)["notes"])
    (ks,), (msd,), (theirs,) = (by_class(api, c) for c in ("KSampler", "ModelSamplingDiscrete", "DiT Block Bending"))
    bent, bent_ui, info = splice.bend_copy(api, ui, BEND, replace_bends=True)
    (new,) = by_class(bent, "ApplyBendsFromJSON")
    assert bent[ks]["inputs"]["model"] == [new, 0] and bent[new]["inputs"]["model"] == [msd, 0]  # theirs is left out
    assert bent[theirs] == api[theirs] and info["copy"] == "full"
    check_graph(bent_ui)
    row = {r[0]: r for r in bent_ui["links"]}[next(n for n in bent_ui["nodes"] if str(n["id"]) == new)["inputs"][0]["link"]]
    assert str(row[1]) == msd

    plain, plain_ui, info = splice.bend_copy(api, ui, replace_bends=True)  # their run without its bends
    assert plain[ks]["inputs"]["model"] == [msd, 0] and not by_class(plain, "ApplyBendsFromJSON")
    assert plain_ui is None and info["copy"] == "pictures_only" and any("unbent picture" in n for n in info["notes"])
    _, _, info = splice.bend_copy(*user_run(), BEND, replace_bends=True)  # nothing to replace: said, and bent as usual
    assert any("no bending nodes" in n for n in info["notes"])


def entry(number: int, api: dict, ui: dict | None, client: str) -> dict:
    extra = {"client_id": client, **({"extra_pnginfo": {"workflow": ui}} if ui else {})}
    return {"prompt": [number, f"run{number}", api, extra, ["9"]],
            "status": {"status_str": "success", "completed": True, "messages": []},
            "outputs": {"9": {"images": [{"filename": f"harbour_{number}.png", "subfolder": "", "type": "output"}]}}}


def test_the_newest_own_run_skips_runs_this_skill_queued(monkeypatch):
    api, ui = user_run()
    history = [("run3", entry(3, api, None, splice.AGENT_CLIENT)), ("run2", entry(2, api, ui, "the-browser-tab")),
               ("run1", entry(1, api, ui, "the-browser-tab"))]
    monkeypatch.setattr(cc, "_history", lambda n: history[:n])
    pid, e = splice.load_run()
    assert pid == "run2" and splice.run_graphs(e) == (api, ui)
    monkeypatch.setattr(cc, "_history", lambda n: history[:1])
    with pytest.raises(cc.CanvasError, match="run their workflow once"):
        splice.load_run()


# ---------------------------------------------------------------- through the MCP server
@pytest.fixture(scope="module")
def comfy():
    """Answers /api/object_info and /api/history, stores /api/prompt and /api/userdata posts."""
    with um.patch.object(cc, "object_info", return_value=OI):
        api, ui = user_run()
    history = {"run1": entry(1, api, ui, "the-browser-tab")}
    posted = {"prompts": [], "userdata": {}}

    class Handler(http.server.BaseHTTPRequestHandler):
        def _json(self, obj, code=200):
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):  # noqa: N802
            path = urlsplit(self.path).path
            if path == "/api/object_info":
                self._json(OI)
            elif path == "/api/history":
                self._json(history)
            elif path.startswith("/api/history/"):
                pid = path.rsplit("/", 1)[1]
                self._json({pid: history[pid]} if pid in history else {})
            else:
                self.send_error(404)

        def do_POST(self):  # noqa: N802
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            path = urlsplit(self.path).path
            if path == "/api/prompt":
                pid = f"run{len(history) + 1}"
                posted["prompts"].append(body)
                history[pid] = entry(len(history) + 1, body["prompt"], (body.get("extra_data") or {})
                                     .get("extra_pnginfo", {}).get("workflow"), body["client_id"])
                self._json({"prompt_id": pid, "node_errors": {}})
            elif path.startswith("/api/userdata/"):
                posted["userdata"][unquote(path.split("/api/userdata/", 1)[1])] = body
                self._json({})
            else:
                self.send_error(404)

        def log_message(self, *a):
            pass

    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}", posted
    srv.shutdown()


def test_a_surprise_round_on_a_copy_of_their_run_through_the_tools(comfy, tmp_path):
    base, posted = comfy

    async def body(s):
        tools = {t.name for t in (await s.list_tools()).tools}
        seen = await s.call_tool("inspect_run", {})
        drawn = data(await s.call_tool("surprise_bends", {"arch": "sd15", "seed": 5}))
        read = data(await s.call_tool("check_bends", {"bends": PASTED, "arch": "sd15"}))
        refused = await s.call_tool("bend_run", {"name": "pasted", "bends": PASTED})
        pasted = data(await s.call_tool("bend_run", {"name": "pasted", "bends": read["bends_json"]}))
        made, ran = [], []
        for c in drawn["candidates"]:
            name = f"harbour_r1_{c['label']}"
            made.append(data(await s.call_tool("bend_run", {"name": name, "bends": json.loads(c["bends_json"]),
                                                            "clamp": c["clamp"], "session": "harbour-1004"})))
            ran.append(data(await s.call_tool("run_workflow", {"name": name, "max_wait": 5})))
        proposed = data(await s.call_tool("propose_workflow", {"name": "harbour_r1_A", "message": "your copy"}))
        again = data(await s.call_tool("inspect_run", {}))  # their own run is still the newest *own* run
        nothing = await s.call_tool("surprise_bends", {"arch": "flux"})
        return tools, seen, drawn, made, ran, proposed, again, nothing, read, refused, pasted

    tools, seen, drawn, made, ran, proposed, again, nothing, read, refused, pasted = run(base, tmp_path, body)
    assert {"inspect_run", "bend_run", "surprise_bends", "intro_examples", "check_bends"} <= tools
    assert read["summary"][0].startswith("Recompose:") and len(read["errors"]) == 1
    assert refused.is_error and "did you mean 'multiply'" in text(refused)
    assert pasted["copy"] == "full" and len(pasted["bends"]) == 2  # the tidied document: the usable bends
    seen = data(seen)
    assert seen["prompt_id"] == "run1" and seen["copy"] == "full" and seen["loras"][0]["name"] == "mine.safetensors"
    assert seen["images"][0]["url"].startswith(f"{base}/api/view?filename=harbour_1.png")
    assert all(m["from_run"] == "run1" and m["copy"] == "full" for m in made)
    assert [r["status"] for r in ran] == ["success"] * 3 and again["prompt_id"] == "run1"
    clamps = [next(n for n in p["prompt"].values() if n["class_type"] == "ApplyBendsFromJSON")["inputs"]["clamp"]
              for p in posted["prompts"]]
    assert clamps == ["safe", "safe", "none"]  # the wild card is not pulled back into the safe range
    for p in posted["prompts"]:  # every bent picture carries the bent canvas graph
        check_graph(p["extra_data"]["extra_pnginfo"]["workflow"])
        assert p["extra_data"]["extra_pnginfo"]["workflow"]["extra"]["agent"]["session"] == "harbour-1004"
    assert proposed["saved_as"] == "workflows/agent_bending/harbour_r1_A.json"
    assert list(posted["userdata"]) == ["workflows/agent_bending/harbour_r1_A.json"]  # never over their own workflow
    assert nothing.is_error and "no safe-range table" in text(nothing)


def test_a_pictures_only_copy_runs_but_cannot_be_proposed(comfy, tmp_path):
    base, posted = comfy
    n = len(posted["prompts"])
    # what bend_run stores when the run carries no canvas graph: the bent prompt and a marker instead of a graph
    bent, bent_ui, _ = splice.bend_copy(user_run()[0], None, BEND, name="nested")
    assert bent_ui is None
    built = tmp_path / "workflows"
    built.mkdir()
    (built / "nested.api.json").write_text(json.dumps(bent), encoding="utf-8")
    (built / "nested.ui.json").write_text(json.dumps({"nodes": [], "links": [], "extra": {"agent": {
        "pictures_only": True}}}), encoding="utf-8")

    async def body(s):
        ran = data(await s.call_tool("run_workflow", {"name": "nested", "max_wait": 5}))
        refused = await s.call_tool("propose_workflow", {"name": "nested", "message": "x"})
        return ran, refused

    ran, refused = run(base, tmp_path, body)
    assert ran["status"] == "success" and "extra_data" not in posted["prompts"][n]  # no empty graph in the picture
    assert refused.is_error and "pictures-only" in text(refused) and "by hand" in text(refused)
