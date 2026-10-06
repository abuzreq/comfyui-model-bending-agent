#!/usr/bin/env python3
"""The user's own bend knowledge base and the community one, as the skill uses them.

- log_round: turn a finished ComfyUI run into a record in the user's own knowledge base (the work folder's kb/). The
  prompt and input image are kept in a private sidecar (private.json) that is never shared; the record itself follows
  the community format. Nothing is uploaded: the community base takes full runs (kb_run.py), not single rounds.
- find: rank cells for a goal from the community knowledge base, the user's own, or both.
- community: the snapshot bundled with the skill (data/kb/community/), or the latest index from the Hugging Face
  dataset, fetched without a token and cached in the work folder.

Script use (Claude Code):
  python kb_local.py log <prompt_id> --session S [--verdict kept] [--words "..."] [--baseline <prompt_id>]
  python kb_local.py find "more abstract" --arch sd15 --sources community,mine
  python kb_local.py intro --arch sd15          a few community examples to show what bending can do
  python kb_local.py status
"""

from __future__ import annotations

import io
import json
import os
import re
import shutil
import sys
import urllib.parse
import urllib.request
from pathlib import Path

import comfy_canvas as cc
import kb

WORK = cc.work_dir()
OWN = WORK / "kb"
CACHE = WORK / "kb_community"
SESSIONS = WORK / "sessions"
SNAPSHOT = kb.DATA / "community"
DATASET = os.environ.get("BEND_KB_DATASET", "abuzreq/model-bending-knowledge-base")
REVISION = os.environ.get("BEND_KB_REVISION", "main")  # a branch, tag or commit of the dataset; a commit pins it
HF = f"https://huggingface.co/datasets/{DATASET}/resolve/main"
INDEX_FILES = ("cells.jsonl", "findings.jsonl", "meta.json")
MAX_INDEX_BYTES = 64 * 2**20
MAX_TEXT = 2000  # longer free text from the community index is cut
_HIDDEN = re.compile("[\x00-\x08\x0b\x0c\x0e-\x1f\x7f\u200b-\u200f\u202a-\u202e\u2060-\u206f\ufeff]")
SESSION_RE = re.compile(r"[A-Za-z0-9][\w.-]{0,63}")
SOURCES = ("community", "mine")


# --------------------------------------------------------------------------- logging a round
def _history(pid: str) -> dict:
    h = cc._req("GET", f"/api/history/{pid}")
    if pid not in h:
        raise cc.CanvasError(f"no finished run {pid!r} in ComfyUI's history")
    if (h[pid].get("status") or {}).get("status_str") == "error":
        raise cc.CanvasError(f"run {pid!r} failed in ComfyUI; only finished runs can be recorded")
    return h[pid]


def _resolved_json(entry: dict) -> str | None:
    """Apply Bends from JSON's resolved_json, if a PreviewAny / report sink showed it."""
    for out in (entry.get("outputs") or {}).values():
        for t in out.get("text") or []:
            try:
                d = json.loads(t)
            except (TypeError, json.JSONDecodeError):
                continue
            if isinstance(d, dict) and isinstance(d.get("bends"), list):
                return t
    return None


def _first_image_url(pid: str, entry: dict) -> str | None:
    imgs = cc.describe_run(pid, entry)["images"]
    return imgs[0]["url"] if imgs else None


def _webp(data: bytes, max_side: int = 768) -> bytes:
    from PIL import Image
    im = Image.open(io.BytesIO(data)).convert("RGB")
    im.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    im.save(buf, "WEBP", quality=85)
    return buf.getvalue()


def log_round(prompt_id: str, session: str, verdict: str = "", words: str = "", caption: str = "",
              change: str = "", effect_tags: list[str] | None = None, agent_model: str = "",
              baseline_prompt_id: str = "", arch: str = "", contributor: str = "",
              keywords: list[str] | None = None, prompt_version: str = "") -> dict:
    """Record a finished bent run in the user's own knowledge base. verdict/words are the user's (human);
    caption/change/keywords/effect_tags are the agent's (AI: agent_model names the model and is then required).
    prompt_version = kb.CAPTION_PROMPT_VERSION when the agent wrote them with the knowledge base's description prompt
    (description_prompt), so they can be compared with the community's captions."""
    if (caption or change or effect_tags or keywords) and not agent_model:
        raise ValueError("agent_model is required when the agent adds a caption, change, keywords or effect tags")
    if prompt_version and prompt_version != kb.CAPTION_PROMPT_VERSION:
        raise ValueError(f"prompt_version is {kb.CAPTION_PROMPT_VERSION!r} (the description prompt) or empty")
    _session_file(session)  # the same rule for session ids everywhere
    entry = _history(prompt_id)
    model, setup, bends = kb.setup_from_api_prompt(entry["prompt"][2], _resolved_json(entry))
    if arch:
        model["arch"] = arch
    if not bends:
        raise ValueError("this run has no bends (an unbent run is a baseline: pass it as baseline_prompt_id)")
    private = {k: setup.get(k) for k in ("prompt", "negative", "input_image") if setup.get(k)}
    rec = kb.make_record(source="session", model=model, setup=setup, bends=bends,
                         consent={"prompt": False, "input_image": False}, salt=kb.local_salt(OWN / "salt"),
                         contributor=contributor or None,
                         provenance={"producer": "comfyui-model-bending skill", "session": session,
                                     "prompt_id": prompt_id})
    url = _first_image_url(prompt_id, entry)
    if not url:
        raise ValueError("the run produced no image")
    out_bytes = cc.fetch_bytes(url)
    measurements = {}
    if baseline_prompt_id:
        import metrics
        b_entry = _history(baseline_prompt_id)
        b_url = _first_image_url(baseline_prompt_id, b_entry)
        if b_url:
            bid = kb.baseline_id(rec)
            kb.save_baseline(OWN, bid, _webp(cc.fetch_bytes(b_url)))
            rec["outputs"]["baseline_id"] = bid
            row = metrics.compare(b_url, [url])[1]
            # the community base's strict rule: pixel flags are recorded, only noise / extreme frames count as broken
            # here (no CLIP in the skill, so the CLIP part of the rule is left to the dataset's own check)
            flags = metrics.kb_flags(row)
            reasons = kb.degenerate_reasons(flags)
            measurements = {"mae_vs_baseline": row["mae"], "std": row["std"], "hf_ratio": row["hf_ratio"],
                            "pixel_flags": flags,
                            "degenerate": kb.measure("degenerate", bool(reasons), reasons=reasons)}
    ints = []
    target = {"record": rec["id"]}
    who = kb.human(contributor or "the user")
    if verdict:
        ints.append(kb.make_interpretation(target, "verdict", verdict, who, "the user's choice in the session"))
    if words:
        ints.append(kb.make_interpretation(target, "note", words, who, "the user's own words"))
    agent = kb.ai(agent_model, prompt_version or "skill-log-round-v1", "comfyui-model-bending skill") \
        if agent_model else None
    # the session's agent knows the recipe, unlike the maintainer's captioner: the basis says so
    basis = ("baseline|output pair, described in the session with the description prompt (recipe known)"
             if prompt_version else "output (and baseline, when given)")
    vocab = kb.load_vocab()["tags"]
    words_list = [str(k).strip().lower() for k in (keywords or []) if str(k).strip()][:8]
    for kind, val in (("caption", caption), ("change", change), ("keywords", words_list),
                      ("effect_tags", [t for t in (effect_tags or []) if t in vocab])):
        if val:
            ints.append(kb.make_interpretation(target, kind, val, agent, basis))
    d = kb.save_record(OWN, rec, {"output.webp": _webp(out_bytes)},
                       kb.make_measurements(rec["id"], measurements) if measurements else None, ints)
    if private:
        (d / "private.json").write_text(json.dumps(private, indent=1, ensure_ascii=False), encoding="utf-8")
    skipped = [t for t in (effect_tags or []) if t not in vocab]
    return {"record": rec["id"], "folder": str(d), "cell": kb.cell_key(kb.cell_fields(rec)) if kb.cell_fields(rec)
            else None, "bends": [{k: b.get(k) for k in ("path", "op", "args", "window", "bucket")} for b in rec["bends"]],
            "interpretations": len(ints), "shared": False,
            **({"unknown_tags_dropped": skipped} if skipped else {})}


def logged_prompt_ids() -> set[str]:
    """The ComfyUI runs already recorded in the user's own knowledge base."""
    ids = set()
    for p in OWN.rglob("record.json"):
        try:
            pid = (json.loads(p.read_text(encoding="utf-8")).get("provenance") or {}).get("prompt_id")
        except (OSError, json.JSONDecodeError):
            continue
        if pid:
            ids.add(pid)
    return ids


# --------------------------------------------------------------------------- sources
def _session_file(session: str) -> Path:
    """The file for a session id; ids are letters, digits, '-', '_' and '.', so they can never leave the folder."""
    if not SESSION_RE.fullmatch(session or "") or ".." in session:
        raise ValueError(f"unusable session id {session!r}: use up to 64 letters, digits, '-', '_' or '.', e.g. "
                         f"lighthouse-0929")
    return SESSIONS / f"{session}.json"


def session_sources(session: str, sources: list[str] | None = None) -> list[str] | None:
    """Read (or, with sources, store) which knowledge the user chose for a session: community, mine, both, none."""
    p = _session_file(session)
    data = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
    if sources is not None:
        bad = [s for s in sources if s not in SOURCES]
        if bad:
            raise ValueError(f"unknown source(s) {bad}; use community, mine, or an empty list for neither")
        data.update(kb_sources=list(sources), decided=kb.now())
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(data, indent=1), encoding="utf-8")
    return data.get("kb_sources")


def clean_text(v):
    """Community text as plain data: control and invisible characters removed, long strings cut."""
    if isinstance(v, str):
        s = _HIDDEN.sub("", v)
        return s if len(s) <= MAX_TEXT else s[:MAX_TEXT] + "…"
    if isinstance(v, list):
        return [clean_text(x) for x in v]
    if isinstance(v, dict):
        return {clean_text(k) if isinstance(k, str) else k: clean_text(x) for k, x in v.items()}
    return v


def _get(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=30) as r:
        data = r.read(MAX_INDEX_BYTES + 1)
    if len(data) > MAX_INDEX_BYTES:
        raise ValueError(f"{url} is larger than {MAX_INDEX_BYTES} bytes")
    return data


def _revision_commit() -> str:
    """The commit of BEND_KB_REVISION (default main) on the dataset, so all index files come from one version."""
    if not re.fullmatch(r"[\w.-]+/[\w.-]+", DATASET) or not re.fullmatch(r"[\w./-]{1,100}", REVISION):
        raise ValueError("BEND_KB_DATASET must be owner/name and BEND_KB_REVISION a branch, tag or commit")
    if re.fullmatch(r"[0-9a-f]{40}", REVISION):
        return REVISION
    info = json.loads(_get(f"https://huggingface.co/api/datasets/{DATASET}/revision/"
                           f"{urllib.parse.quote(REVISION, safe='')}"))
    sha = str(info.get("sha") or "")
    if not re.fullmatch(r"[0-9a-f]{40}", sha):
        raise ValueError("the dataset did not report a commit for that revision")
    return sha


def refresh_community() -> dict | None:
    """Fetch the community index at one commit, check that every line parses, clean its text, and replace the cached
    copy in one step. Returns the source record, or None (offline, or the download did not check out): the previous
    copy, or the bundled snapshot, then keeps serving."""
    try:
        sha = _revision_commit()
        base = f"https://huggingface.co/datasets/{DATASET}/resolve/{sha}/index"
        files = {}
        for name in INDEX_FILES:
            text = _get(f"{base}/{name}").decode("utf-8")
            if name.endswith(".jsonl"):
                rows = [json.loads(ln) for ln in text.splitlines() if ln.strip()]
                if not all(isinstance(r, dict) for r in rows):
                    raise ValueError(f"{name}: not a list of objects")
                files[name] = "".join(json.dumps(clean_text(r), ensure_ascii=False) + "\n" for r in rows)
            else:
                meta = json.loads(text)
                if not isinstance(meta, dict):
                    raise ValueError(f"{name}: not an object")
                files[name] = json.dumps(clean_text(meta), indent=1, ensure_ascii=False)
    except Exception:  # noqa: BLE001 - offline, not published, or malformed: keep what we have
        return None
    source = {"dataset": DATASET, "revision": REVISION, "commit": sha, "fetched": kb.now()}
    new = CACHE.with_name(CACHE.name + ".new")
    shutil.rmtree(new, ignore_errors=True)
    new.mkdir(parents=True)
    for name, text in files.items():
        (new / name).write_text(text, encoding="utf-8")
    (new / "source.json").write_text(json.dumps(source, indent=1), encoding="utf-8")
    shutil.rmtree(CACHE, ignore_errors=True)
    new.rename(CACHE)
    return source


def _meta(folder: Path) -> dict:
    p = folder / "meta.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def _source(folder: Path) -> dict:
    p = folder / "source.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def community_index(refresh: bool = False) -> tuple[Path, str]:
    """Folder holding cells.jsonl / findings.jsonl / meta.json, and where it came from: the latest index from the
    dataset (refresh, or a newer cached copy), else the snapshot bundled with the skill. Only copies fetched by
    refresh_community (checked and cleaned, with source.json) are used."""
    if refresh and (src := refresh_community()):
        return CACHE, f"latest from huggingface.co/datasets/{DATASET} (commit {src['commit'][:10]})"
    src = _source(CACHE)
    if src and (CACHE / "cells.jsonl").exists() and _meta(CACHE).get("built", "") >= _meta(SNAPSHOT).get("built", ""):
        return CACHE, (f"cached from huggingface.co/datasets/{DATASET} ({_meta(CACHE).get('built')}, "
                       f"commit {src.get('commit', '')[:10]})")
    return SNAPSHOT, f"snapshot bundled with the skill ({_meta(SNAPSHOT).get('built', 'empty')})"


def own_index() -> Path:
    """The user's own knowledge base, re-aggregated (cheap: it is small)."""
    if not (OWN / "records").exists():
        return OWN / "index"
    kb.build_index(OWN)
    return OWN / "index"


def find(goal: str, arch: str = "", sources: list[str] | None = None, limit: int = 6, refresh: bool = False) -> dict:
    sources = list(sources or [])
    if not sources:
        return {"sources": [], "results": [], "note": "the user chose not to use stored knowledge; explore fresh"}
    out = {"goal": goal, "sources": {}, "results": []}
    for src in sources:
        if src == "community":
            idx, where = community_index(refresh)
        else:
            idx, where = own_index(), f"your own records in {OWN}"
        cells, findings = kb.read_jsonl(idx / "cells.jsonl"), kb.read_jsonl(idx / "findings.jsonl")
        res = kb.rank_cells(cells, goal, arch=arch or None, findings=findings, limit=limit)
        out["sources"][src] = {"from": where, "cells": len(cells), "considered": res["considered"],
                               "note": res.get("note"), "general_findings": res.get("general_findings", [])}
        out["effect_tags"] = res["effect_tags"]
        for r in res["results"]:
            r["source"] = src
            # the recipe says where it came from, so a render of it can be traced back (the node ignores "kb")
            r["recipe"] = {**r["recipe"], "kb": {"dataset": DATASET if src == "community" else "local",
                                                 "cell": r["cell"]}}
            if src == "community":
                r["example_urls"] = [example_url(cells, r["cell"], rid) for rid in r["examples"]]
                if DATASET == kb.COMMUNITY_DATASET:
                    r["link"] = kb.cell_link(r["cell"])
                    r["example_links"] = [kb.record_link(rid) for rid in r["examples"]]
        out["results"] += res["results"]
    out["results"].sort(key=lambda r: -r["score"])
    out["results"] = out["results"][:limit]
    if not out["results"]:
        out["note"] = ("nothing matched; the knowledge base has no records for this goal on this model family yet. "
                       "Explore with the general heuristics and log what works (log_round).")
    return out


def example_url(cells: list[dict], cell_key: str, rid: str) -> str | None:
    """A public image URL for a community example record."""
    c = next((c for c in cells if c["key"] == cell_key), None)
    path = ((c or {}).get("example_paths") or {}).get(rid)
    return f"{HF}/{path}/output.webp" if path else None


def describe_cell(cell_key: str, sources: list[str] | None = None) -> dict:
    for src in sources or list(SOURCES):
        idx = community_index()[0] if src == "community" else own_index()
        c = next((c for c in kb.read_jsonl(idx / "cells.jsonl") if c["key"] == cell_key), None)
        if c:
            return {"source": src, **c}
    raise ValueError(f"no cell {cell_key!r} in {sources or list(SOURCES)}")


# --------------------------------------------------------------------------- examples for a first session
INTRO_PINS = kb.DATA / "intro.json"  # hand-picked examples per model family; the file's note explains the format
_GRADE_POINTS = {"replicated": 3, "multi-prompt": 2, "multi-seed": 1, "anecdotal": 0}
_GRADE_WORDS = {"replicated": "seen on several prompts and starting points", "multi-prompt": "seen on several prompts",
                "multi-seed": "seen on several starting points", "anecdotal": "seen once"}
_AMOUNT_WORDS = {
    "rotate": {"slight": "turned slightly", "quarter": "turned about a quarter turn", "half": "turned upside down"},
    "multiply": {"zero": "switched off", "damp": "weakened", "boost": "strengthened", "strong": "strengthened a lot",
                 "negative": "inverted"},
    "add_noise": {"light": "shaken up lightly", "medium": "shaken up", "heavy": "shaken up strongly",
                  "extreme": "shaken up very strongly"},
    "scale": {"shrink": "zoomed out", "grow": "zoomed in", "strong_grow": "zoomed in a lot"},
    "add_scalar": {"low": "shifted a little", "mid": "shifted", "high": "shifted a lot"},
}
_WINDOW_WORDS = {"all": "through the whole painting", "early": "early in the painting", "middle": "in the middle of "
                 "the painting", "late": "late in the painting"}


def _intro_score(c: dict) -> float:
    tags = c.get("effect_tags") or {}
    share = lambda t: (tags.get(t) or {}).get("share", 0)  # noqa: E731
    return (_GRADE_POINTS.get(c.get("grade"), 0) + 0.5 * bool(c.get("study_backed"))
            + 0.3 * min(c.get("n_described") or 0, 3) - 4 * (c.get("degenerate_rate") or 0)
            - 2 * share("degenerate") - share("subtle"))


def _baseline_url(c: dict, rid: str) -> str | None:
    """The unbent picture an example was compared with. Indexes built before example_baselines existed do not
    name it: then the example's own record does."""
    bid = (c.get("example_baselines") or {}).get(rid)
    path = (c.get("example_paths") or {}).get(rid)
    if not bid and path:
        try:
            bid = (json.loads(_get(f"{HF}/{path}/record.json")).get("outputs") or {}).get("baseline_id")
        except Exception:  # noqa: BLE001 - offline, or the record is not published: show the example alone
            bid = None
    return f"{HF}/baselines/{bid}.webp" if isinstance(bid, str) and re.fullmatch(r"[0-9a-f]{8,64}", bid) else None


def _how(op: str, bucket: str) -> str:
    return _AMOUNT_WORDS.get(op, {}).get(bucket, f"{op} ({bucket})")


def _cell_example(c: dict, pool: list[dict], vocab: dict, plain_group: dict, part: str) -> dict:
    rid = c["examples"][0]
    cap = ((c.get("example_captions") or {}).get(rid) or {}).get("caption") or {}
    tags = [{"tag": t, "means": (vocab.get(t) or {}).get("description"), "by": v.get("by")}
            for t, v in list((c.get("effect_tags") or {}).items())[:3]]
    return {"cell": c["key"], "part": part, "where": plain_group[c["group"]], "how": _how(c["op"], c["bucket"]),
            "when": _WINDOW_WORDS.get(c.get("window"), c.get("window")),
            "looked_like": tags, "keywords": (c.get("keywords") or [])[:4],
            "evidence": _GRADE_WORDS.get(c.get("grade"), c.get("grade")), "tested_on": c.get("checkpoints"),
            "image_url": example_url(pool, c["key"], rid), "unbent_url": _baseline_url(c, rid),
            **({"caption": cap.get("value"), "caption_by": cap.get("author")} if cap else {}),
            "recipe": c.get("recipe")}


_ID = re.compile(r"[0-9a-f]{8,64}")


def _pin_example(pin: dict, family: str, plain_group: dict) -> dict | None:
    """A hand-picked example (data/kb/intro.json) as intro_examples returns it; None if the entry is not usable."""
    rid, bid, src = pin.get("record"), pin.get("baseline"), pin.get("source")
    if not (isinstance(rid, str) and _ID.fullmatch(rid) and src in kb.SOURCES and pin.get("group") in plain_group):
        return None
    return {"record": rid, "part": pin.get("part"), "where": plain_group[pin["group"]],
            "how": _how(pin.get("op"), kb.amount_bucket(pin.get("op"), pin.get("args") or {})),
            "when": _WINDOW_WORDS.get(pin.get("window", "all"), pin.get("window")),
            "caption": pin.get("caption"), "caption_by": pin.get("caption_by"),
            "evidence": "one render, picked by hand as a clear example", "tested_on": [pin.get("checkpoint")],
            "image_url": f"{HF}/records/{family}/{src}/{rid}/output.webp",
            "unbent_url": f"{HF}/baselines/{bid}.webp" if isinstance(bid, str) and _ID.fullmatch(bid) else None,
            "recipe": pin.get("recipe")}


def intro_examples(arch: str = "sd15", n: int = 4, refresh: bool = False) -> dict:
    """A few example bends from the community knowledge base for a first session: the first three cover the early
    stages, the core and the late stages of the model. Each says what was bent in plain words, what it looked like
    and who said so, how well it is evidenced, and gives the example picture (with the unbent picture when it is
    known). Hand-picked examples (data/kb/intro.json) come first, where the index in use holds their source;
    the rest are picked from the index by evidence. Illustrations only: this does not choose the session's
    knowledge sources."""
    from surprise import PLAIN_GROUP, REGIONS
    n = max(1, min(n, 6))
    idx, where = community_index(refresh)
    cells = [c for c in kb.read_jsonl(idx / "cells.jsonl") if c.get("example_paths")]
    family = kb.family_of(arch)
    note = None
    pool = [c for c in cells if c.get("family") == family]
    if not pool:
        family = "sd1"
        pool = [c for c in cells if c.get("family") == family]
        note = (f"there are no examples for {arch or 'this model'} yet: these come from SD1.5-type models, and "
                f"this model may respond differently")
    vocab = kb.load_vocab()["tags"]
    region_of = {g: r for r, gs in REGIONS.items() for g in gs}
    published = {s for c in pool for s in c.get("sources") or {}}  # sources the index in use holds records of
    pool = [c for c in pool if c.get("group") in region_of]
    by_key = {c["key"]: c for c in pool}

    # hand-picked entries that can be shown: a cell key of the index, or a full entry whose source is published
    pins = json.loads(INTRO_PINS.read_text(encoding="utf-8")).get(family, []) if INTRO_PINS.exists() else []
    usable = []
    for pin in pins:
        if isinstance(pin, str) and pin in by_key:
            c = by_key[pin]
            usable.append((_cell_example(c, pool, vocab, PLAIN_GROUP, region_of[c["group"]]), False, c))
        elif isinstance(pin, dict) and pin.get("source") in published:
            ex = _pin_example(pin, family, PLAIN_GROUP)
            if ex and ex["part"] in REGIONS:
                usable.append((ex, bool(pin.get("fallback")), None))
    out: list[dict] = []
    for part in REGIONS:  # the first usable entry of each part, a fallback only when the part has nothing else
        of_part = [u for u in usable if u[0]["part"] == part]
        first = next((u for u in of_part if not u[1]), of_part[0] if of_part else None)
        if first and len(out) < n:
            out.append(first[0])
    out += [u[0] for u in usable if not u[1] and u[0] not in out][: n - len(out)]

    picked = {u[2]["key"] for u in usable if u[2] is not None and u[0] in out}
    used_ops, used_tags, regions = set(), set(), list(REGIONS)
    while len(out) < n:  # the rest: from the index, by evidence, a different part and kind of nudge each time
        have = [e["part"] for e in out]
        region = next((r for r in regions if r not in have), regions[len(out) % len(regions)])
        cands = [c for c in pool if region_of[c["group"]] == region and c["key"] not in picked] \
            or [c for c in pool if c["key"] not in picked]
        if not cands:
            break
        best = max(cands, key=lambda c: (_intro_score(c) - 1.5 * (c["op"] in used_ops)
                                         - (next(iter(c.get("effect_tags") or {}), None) in used_tags), c["key"]))
        picked.add(best["key"])
        used_ops.add(best["op"])
        used_tags.add(next(iter(best.get("effect_tags") or {}), None))
        out.append(_cell_example(best, pool, vocab, PLAIN_GROUP, region))
    return {"from": where, "family": family, "examples": out, **({"note": note} if note else {}),
            "say": "Show the pictures as one sheet (view_images, or metrics.py sheet), unbent next to bent where "
                   "there is one. Say that these are examples from the shared library of bends other people have "
                   "tried, on the models listed, and that captions, descriptions and keywords are the reading of "
                   "whoever `by` / `caption_by` names. They illustrate what bending can do; whether to use the "
                   "library for the artist's own work is still their choice (kb_sources)."}


def status(session: str = "") -> dict:
    idx, where = community_index()
    meta = json.loads((idx / "meta.json").read_text(encoding="utf-8")) if (idx / "meta.json").exists() else {}
    own = sum(1 for _ in kb.iter_records(OWN)) if (OWN / "records").exists() else 0
    return {"community": {"from": where, **{k: meta.get(k) for k in ("built", "records", "cells", "findings",
                                                                     "families")}},
            "mine": {"folder": str(OWN), "records": own},
            **({"session_sources": session_sources(session)} if session else {})}


def description_prompt() -> dict:
    """The knowledge base's description prompt, for an agent describing a user's bent result, plus how to use it."""
    return {"prompt_version": kb.CAPTION_PROMPT_VERSION, "prompt": kb.caption_prompt(),
            "effect_tags": list(kb.load_vocab()["tags"]),
            "how": ["Look at the unbent picture and the bent one side by side (view_images, unbent first).",
                    "Follow the prompt as if you had not been told the bend: describe only what is visible.",
                    "Answer caption, change, keywords and effect_tags for each pair.",
                    "Show the user your description and let them edit or drop it before it is kept.",
                    f"Save it with log_round(..., caption, change, keywords, effect_tags, agent_model, "
                    f"prompt_version='{kb.CAPTION_PROMPT_VERSION}')."]}


def main(argv=None) -> None:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    lg = sub.add_parser("log")
    lg.add_argument("prompt_id")
    lg.add_argument("--session", required=True)
    for k in ("verdict", "words", "caption", "change", "agent-model", "baseline", "arch", "contributor"):
        lg.add_argument(f"--{k}", default="")
    lg.add_argument("--tags", default="", help="comma-separated effect tags (the agent's)")
    lg.add_argument("--keywords", default="", help="comma-separated keywords (the agent's)")
    lg.add_argument("--prompt-version", default="", help=f"{kb.CAPTION_PROMPT_VERSION} when written with describe-prompt")
    sub.add_parser("describe-prompt", help="the knowledge base's prompt for describing a bent result")
    fd = sub.add_parser("find")
    fd.add_argument("goal")
    fd.add_argument("--arch", default="")
    fd.add_argument("--sources", default="community,mine")
    fd.add_argument("--limit", type=int, default=6)
    fd.add_argument("--refresh", action="store_true")
    st = sub.add_parser("status")
    st.add_argument("--session", default="")
    so = sub.add_parser("sources", help="store a session's choice: community, mine, both, none")
    so.add_argument("session")
    so.add_argument("choice", choices=("community", "mine", "both", "none"))
    it = sub.add_parser("intro", help="a few community examples for a first session, one per part of the model")
    it.add_argument("--arch", default="sd15")
    it.add_argument("-n", type=int, default=4)
    it.add_argument("--refresh", action="store_true")
    a = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if a.cmd == "intro":
        r = intro_examples(a.arch, a.n, a.refresh)
    elif a.cmd == "log":
        r = log_round(a.prompt_id, a.session, a.verdict, a.words, a.caption, a.change,
                      [t for t in a.tags.split(",") if t], a.agent_model, a.baseline, a.arch, a.contributor,
                      [k for k in a.keywords.split(",") if k.strip()], a.prompt_version)
    elif a.cmd == "describe-prompt":
        r = description_prompt()
    elif a.cmd == "find":
        r = find(a.goal, a.arch, [s for s in a.sources.split(",") if s], a.limit, a.refresh)
    elif a.cmd == "sources":
        r = session_sources(a.session, {"both": ["community", "mine"], "none": []}.get(a.choice, [a.choice]))
    else:
        r = status(a.session)
    print(json.dumps(r, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
