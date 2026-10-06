#!/usr/bin/env python3
"""Share a bent render with the bend knowledge base by hand: turn two ComfyUI pictures (the bent render and the same
setup without the bend) into a contribution folder, and check a folder before you upload it. Standard library only.

  python contribute.py prepare --bent bent.png --unbent unbent.png --agree-cc0 [--name "Ada" | --not-named]
                               [--describe "what changed"] [--out contribution]
  python contribute.py check contribution

ComfyUI saves the workflow inside every PNG (and inside its WebP files), so the pictures carry the model, prompt,
seed, sampler and the Apply Bends from JSON document; prepare reads them from there. Pictures must be PNG or lossless
WebP: JPEG and lossy WebP change the measurements, so they are refused.

The folder that prepare writes holds records/<family>/artist/<id>/ (record.json, output.png, workflow.json and, with
--describe, interpretations.jsonl) and baselines/<baseline id>.png. Upload both, as they are, to the dataset
huggingface.co/datasets/abuzreq/model-bending-knowledge-base with "Upload files" and "Open a Pull Request".
"""

from __future__ import annotations

import argparse
import json
import re
import struct
import sys
import zlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import kb  # noqa: E402

DATASET = kb.COMMUNITY_DATASET if hasattr(kb, "COMMUNITY_DATASET") else "abuzreq/model-bending-knowledge-base"
LICENSE = "CC0-1.0"


class ContributionError(ValueError):
    """Something in the pictures or the folder that has to be fixed before sharing."""


# --------------------------------------------------------------------------- reading ComfyUI's pictures
def image_kind(data: bytes) -> str:
    """png, webp-lossless, webp-lossy, jpeg or unknown."""
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "png"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        chunks = set(_riff_chunks(data))
        if b"VP8 " in chunks:
            return "webp-lossy"
        return "webp-lossless" if b"VP8L" in chunks else "unknown"
    if data[:3] == b"\xff\xd8\xff":
        return "jpeg"
    return "unknown"


def _riff_chunks(data: bytes):
    i = 12
    while i + 8 <= len(data):
        tag, size = data[i:i + 4], struct.unpack("<I", data[i + 4:i + 8])[0]
        yield tag
        i += 8 + size + (size & 1)


def _png_text(data: bytes) -> dict[str, str]:
    out, i = {}, 8
    while i + 8 <= len(data):
        size, tag = struct.unpack(">I", data[i:i + 4])[0], data[i + 4:i + 8]
        body = data[i + 8:i + 8 + size]
        if tag == b"tEXt":
            k, _, v = body.partition(b"\0")
            out[k.decode("latin-1")] = v.decode("latin-1")
        elif tag == b"zTXt":
            k, _, v = body.partition(b"\0")
            out[k.decode("latin-1")] = zlib.decompress(v[1:]).decode("latin-1")
        elif tag == b"iTXt":
            k, _, rest = body.partition(b"\0")
            comp, _method, rest = rest[0], rest[1], rest[2:]
            _lang, _, rest = rest.partition(b"\0")
            _tk, _, text = rest.partition(b"\0")
            out[k.decode("latin-1")] = (zlib.decompress(text) if comp else text).decode("utf-8")
        elif tag == b"IEND":
            break
        i += 12 + size
    return out


def _webp_text(data: bytes) -> dict[str, str]:
    """ComfyUI's WebP saver puts "prompt:<json>" and "workflow:<json>" in EXIF tags 0x0110 and 0x010F."""
    i, out = 12, {}
    while i + 8 <= len(data):
        tag, size = data[i:i + 4], struct.unpack("<I", data[i + 4:i + 8])[0]
        if tag == b"EXIF":
            for m in re.finditer(rb"(prompt|workflow):(\{.*?\})\x00", data[i + 8:i + 8 + size], re.S):
                out[m.group(1).decode()] = m.group(2).decode("utf-8", "replace")
        i += 8 + size + (size & 1)
    return out


def read_picture(path: Path) -> tuple[bytes, str, dict]:
    """(bytes, kind, the API prompt saved inside it)."""
    data = Path(path).read_bytes()
    kind = image_kind(data)
    if kind == "jpeg":
        raise ContributionError(f"{path}: JPEG is not accepted. Its compression changes the measurements; save as PNG "
                                "(what ComfyUI's Save Image does) or lossless WebP.")
    if kind == "webp-lossy":
        raise ContributionError(f"{path}: this WebP is lossy. Save it as PNG or lossless WebP.")
    if kind == "unknown":
        raise ContributionError(f"{path}: not a PNG or WebP picture.")
    text = _png_text(data) if kind == "png" else _webp_text(data)
    try:
        prompt = json.loads(text["prompt"])
    except (KeyError, ValueError):
        raise ContributionError(f"{path}: no ComfyUI workflow inside. Use the file ComfyUI saved (Save Image), not "
                                "a screenshot or an edited copy.") from None
    return data, kind, prompt


def _bend_document(prompt: dict) -> dict:
    for node in prompt.values():
        if node.get("class_type") == "ApplyBendsFromJSON":
            try:
                doc = json.loads(node.get("inputs", {}).get("bends_json") or "{}")
            except ValueError:
                return {}
            return doc if isinstance(doc, dict) else {"bends": doc}
    return {}


def facts(prompt: dict) -> tuple[dict, dict, list[dict]]:
    """(model, setup, bends) of a run, with the document's steps_min / steps_max given to each bend."""
    model, setup, bends = kb.setup_from_api_prompt(prompt)
    doc = _bend_document(prompt)
    window = {k: doc[k] for k in ("steps_min", "steps_max") if doc.get(k) is not None}
    if window:
        bends = [b if ("t" in b or "steps" in b) else {**b, **window} for b in bends]
    return model, setup, bends


# --------------------------------------------------------------------------- prepare
def prepare(bent: Path, unbent: Path, out: Path, *, name: str | None, agree: bool, describe: str = "",
            model_url: str = "") -> Path:
    """Write the contribution folder for one bent render; returns the record folder."""
    if not agree:
        raise ContributionError("Everything in the knowledge base is shared under CC0 (anyone may use it for anything). "
                                "Add --agree-cc0 to agree.")
    b_data, b_kind, b_prompt = read_picture(bent)
    u_data, u_kind, u_prompt = read_picture(unbent)
    model, setup, bends = facts(b_prompt)
    u_model, u_setup, u_bends = facts(u_prompt)
    if not bends:
        raise ContributionError(f"{bent}: no bend found in its workflow (Apply Bends from JSON).")
    if u_bends:
        raise ContributionError(f"{unbent}: this picture is bent too. The unbent picture is the same workflow with the "
                                "bend node bypassed or removed.")
    if not model.get("checkpoint"):
        raise ContributionError(f"{bent}: no checkpoint loader found in its workflow.")
    if not (setup.get("prompt") or "").strip():
        raise ContributionError(f"{bent}: no prompt found. The prompt is shared with every contribution, so others can "
                                "make the bend again.")
    setup.setdefault("negative", "")
    u_setup.setdefault("negative", "")
    if kb.record_id(model, setup, []) != kb.record_id(u_model, u_setup, []):
        diff = sorted(k for k in set(setup) | set(u_setup) if k in kb.ID_SETUP_KEYS and setup.get(k) != u_setup.get(k))
        diff += ["checkpoint"] if model.get("checkpoint") != u_model.get("checkpoint") else []
        raise ContributionError(f"The two pictures were not made with the same settings ({', '.join(diff)} differ). "
                                "The unbent picture must use the same model, prompt, seed and sampler, without the bend.")
    ext = "png" if b_kind == "png" else "webp"
    rec = kb.make_record(source="artist", model={**model, **({"source_url": model_url} if model_url else {})},
                         setup=setup, bends=bends, consent={"prompt": True, "input_image": False},
                         outputs={"image": f"output.{ext}", "workflow": "workflow.json"},
                         provenance={"producer": "contribute.py"})
    rec["outputs"]["baseline_id"] = kb.baseline_id(rec)
    rec["contribution"] = {"by": {"type": "human", "name": name or None}, "via": "hand",
                           "agreed": {"license": LICENSE, "at": kb.now()}}
    errors = kb.validate(rec, "record")
    if errors:
        raise ContributionError("; ".join(errors))
    d = Path(out) / "records" / rec["model"]["family"] / "artist" / rec["id"]
    d.mkdir(parents=True, exist_ok=True)
    (d / "record.json").write_text(json.dumps(rec, indent=1, ensure_ascii=False), encoding="utf-8")
    (d / f"output.{ext}").write_bytes(b_data)
    (d / "workflow.json").write_text(json.dumps(b_prompt, indent=1, ensure_ascii=False), encoding="utf-8")
    if describe.strip():
        author = {"type": "human", **({"name": name} if name else {})}
        entry = kb.make_interpretation({"record": rec["id"]}, "change", describe.strip(), author,
                                       basis="the contributor, comparing their bent and unbent pictures")
        (d / "interpretations.jsonl").write_text(json.dumps(entry, ensure_ascii=False) + "\n", encoding="utf-8")
    base = Path(out) / "baselines"
    base.mkdir(parents=True, exist_ok=True)
    (base / f"{rec['outputs']['baseline_id']}.{'png' if u_kind == 'png' else 'webp'}").write_bytes(u_data)
    return d


# --------------------------------------------------------------------------- check
def check(folder: Path) -> list[str]:
    """Problems with a contribution folder (the one prepare writes, or a single record folder inside it)."""
    folder = Path(folder)
    recs = [folder] if (folder / "record.json").exists() else sorted(p.parent for p in folder.glob("records/*/*/*/record.json"))
    if not recs:
        return [f"{folder}: no record.json found (expected records/<family>/<source>/<id>/record.json)"]
    root = folder if not (folder / "record.json").exists() else folder.parents[3] if len(folder.parents) > 3 else folder
    problems = []
    for d in recs:
        where = d.name
        try:
            rec = json.loads((d / "record.json").read_text(encoding="utf-8"))
        except ValueError as e:
            problems.append(f"{where}: record.json is not valid JSON ({e})")
            continue
        problems += [f"{where}: {e}" for e in kb.validate(rec, "record")]
        if rec.get("source") not in ("artist", "session"):
            problems.append(f"{where}: source must be 'artist' (by hand) or 'session' (with an agent)")
        if not (rec.get("consent") or {}).get("prompt") or not (rec.get("setup") or {}).get("prompt"):
            problems.append(f"{where}: the prompt must be shared (consent.prompt true and setup.prompt set)")
        if "negative" not in (rec.get("setup") or {}):
            problems.append(f"{where}: setup.negative is missing (use \"\" when there was none)")
        if rec.get("license") != LICENSE:
            problems.append(f"{where}: license must be {LICENSE}")
        if "contribution" not in rec:
            problems.append(f"{where}: the contribution block is missing (who shared it, how, and the CC0 agreement)")
        try:
            want = kb.record_id(rec["model"], rec["setup"], rec["bends"])
            if rec.get("id") != want or d.name != want:
                problems.append(f"{where}: the id should be {want} (the folder name and record.json's id)")
        except (KeyError, TypeError):
            pass
        img = (rec.get("outputs") or {}).get("image") or ""
        if not (d / img).exists():
            problems.append(f"{where}: {img or 'the picture'} is missing")
        else:
            kind = image_kind((d / img).read_bytes())
            if kind not in ("png", "webp-lossless"):
                problems.append(f"{where}: {img} is {kind}; use PNG or lossless WebP")
        if not (d / "workflow.json").exists():
            problems.append(f"{where}: workflow.json is missing (the ComfyUI workflow that made it)")
        bid = (rec.get("outputs") or {}).get("baseline_id")
        if not bid:
            problems.append(f"{where}: outputs.baseline_id is missing")
        elif not any((root / "baselines" / f"{bid}.{e}").exists() for e in ("png", "webp")):
            problems.append(f"{where}: the unbent picture baselines/{bid}.png (or .webp) is missing")
        elif bid != kb.baseline_id(rec):
            problems.append(f"{where}: baseline_id should be {kb.baseline_id(rec)}")
        if (d / "interpretations.jsonl").exists():
            for n, line in enumerate((d / "interpretations.jsonl").read_text(encoding="utf-8").splitlines(), 1):
                if line.strip():
                    try:
                        problems += [f"{where}: interpretations line {n}: {e}" for e in kb.validate(json.loads(line), "interpretation")]
                    except ValueError:
                        problems.append(f"{where}: interpretations line {n} is not valid JSON")
    return problems


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare", help="make a contribution folder from a bent and an unbent ComfyUI picture")
    p.add_argument("--bent", type=Path, required=True)
    p.add_argument("--unbent", type=Path, required=True)
    who = p.add_mutually_exclusive_group()
    who.add_argument("--name", help="the name the navigator shows with your render")
    who.add_argument("--not-named", action="store_true", help="share without a name (shown as \"not named\")")
    p.add_argument("--describe", default="", help="in your own words: what the bend changed")
    p.add_argument("--model-url", default="", help="where others can get the checkpoint")
    p.add_argument("--agree-cc0", action="store_true", help="share under CC0: anyone may use it for anything")
    p.add_argument("--out", type=Path, default=Path("contribution"))
    c = sub.add_parser("check", help="check a contribution folder before uploading it")
    c.add_argument("folder", type=Path)
    a = ap.parse_args(argv)
    try:
        if a.cmd == "prepare":
            d = prepare(a.bent, a.unbent, a.out, name=None if a.not_named else a.name, agree=a.agree_cc0,
                        describe=a.describe, model_url=a.model_url)
            problems = check(a.out)
            print(f"Wrote {d}")
            if problems:
                print("\n".join(problems))
                return 1
            print(f"Checked: ready. Upload the folders inside {a.out} (records/ and baselines/) to "
                  f"https://huggingface.co/datasets/{DATASET} with Files > Add file > Upload files, and choose "
                  "\"Open a Pull Request\".")
            return 0
        problems = check(a.folder)
        print("\n".join(problems) if problems else "Checked: ready to upload.")
        return 1 if problems else 0
    except ContributionError as e:
        print(e, file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
