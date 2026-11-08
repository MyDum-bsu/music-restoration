#!/usr/bin/env python3
"""Fetch the hash-pinned public corpus and write a benchmark manifest.

Usage: python3 research/fetch_declipping.py --out data/declipping
Use --cached-only to verify existing files without network access.
No audio from the corpus is redistributed in this repository.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen

CATALOG = Path(__file__).resolve().parents[1] / "docs/validation/declipping-corpus.json"


def _target(root, relative):
    target = (root / relative).resolve()
    target.relative_to(root.resolve())
    return target


def prepare(catalog, out, cached_only=False):
    """Verify exact source bytes and return a manifest with resolved local paths."""
    out = Path(out)
    for spec in catalog["files"]:
        target = _target(out, spec["path"])
        exists = target.exists()
        if exists:
            if target.stat().st_size != spec["bytes"]:
                raise ValueError(f"cached size mismatch: {target}")
            raw = target.read_bytes()
        elif cached_only:
            raise FileNotFoundError(target)
        else:
            with urlopen(spec["url"], timeout=30) as response:
                raw = response.read(spec["bytes"] + 1)
        if len(raw) != spec["bytes"] or hashlib.sha256(raw).hexdigest() != spec["sha256"]:
            raise ValueError(f"corpus size/hash mismatch: {target}")
        if not exists:
            target.parent.mkdir(parents=True, exist_ok=True)
            partial = target.with_suffix(target.suffix + ".part")
            partial.write_bytes(raw)
            partial.replace(target)
    manifest = {"source": catalog["source"], "cases": copy.deepcopy(catalog["cases"])}
    for case in manifest["cases"]:
        for spec in [case["clean"], case["degraded"], *case.get("references", {}).values()]:
            spec["path"] = str(_target(out, spec["path"]))
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--cached-only", action="store_true")
    args = parser.parse_args()
    catalog = json.loads(CATALOG.read_text())
    manifest = prepare(catalog, args.out, args.cached_only)
    target = Path(args.out) / "benchmark-manifest.json"
    target.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Verified {len(catalog['files'])} files; manifest: {target}")


if __name__ == "__main__":
    main()
