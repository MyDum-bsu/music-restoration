#!/usr/bin/env python3
"""corpus/v1 manifest schema and validator.

A v1 manifest is the single source of truth for a benchmark tier: dataset
version, pinned toolchain, RNG, hash-pinned audio files, per-case rights and
group keys (take/fingerprint/session/artist/album/stem_family) that define the
sealed split. `validate_manifest` is deliberately strict: it returns a list of
errors, never guesses, and the runner refuses to execute when it is non-empty.

Gates:
- every referenced file carries sha256 + bytes; --check-hashes verifies content
- `operation` rights are per case; a case without the right cannot be run
- group keys must not intersect across different `split` values
- sealed tier additionally requires a named custodian and a fully registered
  pilot (sample_size, gate margins, coverage, CPU p95/RSS); unset values block
  sealed but do not block dev/smoke
"""
import hashlib
import json
import os
import platform
import re
import subprocess
from pathlib import Path

SCHEMA = "corpus/v1"
OPERATIONS = ("local_inference", "colab", "training", "listening", "publication")
TIERS = ("smoke", "dev", "sealed")
SPLITS = ("train", "eval", "control")
GROUP_KEYS = ("take", "fingerprint", "session", "artist", "album", "stem_family")
RELEASE_RATES = (44100, 48000, 96000)
PILOT_FIELDS = ("sample_size", "gate_margin_sdr_db", "gate_margin_vs_external_db",
                "min_coverage", "cpu_p95_s_per_audio_min", "rss_mb_p95")
TOOLCHAIN_FIELDS = ("python", "numpy", "scipy", "ffmpeg", "libflac")
_ID = re.compile(r"[A-Za-z0-9_-]+")
_SHA256 = re.compile(r"[0-9a-f]{64}")
# exact pin like "3.12.3" or "N-125907-ga7e72069f1": no ranges, placeholders or globs
_PIN = re.compile(r"[A-Za-z0-9]+(?:[.\-_+][A-Za-z0-9]+)*")


def _version(cmd, pattern=re.compile(r"[0-9]+(?:[.\-_+][A-Za-z0-9]+)+")):
    out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
    m = pattern.search(out)
    if not m:
        raise RuntimeError(f"cannot parse version from: {cmd}")
    return m.group(0)


def actual_toolchain():
    """Detected exact toolchain pins for this environment."""
    import numpy, scipy
    return {"python": platform.python_version(), "numpy": numpy.__version__,
            "scipy": scipy.__version__,
            "ffmpeg": _version(["ffmpeg", "-version"]),
            "libflac": _version(["flac", "--version"], re.compile(r"[0-9.]+"))}


def toolchain_errors(toolchain):
    errors = []
    for field in TOOLCHAIN_FIELDS:
        value = (toolchain or {}).get(field)
        if not isinstance(value, str) or not _PIN.fullmatch(value) \
                or not any(ch.isdigit() for ch in value):
            errors.append(f"toolchain.{field} must be an exact pinned version, "
                          f"got {value!r}")
    return errors


def pilot_errors(pilot):
    """Positive numbers for every gate field plus registration time."""
    errors = []
    for field in PILOT_FIELDS:
        value = (pilot or {}).get(field)
        if not isinstance(value, (int, float)) or isinstance(value, bool) \
                or value <= 0:
            errors.append(f"sealed tier blocked: pilot.{field} not registered "
                          f"as a positive number")
    if not (pilot or {}).get("registered_at"):
        errors.append("sealed tier blocked: pilot.registered_at missing")
    return errors


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def manifest_digest(doc):
    """Content hash of the manifest itself (protocol pin)."""
    return hashlib.sha256(json.dumps(doc, sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()


def _files_index(doc):
    return {f["path"]: f for f in doc.get("files", [])}


def _held(entry_or_spec):
    return (entry_or_spec or {}).get("held_by") == "custodian"


def _check_spec(spec, idx, where, errors):
    for field in ("path", "sha256", "bytes"):
        if field not in spec:
            errors.append(f"{where}: missing '{field}'")
            return
    entry = idx.get(spec["path"])
    held = _held(entry) or _held(spec)
    # format is checked whenever the fields carry a value, custodian-held or not
    if spec["sha256"] is not None and \
            (not isinstance(spec["sha256"], str) or not _SHA256.fullmatch(spec["sha256"])):
        errors.append(f"{where}: sha256 must be 64 lowercase hex chars")
    if spec["bytes"] is not None and \
            (not isinstance(spec["bytes"], int) or isinstance(spec["bytes"], bool)
             or spec["bytes"] <= 0):
        errors.append(f"{where}: bytes must be a positive int")
    if entry is None:
        errors.append(f"{where}: '{spec['path']}' not declared in files[]")
    else:
        if _held(entry) != _held(spec):
            errors.append(f"{where}: held_by disagrees with files[]")
        # filled fields must agree with files[] even for custodian-held inputs;
        # a pin present in files[] but missing from the spec is an error too
        for field in ("sha256", "bytes"):
            if entry[field] is not None and spec[field] is None:
                errors.append(f"{where}: '{field}' pinned in files[] but "
                              f"missing from spec")
            elif spec[field] is not None and entry[field] != spec[field]:
                errors.append(f"{where}: spec for '{spec['path']}' disagrees "
                              f"with files[]")
    if held:
        return  # custodian-held bytes: content checked by custodian, not here
    if spec["sha256"] is None or spec["bytes"] is None:
        errors.append(f"{where}: sha256/bytes required for non-custodian files")


def validate_manifest(doc, check_hashes_root=None):
    """Return a list of errors; empty means the manifest is fit to run."""
    from research.corpus_synth import DEFECT_OPERATORS
    errors = []
    if doc.get("schema") != SCHEMA:
        errors.append(f"schema must be '{SCHEMA}'")
    tier = doc.get("tier")
    if tier not in TIERS:
        errors.append(f"tier must be one of {TIERS}")
    if not doc.get("dataset_version"):
        errors.append("missing dataset_version")
    if not doc.get("toolchain"):
        errors.append("missing toolchain pin")
    else:
        errors += toolchain_errors(doc["toolchain"])
    if doc.get("rng", {}).get("seed") is None:
        errors.append("missing rng.seed")

    idx = _files_index(doc)
    paths = [f["path"] for f in doc.get("files", [])]
    if len(paths) != len(set(paths)):
        errors.append("duplicate paths in files[]")

    cases = doc.get("cases", [])
    ids = [c.get("id", "?") for c in cases]
    if len(ids) != len(set(ids)) or any(not _ID.fullmatch(i) for i in ids):
        errors.append("case ids must be unique simple names")
    if not cases:
        errors.append("manifest has no cases")

    sealed_case_ids = set()
    for c in cases:
        where = f"case {c.get('id', '?')}"
        rights = c.get("rights")
        if not isinstance(rights, dict) or any(k not in rights for k in OPERATIONS):
            errors.append(f"{where}: rights must declare all of {OPERATIONS}")
        else:
            for k, v in rights.items():
                if not isinstance(v, bool):
                    errors.append(f"{where}: rights.{k} must be a bool, got {v!r}")
        groups = c.get("groups", {})
        for key in GROUP_KEYS:
            if key not in groups:
                errors.append(f"{where}: missing group key '{key}'")
        if c.get("channels") != 2:
            errors.append(f"{where}: release profile requires stereo (channels=2)")
        if c.get("rate") not in RELEASE_RATES:
            errors.append(f"{where}: rate must be one of {RELEASE_RATES}")
        split = c.get("split")
        if tier != "smoke" and split not in SPLITS:
            errors.append(f"{where}: split must be one of {SPLITS}")
        if tier == "sealed" or c.get("sealed"):
            sealed_case_ids.add(c.get("id"))
        kind = c.get("kind")
        if kind == "paired_synthetic":
            syn = c.get("synthesis", {})
            for field in ("generator", "seed", "rate", "seconds", "operator"):
                if field not in syn:
                    errors.append(f"{where}: synthesis missing '{field}'")
            if syn.get("rate") != c.get("rate"):
                errors.append(f"{where}: synthesis.rate disagrees with case rate")
        elif kind not in ("paired_file", "unpaired"):
            errors.append(f"{where}: kind must be paired_synthetic/paired_file/unpaired")
        if kind == "paired_synthetic":
            if "mask" not in c:
                errors.append(f"{where}: missing 'mask' file spec")
            else:
                _check_spec(c["mask"], idx, f"{where}.mask", errors)
        if kind == "paired_synthetic" and \
                c.get("synthesis", {}).get("operator") in DEFECT_OPERATORS and \
                c.get("clean", {}).get("sha256") == c.get("degraded", {}).get("sha256"):
            errors.append(f"{where}: defect operator produced clean == degraded "
                          f"(empty defect)")
        if kind != "unpaired":
            for role in ("clean", "degraded"):
                if role not in c:
                    errors.append(f"{where}: missing '{role}' file spec")
                else:
                    _check_spec(c[role], idx, f"{where}.{role}", errors)
        else:
            _check_spec(c.get("degraded", {}), idx, f"{where}.degraded", errors)
        for name, spec in c.get("references", {}).items():
            _check_spec(spec, idx, f"{where}.references.{name}", errors)

    errors += check_group_leakage(cases)

    if tier == "sealed":
        custodian = doc.get("custodian") or {}
        if not custodian.get("name"):
            errors.append("sealed tier blocked: custodian not appointed")
        errors += pilot_errors(doc.get("pilot"))
    elif doc.get("pilot"):
        for field in PILOT_FIELDS:
            if field not in doc["pilot"]:
                errors.append(f"pilot missing field '{field}'")

    if check_hashes_root is not None and not errors:
        errors += verify_files(doc, check_hashes_root)
    return errors


def check_group_leakage(cases):
    """A take/fingerprint/session/artist/album/stem_family may not span splits."""
    errors = []
    seen = {}  # (key, value) -> split
    for c in cases:
        split = c.get("split")
        if split is None:
            continue
        for key in GROUP_KEYS:
            value = (c.get("groups") or {}).get(key)
            if value is None:
                continue
            tag = (key, value)
            if tag in seen and seen[tag] != split:
                errors.append(f"case {c['id']}: group {key}='{value}' appears in both "
                              f"'{seen[tag]}' and '{split}'")
            seen.setdefault(tag, split)
    return errors


def case_allowed(case, operation):
    return bool((case.get("rights") or {}).get(operation))


def sealed_cases(doc):
    return [c for c in doc.get("cases", []) if doc.get("tier") == "sealed" or c.get("sealed")]


def verify_files(doc, root):
    """Verify declared bytes on disk; sealed files held by custodian are skipped."""
    from research import corpus_synth
    errors = []
    root = Path(root).resolve()
    skipped = set()  # paths already failed or custodian-held: never opened again
    for f in doc.get("files", []):
        target = (root / f["path"]).resolve()
        try:
            target.relative_to(root)
        except ValueError:
            errors.append(f"file escapes root: {f['path']}")
            skipped.add(f["path"])
            continue
        if f.get("held_by") == "custodian":
            skipped.add(f["path"])
            continue  # custodian-held sealed bytes are not distributable
        if not target.exists():
            errors.append(f"missing file: {f['path']}")
            skipped.add(f["path"])
        elif target.stat().st_size != f["bytes"] or digest(target) != f["sha256"]:
            errors.append(f"hash mismatch: {f['path']}")
            skipped.add(f["path"])
    for c in doc.get("cases", []):
        syn = c.get("synthesis") or {}
        if syn.get("operator") not in corpus_synth.DEFECT_OPERATORS or "mask" not in c:
            continue
        if c["mask"]["path"] in skipped:
            continue  # do not read/decode a mask that already failed or is held
        target = (root / c["mask"]["path"]).resolve()
        if not target.exists():
            continue  # already reported by the files[] pass
        try:
            mask = corpus_synth.decode_mask(target.read_bytes())
        except ValueError as exc:
            errors.append(f"case {c['id']}: corrupt mask: {exc}")
            continue
        if not mask.any():
            errors.append(f"case {c['id']}: defect operator '{syn['operator']}' "
                          f"has an empty true mask")
    return errors


def resolve(doc, root, spec, allow_custodian=False):
    """Verified absolute path for a manifest file spec."""
    root = Path(root).resolve()
    target = (root / spec["path"]).resolve()
    target.relative_to(root)
    entry = _files_index(doc).get(spec["path"])
    if entry is None or entry["sha256"] != spec["sha256"] \
            or entry["bytes"] != spec["bytes"]:
        raise ValueError(f"undeclared or mismatched file spec: {spec['path']}")
    if entry.get("held_by") == "custodian" and not allow_custodian:
        raise PermissionError(f"custodian-held sealed input: {spec['path']}")
    if not _SHA256.fullmatch(spec["sha256"] or ""):
        raise ValueError(f"malformed sha256 in spec: {spec['path']}")
    if not target.exists() or target.stat().st_size != spec["bytes"] \
            or digest(target) != spec["sha256"]:
        raise ValueError(f"input hash/size mismatch: {target}")
    return target


def is_sealed_case(doc, case):
    """Sealed by tier, flag, or any custodian-held input."""
    if doc.get("tier") == "sealed" or case.get("sealed"):
        return True
    idx = _files_index(doc)
    specs = [c for c in (case.get(r) for r in ("clean", "degraded", "mask"))
             if isinstance(c, dict)]
    specs += list(case.get("references", {}).values())
    return any(_held(s) or _held(idx.get(s.get("path"))) for s in specs)


def used_paths(cases):
    """File paths referenced by a set of cases."""
    paths = set()
    for c in cases:
        for role in ("clean", "degraded", "mask"):
            if isinstance(c.get(role), dict):
                paths.add(c[role]["path"])
        for spec in c.get("references", {}).values():
            paths.add(spec["path"])
    return paths


def without_sealed(doc):
    """Manifest view with every sealed case (and its private files) removed."""
    keep = [c for c in doc.get("cases", []) if not is_sealed_case(doc, c)]
    paths = used_paths(keep)
    return {**doc, "cases": keep,
            "files": [f for f in doc.get("files", []) if f["path"] in paths]}


def validate_cross_manifest_leakage(manifests):
    """No group value may recur across different manifests (dev vs sealed...).

    Manifests are told apart by identity label (tier+dataset_version+position),
    not dataset_version alone — two manifests may share a version string.
    """
    errors = []
    seen = {}  # (key, value) -> manifest label
    for i, doc in enumerate(manifests):
        label = f"{doc.get('tier', '?')}:{doc.get('dataset_version', '?')}#{i}"
        for c in doc.get("cases", []):
            for key in GROUP_KEYS:
                value = (c.get("groups") or {}).get(key)
                if value is None:
                    continue
                tag = (key, value)
                if tag in seen and seen[tag] != label:
                    errors.append(f"case {c['id']}: group {key}='{value}' shared "
                                  f"between {seen[tag]} and {label}")
                seen.setdefault(tag, label)
    return errors


def load(path):
    return json.loads(Path(path).read_text())
