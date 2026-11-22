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
import re
from pathlib import Path

SCHEMA = "corpus/v1"
OPERATIONS = ("local_inference", "colab", "training", "listening", "publication")
TIERS = ("smoke", "dev", "sealed")
SPLITS = ("train", "eval", "control")
GROUP_KEYS = ("take", "fingerprint", "session", "artist", "album", "stem_family")
RELEASE_RATES = (44100, 48000, 96000)
PILOT_FIELDS = ("sample_size", "gate_margin_sdr_db", "gate_margin_vs_external_db",
                "min_coverage", "cpu_p95_s_per_audio_min", "rss_mb_p95")
_ID = re.compile(r"[A-Za-z0-9_-]+")


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


def _check_spec(spec, idx, where, errors):
    for field in ("path", "sha256", "bytes"):
        if field not in spec:
            errors.append(f"{where}: missing '{field}'")
            return
    entry = idx.get(spec["path"])
    if entry is None:
        errors.append(f"{where}: '{spec['path']}' not declared in files[]")
    elif entry["sha256"] != spec["sha256"] or entry["bytes"] != spec["bytes"]:
        errors.append(f"{where}: spec for '{spec['path']}' disagrees with files[]")


def validate_manifest(doc, check_hashes_root=None):
    """Return a list of errors; empty means the manifest is fit to run."""
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
        pilot = doc.get("pilot") or {}
        for field in PILOT_FIELDS:
            if pilot.get(field) is None:
                errors.append(f"sealed tier blocked: pilot.{field} not registered")
        if not pilot.get("registered_at"):
            errors.append("sealed tier blocked: pilot.registered_at missing")
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
    errors = []
    root = Path(root).resolve()
    for f in doc.get("files", []):
        target = (root / f["path"]).resolve()
        try:
            target.relative_to(root)
        except ValueError:
            errors.append(f"file escapes root: {f['path']}")
            continue
        if f.get("held_by") == "custodian":
            continue  # custodian-held sealed bytes are not distributable
        if not target.exists():
            errors.append(f"missing file: {f['path']}")
        elif target.stat().st_size != f["bytes"] or digest(target) != f["sha256"]:
            errors.append(f"hash mismatch: {f['path']}")
    return errors


def resolve(doc, root, spec):
    """Verified absolute path for a manifest file spec."""
    root = Path(root).resolve()
    target = (root / spec["path"]).resolve()
    target.relative_to(root)
    entry = _files_index(doc).get(spec["path"])
    if entry is None or entry["sha256"] != spec["sha256"]:
        raise ValueError(f"undeclared or mismatched file spec: {spec['path']}")
    if entry.get("held_by") == "custodian":
        raise PermissionError(f"custodian-held sealed input: {spec['path']}")
    if not target.exists() or digest(target) != spec["sha256"]:
        raise ValueError(f"input hash mismatch: {target}")
    return target


def load(path):
    return json.loads(Path(path).read_text())
