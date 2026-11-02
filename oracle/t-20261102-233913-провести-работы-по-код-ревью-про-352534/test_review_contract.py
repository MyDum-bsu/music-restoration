"""Oracle for t-20261102-233913: contracts the code-review follow-up must satisfy.

Static + subprocess checks only, no network, no media files.
Run from the repository root:  python3 -m pytest .factory/integration/oracle/<ID> -q
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

ID = "t-20261102-233913-провести-работы-по-код-ревью-про-352534"


def repo_root() -> Path:
    here = Path(__file__).resolve()
    for p in [here, *here.parents]:
        if (p / "pyproject.toml").is_file() and (p / "rnext").is_dir():
            return p
    raise RuntimeError("repository root not found (no pyproject.toml + rnext/ above oracle)")


ROOT = repo_root()


def read(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8")


def py_files(*dirs: str) -> list[Path]:
    out: list[Path] = []
    for d in dirs:
        base = ROOT / d
        if base.is_dir():
            out.extend(sorted(base.rglob("*.py")))
    return out


def skill_docs() -> list[Path]:
    base = ROOT / "skills"
    assert base.is_dir(), "skills/ is missing"
    docs = sorted(base.rglob("*.md"))
    assert docs, "no skills/**/*.md found"
    return docs


# ---------------------------------------------------------------- HIGH: one scorer
def test_single_release_quality_scorer():
    planner = read("rnext/planner.py")
    hunt = read("scripts/tracker_hunt.py")

    assert re.search(r"^def score_facts\(", planner, re.M), (
        "rnext/planner.py must own the only version-quality scorer: def score_facts(facts)"
    )
    assert "score_facts" in hunt, (
        "scripts/tracker_hunt.py must delegate to rnext.planner.score_facts"
    )

    # The weight tables may live in exactly one source file, the planner.
    owners = [p for p in py_files("rnext", "scripts", "research") if "BDRemux" in p.read_text(encoding="utf-8")]
    names = [p.relative_to(ROOT).as_posix() for p in owners]
    assert names == ["rnext/planner.py"], (
        "version-quality weights must be implemented once, in rnext/planner.py; "
        f"found BDRemux table in: {names}"
    )


# ---------------------------------------------------------------- HIGH: no machine paths
MACHINE_PATH = re.compile(r"/media/|/home/|Desktop/projects|[A-Za-z]:\\\\")


def test_skills_have_no_machine_paths():
    offenders = []
    for doc in skill_docs():
        text = doc.read_text(encoding="utf-8")
        for line_no, line in enumerate(text.splitlines(), 1):
            if MACHINE_PATH.search(line):
                offenders.append(f"{doc.relative_to(ROOT).as_posix()}:{line_no}: {line.strip()}")
    assert not offenders, "machine-specific paths in skills:\n" + "\n".join(offenders)


# ---------------------------------------------------------------- MEDIUM: paths in skills exist
REF = re.compile(r"\b(?:scripts|research|rnext|config|tests)/[A-Za-z0-9_./-]+\.(?:py|json|md|sh)\b")
# machine-local files that exist by setup, not in git; never required by the oracle
LOCAL_ONLY = {"config/workspace.json"}


def test_referenced_paths_in_skills_exist():
    missing = []
    for doc in skill_docs():
        for rel in set(REF.findall(doc.read_text(encoding="utf-8"))):
            if rel in LOCAL_ONLY:
                continue
            if not (ROOT / rel).exists():
                missing.append(f"{doc.relative_to(ROOT).as_posix()} -> {rel}")
    assert not missing, "skills reference non-existent paths:\n" + "\n".join(sorted(set(missing)))


# ---------------------------------------------------------------- MEDIUM: qc_gates --orig
def run_help(rel: str) -> str:
    proc = subprocess.run(
        [sys.executable, rel, "--help"],
        cwd=str(ROOT), capture_output=True, text=True, timeout=180,
    )
    assert proc.returncode == 0, f"{rel} --help failed ({proc.returncode}): {proc.stderr[-2000:]}"
    return proc.stdout + proc.stderr


def test_qc_gates_has_optional_orig_mode():
    out = run_help("scripts/qc_gates.py")
    assert "--orig" in out, "qc_gates.py must expose an optional --orig flag"


def test_qc_gates_default_path_unchanged():
    src = read("scripts/qc_gates.py")
    # The additive-only rule: no removal of the legacy spike verdict text.
    assert re.search(r"spike", src, re.I), "qc_gates.py lost its spike gate entirely"
    # --orig must be optional, i.e. argparse entry without required=True.
    for line in src.splitlines():
        if "--orig" in line and "add_argument" in line:
            assert "required=True" not in line, "--orig must stay optional"


# ---------------------------------------------------------------- MEDIUM: parser tests
PARSERS = ("parse_music", "parse_video", "parse_topic_page", "parse_dr_table")


def test_parser_tests_on_html_fixtures():
    tests = sorted((ROOT / "tests").rglob("*.py")) if (ROOT / "tests").is_dir() else []
    assert tests, "tests/ is missing"
    hit = [p for p in tests if all(name in p.read_text(encoding="utf-8") for name in PARSERS)]
    assert hit, f"no test file exercises all of {PARSERS}"
    for p in hit:
        src = p.read_text(encoding="utf-8")
        assert src.count("assert") >= 4, f"{p.relative_to(ROOT)} has fewer than 4 assertions"
    fixtures = sorted((ROOT / "tests").rglob("*.htm*"))
    assert fixtures, "parser tests must be pinned to saved HTML fixtures under tests/"


# ---------------------------------------------------------------- MEDIUM: setup docs + cookie refresh
def test_setup_docs_and_browser_refresh():
    setup = ROOT / "docs/setup.md"
    assert setup.is_file(), "docs/setup.md is missing"
    assert "cf_clearance" in setup.read_text(encoding="utf-8"), "docs/setup.md must cover cf_clearance refresh"
    script = ROOT / "scripts/rt_browser_refresh.py"
    assert script.is_file(), "scripts/rt_browser_refresh.py is missing"
    run_help("scripts/rt_browser_refresh.py")


# ---------------------------------------------------------------- MEDIUM: legacy batch deprecated
def test_release_batch_deprecated():
    p = ROOT / "scripts/release_batch.py"
    if not p.is_file():
        return  # removal is an accepted outcome
    text = p.read_text(encoding="utf-8").lower()
    assert any(w in text for w in ("deprecat", "legacy", "устарел", "устар")), (
        "scripts/release_batch.py still looks like a live entry point: mark it deprecated"
    )


# ---------------------------------------------------------------- LOW: repo hygiene
def test_coding_standards_present():
    assert (ROOT / "CODING_STANDARDS.md").is_file(), "CODING_STANDARDS.md is missing at the repo root"


def test_ci_runs_pytest():
    wf = ROOT / ".github/workflows"
    files = sorted([*wf.glob("*.yml"), *wf.glob("*.yaml")]) if wf.is_dir() else []
    assert files, ".github/workflows/*.yml with pytest is missing"
    assert any("pytest" in f.read_text(encoding="utf-8") for f in files), "CI does not run pytest"


def test_workspace_example_has_no_machine_paths():
    p = ROOT / "config/workspace.example.json"
    assert p.is_file(), "config/workspace.example.json is missing"
    text = p.read_text(encoding="utf-8")
    json.loads(text)
    assert not re.search(r"/media/|/home/", text), "workspace example must not ship machine paths"


def test_version_bumped():
    text = read("pyproject.toml")
    m = re.search(r'^version\s*=\s*"([^"]+)"', text, re.M)
    assert m, "pyproject.toml has no version"
    parts = tuple(int(x) for x in re.findall(r"\d+", m.group(1))[:2])
    assert parts >= (2, 1), f"pyproject version is {m.group(1)}, expected >= 2.1 (history is at v2.1)"


def test_skill_frontmatter_is_quoted():
    bad = []
    for doc in (ROOT / "skills").rglob("SKILL.md"):
        lines = doc.read_text(encoding="utf-8").splitlines()
        assert lines and lines[0].strip() == "---", f"{doc.relative_to(ROOT)}: no YAML frontmatter"
        for line in lines[1:]:
            if line.strip() == "---":
                break
            m = re.match(r"^[\w-]+:\s*(.+)$", line)
            if not m:
                continue
            value = m.group(1)
            if ": " in value and value[0] not in "\"'":
                bad.append(f"{doc.relative_to(ROOT)}: {line}")
    assert not bad, "unquoted YAML values with ': ' in SKILL.md frontmatter:\n" + "\n".join(bad)


def test_review_report_present():
    assert (ROOT / "docs/code-review-2026-11-02.md").is_file(), "the review report must stay in the repo"
