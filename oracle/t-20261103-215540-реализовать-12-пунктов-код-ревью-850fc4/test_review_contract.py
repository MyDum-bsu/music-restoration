"""Contract oracle: 12 items of docs/code-review-2026-11-02.md.

One test per review item, in the order of the task request. Red on the base
revision, green only when all 12 items are actually implemented.

Reproduced for this task from
dev:oracle/t-20261102-233913-провести-работы-по-код-ревью-про-352534/test_review_contract.py
(the dev copy is not present in this worktree).

Path resolution is cwd-independent: the repo root is derived from __file__.
"""

from __future__ import annotations

import ast
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]  # repo root: <root>/.factory/integration/oracle/<ID>/file
SKILLS = ROOT / "skills"

MACHINE_PATHS = ("/media/", "/home/", "Desktop/projects", "C:\\")
SKILL_SUFFIXES = {".md", ".py", ".sh", ".json", ".txt", ".yaml", ".yml"}
PY_SCAN_SKIP = {".git", ".factory", "node_modules", "venv", ".venv", "app", "oracle", "__pycache__"}


def read(rel: str) -> str:
    p = ROOT / rel
    assert p.is_file(), f"missing file: {rel}"
    return p.read_text(encoding="utf-8", errors="replace")


def skill_texts() -> list[tuple[str, str]]:
    assert SKILLS.is_dir(), "skills/ is missing"
    out = []
    for p in sorted(SKILLS.rglob("*")):
        if p.is_file() and p.suffix in SKILL_SUFFIXES:
            out.append((p.relative_to(ROOT).as_posix(), p.read_text(encoding="utf-8", errors="replace")))
    assert out, "no skill files found under skills/"
    return out


def python_files() -> list[tuple[str, str]]:
    out = []
    for p in sorted(ROOT.rglob("*.py")):
        if any(part in PY_SCAN_SKIP for part in p.relative_to(ROOT).parts):
            continue
        out.append((p.relative_to(ROOT).as_posix(), p.read_text(encoding="utf-8", errors="replace")))
    return out


def funcs(src: str, name: str) -> list[ast.FunctionDef]:
    return [n for n in ast.walk(ast.parse(src)) if isinstance(n, ast.FunctionDef) and n.name == name]


# ---------------------------------------------------------------- item 1
def test_01_single_release_scorer():
    planner = read("rnext/planner.py")
    assert funcs(planner, "score_facts"), "rnext/planner.py must define score_facts"

    definers = [rel for rel, src in python_files() if funcs(src, "score_facts")]
    assert definers == ["rnext/planner.py"], (
        "score_facts must be defined exactly once, in rnext/planner.py; found: %s" % definers
    )

    hunt = read("scripts/tracker_hunt.py")
    assert "score_facts" in hunt, "scripts/tracker_hunt.py must delegate to rnext.planner.score_facts"

    defs = funcs(hunt, "score")
    assert defs, "scripts/tracker_hunt.py must keep a score() entry point"
    body = [s for s in defs[0].body
            if not (isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant) and isinstance(s.value.value, str))]
    assert len(body) <= 5, (
        "tracker_hunt.score must be a thin delegation, found %d statements" % len(body)
    )
    numbers = [n for n in ast.walk(defs[0])
               if isinstance(n, ast.Constant) and isinstance(n.value, (int, float))
               and not isinstance(n.value, bool)]
    assert len(numbers) <= 2, (
        "tracker_hunt.score still carries its own weight table (%d numeric literals)" % len(numbers)
    )


# ---------------------------------------------------------------- item 2
def test_02_no_machine_paths_in_skills():
    bad = [(rel, mp) for rel, src in skill_texts() for mp in MACHINE_PATHS if mp in src]
    assert not bad, "machine-specific paths in skills: %s" % sorted(bad)[:6]


# ---------------------------------------------------------------- item 3
REF = re.compile(r"(?<![\w/.-])((?:scripts|research|rnext|config|tests)/[A-Za-z0-9_][A-Za-z0-9_./-]*)")
REF_EXCEPTION = "config/workspace.json"


def test_03_skill_references_exist():
    missing = []
    for rel, src in skill_texts():
        for match in REF.finditer(src):
            ref = match.group(1).rstrip(".,:;)?!»\"'")
            ref = re.sub(r":\d+$", "", ref)
            if any(ch in ref for ch in "<>*$"):
                continue
            if ref == REF_EXCEPTION:
                continue
            if not (ROOT / ref).exists():
                missing.append((rel, ref))
    assert not missing, "skills point at non-existent paths: %s" % sorted(set(missing))[:10]


# ---------------------------------------------------------------- item 4
def test_04_qc_gates_optional_orig():
    src = read("scripts/qc_gates.py")
    assert re.search(r"""['"]--orig['"]""", src), "qc_gates.py must expose an optional --orig"
    assert "spike" in src.lower(), "the spike gate must not be removed"
    assert "orig" in src.lower(), "qc_gates.py must actually use the source (orig) take"


# ---------------------------------------------------------------- item 5
def test_05_parser_tests_on_fixtures():
    tests_dir = ROOT / "tests"
    assert tests_dir.is_dir(), "tests/ is missing"
    py = [p for p in tests_dir.rglob("*.py") if p.is_file()]
    assert py, "tests/ has no python tests"
    blob = "\n".join(p.read_text(encoding="utf-8", errors="replace") for p in py)
    for name in ("parse_music", "parse_video", "parse_topic_page", "parse_dr_table"):
        assert name in blob, "no test covers %s" % name
    fixtures = [p for p in tests_dir.rglob("*.html") if p.is_file()]
    assert fixtures, "no HTML fixtures under tests/**"


# ---------------------------------------------------------------- item 6
def test_06_setup_documents_cookies_and_refresh():
    src = read("docs/setup.md")
    assert "cf_clearance" in src, "docs/setup.md must mention cf_clearance"
    assert "scripts/rt_browser_refresh.py" in src, "docs/setup.md must mention scripts/rt_browser_refresh.py"
    assert "--help" in src, "docs/setup.md must show how to read the refresh script flags (--help)"
    assert (ROOT / "scripts" / "rt_browser_refresh.py").is_file(), "scripts/rt_browser_refresh.py must exist"


# ---------------------------------------------------------------- item 7
def test_07_release_batch_marked_deprecated():
    src = read("scripts/release_batch.py").lower()
    assert "deprecat" in src or "legacy" in src, "scripts/release_batch.py must be marked deprecated/legacy"


# ---------------------------------------------------------------- item 8
def test_08_coding_standards_present():
    p = ROOT / "CODING_STANDARDS.md"
    assert p.is_file(), "CODING_STANDARDS.md must exist at the repo root"
    assert len(p.read_text(encoding="utf-8", errors="replace").strip()) > 200, "CODING_STANDARDS.md is a stub"


# ---------------------------------------------------------------- item 9
def test_09_ci_runs_pytest():
    wf_dir = ROOT / ".github" / "workflows"
    assert wf_dir.is_dir(), ".github/workflows/ is missing"
    workflows = [p for p in sorted(wf_dir.iterdir()) if p.suffix in {".yml", ".yaml"}]
    assert workflows, "no workflow files in .github/workflows/"
    hits = [p.name for p in workflows if "pytest" in p.read_text(encoding="utf-8", errors="replace")]
    assert hits, "no CI workflow runs pytest"


# ---------------------------------------------------------------- item 10
def test_10_workspace_example_is_portable_json():
    rel = "config/workspace.example.json"
    raw = read(rel)
    try:
        json.loads(raw)
    except json.JSONDecodeError as exc:
        raise AssertionError(f"{rel} is not valid JSON: {exc}") from exc
    assert "/media/" not in raw and "/home/" not in raw, f"{rel} still holds machine paths"


# ---------------------------------------------------------------- item 11
def test_11_version_at_least_2_1():
    src = read("pyproject.toml")
    m = re.search(r'^\s*version\s*=\s*"([^"]+)"\s*$', src, re.MULTILINE)
    assert m, "pyproject.toml has no version line"
    parts = []
    for chunk in m.group(1).split("."):
        digits = re.match(r"\d+", chunk)
        parts.append(int(digits.group(0)) if digits else 0)
    while len(parts) < 2:
        parts.append(0)
    assert tuple(parts[:2]) >= (2, 1), f"pyproject.toml version is {m.group(1)}, expected >= 2.1"


# ---------------------------------------------------------------- item 12
def frontmatter(text: str) -> str | None:
    if not text.startswith("---"):
        return None
    end = text.find("\n---", 3)
    if end < 0:
        return None
    return text[3:end]


def test_12_skill_frontmatter_quoted_and_review_kept():
    broken = []
    seen = 0
    for rel, src in skill_texts():
        if not rel.endswith("SKILL.md"):
            continue
        seen += 1
        fm = frontmatter(src)
        assert fm is not None, f"{rel} has no YAML frontmatter"
        for lineno, line in enumerate(fm.splitlines(), 1):
            m = re.match(r"^([A-Za-z_][A-Za-z0-9_-]*):\s*(.*)$", line)
            if not m:
                continue
            val = m.group(2).strip()
            if not val or val[0] in "\"'[{>|":
                continue
            if ":" in val:
                broken.append((rel, lineno, line.strip()))
    assert seen, "no SKILL.md found under skills/"
    assert not broken, "unquoted YAML scalars containing ':': %s" % broken[:5]
    assert (ROOT / "docs" / "code-review-2026-11-02.md").is_file(), (
        "docs/code-review-2026-11-02.md must stay in the repo"
    )


# ---------------------------------------------------------------- guard
def test_13_repo_tests_pass():
    """The rest of the suite must stay green: the pipeline scripts are live."""
    tests_dir = ROOT / "tests"
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", str(tests_dir), "-q"],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, "repo tests failed:\n%s\n%s" % (proc.stdout[-3000:], proc.stderr[-2000:])
