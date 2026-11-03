"""Contract oracle for t-20261103-013706-реализовать-12-пунктов-код-ревью-99d4cf.

Twelve checks, one per item of docs/code-review-2026-11-02.md.
Deterministic: no network, no tracker, no /media, no writes outside tmp.
Each check returns a list of human-readable problems; empty list means pass.
"""

from __future__ import annotations

import ast
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

MACHINE_PATH_RE = re.compile(r"(/media/|/home/[A-Za-z0-9_.-]+/|Desktop/projects)")

SKILL_REF_RE = re.compile(
    r"(?<![\w/.-])"
    r"((?:scripts|research|rnext|config|tests)/[A-Za-z0-9_][A-Za-z0-9_./-]*"
    r"\.(?:py|json|md|sh|yaml|yml))"
)

CHECKS: list = []


def check(fn):
    CHECKS.append(fn)
    return fn


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


def repo_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").is_file() and (parent / "skills").is_dir():
            return parent
    raise RuntimeError("repository root not found above the oracle file")


def skill_files(root: Path) -> list[Path]:
    return sorted((root / "skills").rglob("SKILL.md"))


def frontmatter_block(text: str, path: Path) -> str:
    if not text.startswith("---"):
        raise AssertionError(f"{path}: does not start with a YAML frontmatter block")
    end = text.find("\n---", 3)
    if end == -1:
        raise AssertionError(f"{path}: frontmatter block is not closed")
    return text[3:end]


def run_help(root: Path, script: str) -> tuple[int, str]:
    proc = subprocess.run(
        [sys.executable, script, "--help"],
        cwd=str(root),
        capture_output=True,
        text=True,
        timeout=180,
    )
    return proc.returncode, (proc.stdout or "") + (proc.stderr or "")


def project_version(root: Path) -> tuple[int, ...]:
    raw = (root / "pyproject.toml").read_text(encoding="utf-8")
    try:
        import tomllib
    except ModuleNotFoundError:  # python < 3.11
        match = re.search(r'(?m)^version\s*=\s*"([^"]+)"', raw)
        if match is None:
            raise AssertionError("pyproject.toml: no version line")
        value = match.group(1)
    else:
        value = tomllib.loads(raw)["project"]["version"]
    return tuple(int(part) for part in re.findall(r"\d+", value))


# --------------------------------------------------------------------------- #
# 1. Standards #1 — one release scorer
# --------------------------------------------------------------------------- #


@check
def single_release_scorer(root: Path) -> list[str]:
    problems: list[str] = []
    planner = root / "rnext" / "planner.py"
    hunt = root / "scripts" / "tracker_hunt.py"
    if not planner.is_file():
        problems.append("missing rnext/planner.py")
    if not hunt.is_file():
        problems.append("missing scripts/tracker_hunt.py")
    if problems:
        return problems

    planner_text = planner.read_text(encoding="utf-8")
    hunt_text = hunt.read_text(encoding="utf-8")

    if re.search(r"(?m)^def score_facts\(", planner_text) is None:
        problems.append("rnext/planner.py has no module-level `def score_facts(`")

    if re.search(r"(?m)^def score_candidate\(", hunt_text) is not None:
        problems.append("scripts/tracker_hunt.py still defines its own score_candidate()")

    tree = ast.parse(hunt_text)
    score_fn = next(
        (node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "score"),
        None,
    )
    if score_fn is None:
        problems.append("scripts/tracker_hunt.py has no score() function")
    else:
        called = {c.func.id for c in ast.walk(score_fn) if isinstance(c, ast.Call) and isinstance(c.func, ast.Name)}
        called |= {c.func.attr for c in ast.walk(score_fn) if isinstance(c, ast.Call) and isinstance(c.func, ast.Attribute)}
        if "score_facts" not in called:
            problems.append("scripts/tracker_hunt.score() does not call score_facts()")
    return problems


# --------------------------------------------------------------------------- #
# 2. Standards #2/#6 — no machine paths in skills
# --------------------------------------------------------------------------- #


@check
def no_machine_paths_in_skills(root: Path) -> list[str]:
    problems: list[str] = []
    for path in skill_files(root):
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            match = MACHINE_PATH_RE.search(line)
            if match is not None:
                problems.append(
                    f"{path.relative_to(root)}:{lineno}: machine path `{match.group(0)}`"
                )
    return problems


# --------------------------------------------------------------------------- #
# 3. Standards #3 — every repo path mentioned in a skill exists
# --------------------------------------------------------------------------- #


@check
def skill_referenced_paths_exist(root: Path) -> list[str]:
    problems: list[str] = []
    for path in skill_files(root):
        text = path.read_text(encoding="utf-8")
        for match in SKILL_REF_RE.finditer(text):
            ref = match.group(1)
            if any(token in ref for token in ("*", "<", ">", "{", "}")):
                continue
            if not (root / ref).exists():
                problems.append(f"{path.relative_to(root)}: references missing `{ref}`")
    return problems


# --------------------------------------------------------------------------- #
# 4. Standards #5 — qc_gates --orig is optional, spike gate still present
# --------------------------------------------------------------------------- #


@check
def qc_gates_orig_is_optional(root: Path) -> list[str]:
    problems: list[str] = []
    script = root / "scripts" / "qc_gates.py"
    if not script.is_file():
        return ["missing scripts/qc_gates.py"]

    text = script.read_text(encoding="utf-8")
    if "--orig" not in text:
        problems.append("scripts/qc_gates.py has no --orig flag")
    if re.search(r"spike", text, re.IGNORECASE) is None:
        problems.append("scripts/qc_gates.py lost the spike gate")

    code, output = run_help(root, "scripts/qc_gates.py")
    if code != 0:
        problems.append(f"`qc_gates.py --help` exited {code} (the new flag must stay optional)")
    elif "--orig" not in output:
        problems.append("`qc_gates.py --help` does not list --orig")

    spike_tests = root / "tests" / "test_qc_gates_spikes.py"
    if not spike_tests.is_file():
        problems.append("missing tests/test_qc_gates_spikes.py (behaviour of --orig vs source spikes)")
    return problems


# --------------------------------------------------------------------------- #
# 5. Spec #2 — parser tests on saved HTML fixtures
# --------------------------------------------------------------------------- #


@check
def parser_tests_on_fixtures(root: Path) -> list[str]:
    problems: list[str] = []
    test_file = root / "tests" / "test_parsers.py"
    if not test_file.is_file():
        return ["missing tests/test_parsers.py"]

    text = test_file.read_text(encoding="utf-8")
    for name in ("parse_music", "parse_video", "parse_topic_page", "parse_dr_table"):
        if name not in text:
            problems.append(f"tests/test_parsers.py does not exercise {name}")

    fixtures = root / "tests" / "fixtures"
    html = sorted(fixtures.rglob("*.html")) if fixtures.is_dir() else []
    if len(html) < 4:
        problems.append(f"tests/fixtures holds {len(html)} html fixtures, need at least 4")
    return problems


# --------------------------------------------------------------------------- #
# 6. Spec #1 — documented entry point: setup docs + cookie refresh script
# --------------------------------------------------------------------------- #


@check
def documented_entry_point(root: Path) -> list[str]:
    problems: list[str] = []
    setup = root / "docs" / "setup.md"
    if not setup.is_file():
        return ["missing docs/setup.md"]

    text = setup.read_text(encoding="utf-8")
    if "cf_clearance" not in text:
        problems.append("docs/setup.md does not mention cf_clearance")
    if "rt_browser_refresh" not in text:
        problems.append("docs/setup.md does not point at scripts/rt_browser_refresh.py")

    script = root / "scripts" / "rt_browser_refresh.py"
    if not script.is_file():
        problems.append("missing scripts/rt_browser_refresh.py")
    else:
        code, output = run_help(root, "scripts/rt_browser_refresh.py")
        if code != 0:
            problems.append(f"`rt_browser_refresh.py --help` exited {code}: {output.strip()[:200]}")
    return problems


# --------------------------------------------------------------------------- #
# 7. Standards #4 — release_batch.py is deprecated
# --------------------------------------------------------------------------- #


@check
def release_batch_is_deprecated(root: Path) -> list[str]:
    script = root / "scripts" / "release_batch.py"
    if not script.is_file():
        return ["missing scripts/release_batch.py"]

    problems: list[str] = []
    text = script.read_text(encoding="utf-8")
    if re.search(r"deprecat", text, re.IGNORECASE) is None:
        problems.append("scripts/release_batch.py does not say it is deprecated")
    if "restore_per_track" not in text:
        problems.append("scripts/release_batch.py does not name its replacement (restore_per_track.py)")
    code, _ = run_help(root, "scripts/release_batch.py")
    if code != 0:
        problems.append(f"`release_batch.py --help` exited {code}")
    return problems


# --------------------------------------------------------------------------- #
# 8. Spec #4 — CODING_STANDARDS.md
# --------------------------------------------------------------------------- #


@check
def coding_standards_doc(root: Path) -> list[str]:
    doc = root / "CODING_STANDARDS.md"
    if not doc.is_file():
        return ["missing CODING_STANDARDS.md"]
    text = doc.read_text(encoding="utf-8")
    problems: list[str] = []
    if len(text) < 800:
        problems.append(f"CODING_STANDARDS.md is {len(text)} chars, too thin to be a standard")
    if "pytest" not in text and "tests/" not in text:
        problems.append("CODING_STANDARDS.md says nothing about tests")
    return problems


# --------------------------------------------------------------------------- #
# 9. Spec #3 — CI runs pytest
# --------------------------------------------------------------------------- #


@check
def ci_runs_pytest(root: Path) -> list[str]:
    workflows = root / ".github" / "workflows"
    files = sorted(p for p in workflows.glob("*") if p.suffix in {".yml", ".yaml"}) if workflows.is_dir() else []
    if not files:
        return ["missing .github/workflows/*.yml"]
    problems: list[str] = []
    for path in files:
        text = path.read_text(encoding="utf-8")
        if "pytest" in text:
            return []
        problems.append(f"{path.relative_to(root)} does not run pytest")
    return problems


# --------------------------------------------------------------------------- #
# 10. Spec #1 — portable workspace template
# --------------------------------------------------------------------------- #


@check
def portable_workspace_template(root: Path) -> list[str]:
    example = root / "config" / "workspace.example.json"
    if not example.is_file():
        return ["missing config/workspace.example.json"]

    problems: list[str] = []
    text = example.read_text(encoding="utf-8")
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        problems.append(f"config/workspace.example.json is not valid JSON: {exc}")
        data = None

    if isinstance(data, dict) and not data:
        problems.append("config/workspace.example.json is an empty object")

    for lineno, line in enumerate(text.splitlines(), start=1):
        match = MACHINE_PATH_RE.search(line)
        if match is not None:
            problems.append(
                f"config/workspace.example.json:{lineno}: machine path `{match.group(0)}`"
            )

    docs = [root / "docs" / "setup.md", root / "README.md"]
    mentioned = any(
        doc.is_file() and "workspace.example.json" in doc.read_text(encoding="utf-8")
        for doc in docs
    )
    if not mentioned:
        problems.append("config/workspace.example.json is not mentioned in docs/setup.md or README.md")
    return problems


# --------------------------------------------------------------------------- #
# 11. Standards #7 — version is 2.1 or newer
# --------------------------------------------------------------------------- #


@check
def version_at_least_2_1(root: Path) -> list[str]:
    version = project_version(root)
    if version < (2, 1):
        return [f"pyproject.toml version is {'.'.join(map(str, version))}, need >= 2.1"]
    return []


# --------------------------------------------------------------------------- #
# 12. Spec #5 — every skill frontmatter parses
# --------------------------------------------------------------------------- #


@check
def skill_frontmatter_is_valid_yaml(root: Path) -> list[str]:
    problems: list[str] = []
    try:
        import yaml
    except ModuleNotFoundError:
        yaml = None

    for path in skill_files(root):
        text = path.read_text(encoding="utf-8")
        try:
            block = frontmatter_block(text, path)
        except AssertionError as exc:
            problems.append(str(exc))
            continue

        if yaml is not None:
            try:
                data = yaml.safe_load(block)
            except yaml.YAMLError as exc:
                problems.append(f"{path.relative_to(root)}: frontmatter is not valid YAML: {exc}")
                continue
            if not isinstance(data, dict):
                problems.append(f"{path.relative_to(root)}: frontmatter is not a mapping")
            continue

        # No PyYAML installed: validate the shape instead of parsing it.
        for lineno, line in enumerate(block.splitlines(), start=2):
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            if line != line.lstrip() or "\t" in line:
                problems.append(f"{path.relative_to(root)}:{lineno}: unexpected indent or tab")
                continue
            if ":" not in line:
                problems.append(f"{path.relative_to(root)}:{lineno}: no `key:` in frontmatter line")
                continue
            key, value = line.split(":", 1)
            if not key.strip() or key.strip() != key:
                problems.append(f"{path.relative_to(root)}:{lineno}: malformed key")
                continue
            value = value.strip()
            if value and value[0] not in "\"'[{|>&*" and ": " in value:
                problems.append(
                    f"{path.relative_to(root)}:{lineno}: unquoted `: ` in value of `{key.strip()}`"
                )
    return problems


# --------------------------------------------------------------------------- #
# pytest binding
# --------------------------------------------------------------------------- #


def test_contract_has_twelve_checks() -> None:
    assert len(CHECKS) == 12, [fn.__name__ for fn in CHECKS]


@pytest.mark.parametrize("contract_check", CHECKS, ids=[fn.__name__ for fn in CHECKS])
def test_contract(contract_check) -> None:
    problems = contract_check(repo_root())
    assert problems == [], "\n".join(problems)
