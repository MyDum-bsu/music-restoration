"""Oracle for t-20261102-231434: the code review report exists and is a real review.

The deliverable is one document: `docs/code-review-2026-11-02.md`. It must cover
every item of the raw request (CR-01..CR-12) with a verdict, evidence taken from
the repository (path:line) and a decomposition row. A file that merely agrees
with the request is not accepted: `done` verdicts are cross-checked against the
filesystem, and verdicts are cross-checked between Findings and Decomposition.

Run from the repository root:
    python3 -m pytest .factory/integration/oracle/<ID>/test_review_report.py -q
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

REPORT_REL = "docs/code-review-2026-11-02.md"

# Contract: one line per request item, in the order the request lists them.
ITEMS = {
    "CR-01": "HIGH: merge the two release scorers (rnext/planner.py score_candidate vs scripts/tracker_hunt.py score) into rnext",
    "CR-02": "HIGH: remove machine paths from SKILL.md, use rnext.workspace.data_root and config/workspace.json",
    "CR-03": "MEDIUM: re-check every path in SKILL.md (status_report moved to research/)",
    "CR-04": "MEDIUM: qc_gates.py --orig mode for spikes (fail only when spikes in output > spikes in source)",
    "CR-05": "MEDIUM: tests for parse_music/parse_video/parse_topic_page/parse_dr_table on HTML fixtures",
    "CR-06": "MEDIUM: docs/setup.md + scripts/rt_browser_refresh.py (cf_clearance refresh through a browser)",
    "CR-07": "MEDIUM: deprecate release_batch.py",
    "CR-08": "LOW: CODING_STANDARDS.md",
    "CR-09": "LOW: CI pytest",
    "CR-10": "LOW: config/workspace.example.json",
    "CR-11": "LOW: bump the version",
    "CR-12": "LOW: commit the YAML quoting fix in the 2 SKILL.md files",
}

VERDICTS = {"done", "open", "needs-spec", "rejected"}

# A "done" verdict for these items is only honest if the named artifacts exist.
# Items that add a file have no path:line to cite, so the file itself is the
# evidence. The rest (CR-01..CR-04, CR-07, CR-09, CR-11, CR-12) must still cite
# an existing path, but the oracle does not guess which one.
DONE_REQUIRES_EXISTS = {
    "CR-05": ["tests/"],
    "CR-06": ["docs/setup.md", "scripts/rt_browser_refresh.py"],
    "CR-08": ["CODING_STANDARDS.md"],
    "CR-10": ["config/workspace.example.json"],
}

# The background restoration pipeline is running on this machine right now.
PROTECTED = [
    "scripts/restore_per_track.py",
    "scripts/finish_release.py",
    "scripts/qc_gates.py",
    "/media/mydum/Data",
]

REQUIRED_SECTIONS = ["Summary", "Findings", "Decomposition", "Constraints"]

PATH_RE = re.compile(
    r"`?([A-Za-z0-9_./-]+\.(?:py|md|json|ya?ml|toml|cfg|ini|sh|txt))(?::(\d+))?`?"
)
PLACEHOLDERS = ("TODO", "TBD", "FIXME", "XXX", "???")
MIN_CHARS = 3000


def repo_root() -> Path:
    here = Path(__file__).resolve()
    for p in [here.parent, *here.parents]:
        if (p / "scripts").is_dir() and (p / "rnext").is_dir():
            return p
    pytest.fail("repository root not found: no ancestor has both scripts/ and rnext/")


def l2_section(text: str, name: str) -> str:
    """Body of the `## ... <name> ...` section, up to the next level-1/2 heading."""
    m = re.search(rf"^##[^\n]*\b{name}\b[^\n]*$", text, re.M | re.I)
    if not m:
        pytest.fail(f"report has no '## {name}' section")
    rest = text[m.end():]
    end = re.search(r"^#{1,2}\s", rest, re.M)
    return rest[: end.start()] if end else rest


def item_sections(text: str) -> tuple[dict[str, str], list[str]]:
    """Split the report into `### CR-NN` blocks."""
    bodies: dict[str, list[str]] = {}
    order: list[str] = []
    current: str | None = None
    for line in text.splitlines():
        m = re.match(r"^###\s+(CR-\d{2})\b", line)
        if m:
            current = m.group(1)
            order.append(current)
            bodies.setdefault(current, [])
            continue
        if current is not None and re.match(r"^#{1,3}\s", line):
            current = None
            continue
        if current is not None:
            bodies[current].append(line)
    return {k: "\n".join(v) for k, v in bodies.items()}, order


def verdict_of(body: str) -> str | None:
    m = re.search(r"\bVerdict:\s*`?\s*([A-Za-z-]+)", body, re.I)
    if not m:
        return None
    return m.group(1).strip().lower().replace(" ", "-")


def cited_paths(body: str) -> list[tuple[str, str | None]]:
    return [(a, b) for a, b in PATH_RE.findall(body)]


@pytest.fixture(scope="module")
def root() -> Path:
    return repo_root()


@pytest.fixture(scope="module")
def text(root: Path) -> str:
    path = root / REPORT_REL
    assert path.is_file(), f"missing review report: {REPORT_REL}"
    return path.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def items(text: str) -> tuple[dict[str, str], list[str]]:
    return item_sections(text)


def test_report_is_not_a_stub(text: str) -> None:
    assert len(text) >= MIN_CHARS, (
        f"report is {len(text)} chars, a real review of 12 items needs >= {MIN_CHARS}"
    )
    assert re.search(r"^#\s+\S", text, re.M), "report has no H1 title"
    upper = text.upper()
    for word in PLACEHOLDERS:
        assert word not in upper, f"report still contains the placeholder {word!r}"


def test_required_sections(text: str) -> None:
    found = [m.group(1).strip() for m in re.finditer(r"^##\s+([^\n]+)$", text, re.M)]
    missing = [
        name
        for name in REQUIRED_SECTIONS
        if not any(re.search(rf"\b{name}\b", h, re.I) for h in found)
    ]
    assert not missing, f"report is missing sections: {missing}; found: {found}"


def test_every_item_has_exactly_one_section(items) -> None:
    bodies, order = items
    missing = [k for k in ITEMS if k not in bodies]
    extra = [k for k in bodies if k not in ITEMS]
    dupes = sorted({k for k in order if order.count(k) > 1})
    assert not missing, f"report does not cover: {missing}"
    assert not extra, f"report invented item ids that are not in the request: {extra}"
    assert not dupes, f"item covered more than once: {dupes}"


def test_verdict_and_evidence_per_item(root: Path, items) -> None:
    bodies, _ = items
    problems: list[str] = []
    for key in ITEMS:
        body = bodies.get(key, "")
        verdict = verdict_of(body)
        if verdict not in VERDICTS:
            problems.append(f"{key}: verdict {verdict!r} not in {sorted(VERDICTS)}")
            continue
        paths = cited_paths(body)
        if not paths:
            problems.append(f"{key}: no repository path cited (expected `path`)")
            continue
        if not any((root / rel).exists() for rel, _ in paths):
            problems.append(f"{key}: none of the cited paths exist: {[r for r, _ in paths][:4]}")
        if verdict == "done":
            with_line = [rel for rel, line in paths if line and (root / rel).exists()]
            needs = DONE_REQUIRES_EXISTS.get(key, [])
            if not needs and not with_line:
                problems.append(
                    f"{key}: verdict done needs evidence `path:line`, got only file names"
                )
            for rel in needs:
                if not (root / rel).exists():
                    problems.append(f"{key}: verdict done but {rel} does not exist")
    assert not problems, "evidence problems:\n" + "\n".join(problems)


def test_completed_items_point_at_existing_artifacts(root: Path, items) -> None:
    bodies, _ = items
    problems: list[str] = []
    for key, rels in DONE_REQUIRES_EXISTS.items():
        if verdict_of(bodies.get(key, "")) != "done":
            continue
        for rel in rels:
            if not (root / rel).exists():
                problems.append(f"{key} says done while {rel} is absent")
    assert not problems, "; ".join(problems)


def test_decomposition_matches_findings(text: str, items) -> None:
    bodies, _ = items
    table = l2_section(text, "Decomposition")
    rows: dict[str, list[str]] = {}
    header_seen = False
    for line in table.splitlines():
        if not line.strip().startswith("|"):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if not header_seen:
            header_seen = True
            assert any("item" in c.lower() for c in cells), (
                f"decomposition table has no Item column: {cells}"
            )
            assert any("task" in c.lower() for c in cells), (
                f"decomposition table has no Task column: {cells}"
            )
            continue
        if all(set(c) <= set("-: ") for c in cells):
            continue
        mid = re.match(r"^(CR-\d{2})$", cells[0].strip("`* "))
        if mid:
            rows[mid.group(1)] = cells
    missing = [k for k in ITEMS if k not in rows]
    assert not missing, f"decomposition table has no row for: {missing}"
    for key, cells in rows.items():
        assert len(cells) >= 3, f"{key}: decomposition row has {len(cells)} cells: {cells}"
        cell_verdict = cells[1].strip("`* ").lower().replace(" ", "-")
        found = verdict_of(bodies.get(key, ""))
        assert cell_verdict == found, (
            f"{key}: verdict differs — Findings says {found!r}, "
            f"Decomposition says {cells[1]!r}"
        )
        task = cells[2].strip("`* ")
        assert task and task.lower() not in ("-", "—", "n/a"), (
            f"{key}: decomposition row names no follow-up task"
        )


def test_constraints_section_names_the_live_pipeline(text: str) -> None:
    body = l2_section(text, "Constraints")
    missing = [p for p in PROTECTED if p not in body]
    assert not missing, (
        f"Constraints does not say what must stay untouched: {missing}"
    )
