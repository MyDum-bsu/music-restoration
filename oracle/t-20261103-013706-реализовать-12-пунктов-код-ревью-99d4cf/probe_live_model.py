#!/usr/bin/env python3
"""Live-model probe for t-20261103-013706-реализовать-12-пунктов-код-ревью-99d4cf.

Second entry point to the same contract, usable without pytest. It groups the
twelve checks into the invariants the spec promises and prints one line per
invariant. Exit code 1 when any invariant fails.

It imports the checks from test_review_contract.py instead of re-implementing
them: two entry points, one definition of the criterion.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import test_review_contract as contract  # noqa: E402

INVARIANTS: dict[str, tuple[str, ...]] = {
    "I1_single_scorer": ("single_release_scorer",),
    "I2_no_machine_paths": (
        "no_machine_paths_in_skills",
        "skill_referenced_paths_exist",
        "portable_workspace_template",
    ),
    "I3_pipeline_untouched": ("qc_gates_orig_is_optional", "release_batch_is_deprecated"),
    "I4_agent_critical_code_is_tested": ("parser_tests_on_fixtures", "ci_runs_pytest"),
    "I5_documented_entry_point": ("documented_entry_point",),
    "I6_single_source_of_truth": ("version_at_least_2_1", "coding_standards_doc", "skill_frontmatter_is_valid_yaml"),
}


def main() -> int:
    root = contract.repo_root()
    by_name = {fn.__name__: fn for fn in contract.CHECKS}

    missing = [name for names in INVARIANTS.values() for name in names if name not in by_name]
    if missing:
        print(f"INVARIANT map is stale, unknown checks: {missing}", file=sys.stderr)
        return 2

    failed = False
    for invariant, names in INVARIANTS.items():
        problems: list[str] = []
        for name in names:
            problems.extend(by_name[name](root))
        if problems:
            failed = True
            print(f"INVARIANT {invariant}: FAIL")
            for problem in problems:
                print(f"  - {problem}")
        else:
            print(f"INVARIANT {invariant}: PASS")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
