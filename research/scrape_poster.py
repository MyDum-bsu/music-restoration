#!/usr/bin/env python3
"""Registry of master-profile topics into corpus/master_topics.jsonl."""
import argparse
import sys

from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rnext import workspace  # noqa: E402
from rnext.rtscrape import poster_topics  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("pid")
    ap.add_argument("--append", action="store_true", help="append instead of overwriting")
    args = ap.parse_args()

    rows = poster_topics(args.pid)
    # exclude the pinned site-service topic whose title is the Russian word
    # for "Help"; kept as an escape so the filter keeps matching site data
    help_topic = "\u041f\u043e\u043c\u043e\u0449\u044c"
    rows = [r for r in rows if r.get("title") and not r["title"].startswith(help_topic)]
    rows.sort(key=lambda r: r["topic_id"])

    path = workspace.dir_of("corpus") / "master_topics.jsonl"
    import json
    mode = "a" if args.append else "w"
    known = set()
    if args.append and path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                known.add(json.loads(line).get("topic_id"))
    with open(path, mode, encoding="utf-8") as f:
        n = 0
        for r in rows:
            if r["topic_id"] in known:
                continue
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    total = sum(1 for _ in open(path, encoding="utf-8"))
    print(f"+{n} topics -> {path} (total {total})")
    print(f"total size added: {sum((r.get('size_bytes') or 0) for r in rows)/2**30:.2f} GiB")
    for r in rows:
        print(f"  t={r['topic_id']:>7} {r['size_label'] or '?':>9} s{r['seeders']:<4} {r['title'][:96]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
