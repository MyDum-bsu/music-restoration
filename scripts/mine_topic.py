#!/usr/bin/env python3
"""Parse a restoration topic into per-release JSON."""
import json
import sys

from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rnext.rtclient import RTClient  # noqa: E402
from rnext.rttopic import fetch_topic  # noqa: E402


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: mine_topic.py <topic_id>", file=sys.stderr)
        return 1
    d = fetch_topic(RTClient(), sys.argv[1])
    print(json.dumps(d, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
