#!/usr/bin/env bash
# music-restoration installer: dependency check + agent-skills installation.
#
#   ./install.sh                 # check deps, symlink skills into ~/.agents/skills
#   ./install.sh --claude        # symlink into ~/.claude/skills instead
#   ./install.sh --skills-dir D  # symlink into a custom dir
#   ./install.sh --copy          # copy instead of symlink (updates need re-run)
#   ./install.sh --venv          # also create .venv and pip-install requirements
#
# Idempotent: re-run any time. Never touches your music files.
set -euo pipefail
cd "$(dirname "$0")"
REPO="$(pwd)"

SKILLS_DIR="${HOME}/.agents/skills"
MODE=symlink
VENV=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --claude) SKILLS_DIR="${HOME}/.claude/skills"; shift ;;
    --skills-dir) SKILLS_DIR="$2"; shift 2 ;;
    --copy) MODE=copy; shift ;;
    --venv) VENV=1; shift ;;
    *) echo "unknown option: $1"; exit 1 ;;
  esac
done

fail=0
say() { printf '%s\n' "$*"; }
have() { command -v "$1" >/dev/null 2>&1; }

say "== dependencies =="
for cmd in ffmpeg ffprobe flac metaflac curl gio; do
  if have "$cmd"; then say "  ok      $cmd"; else say "  MISSING $cmd  (install it with your package manager)"; fail=1; fi
done
if python3 -c "import numpy, scipy, matplotlib, mutagen" 2>/dev/null; then
  say "  ok      python3 + numpy/scipy/matplotlib/mutagen"
else
  say "  MISSING python3 deps: pip install -r requirements.txt (or use --venv)"
  fail=1
fi

if [[ -x "restore-dr/bin/stereo_tool_cmd_64" ]]; then
  say "  ok      Stereo Tool CLI (restore-dr/bin)"
else
  say "  NOTE    Stereo Tool CLI not found — ND2 punch stage disabled until you add it."
  say "          Free for non-commercial: https://www.stereo-tool.com -> command line"
  say "          version -> put binary into restore-dr/bin/stereo_tool_cmd_64."
  say "          Everything else (diagnose/measure/contour pipeline) works without it."
fi

if [[ $VENV -eq 1 ]]; then
  python3 -m venv .venv
  # shellcheck disable=SC1091
  source .venv/bin/activate
  pip install -q -r requirements.txt
  say "  ok      .venv created (activate it, or run scripts with .venv/bin/python3)"
fi

say "== config =="
CFG_DIR="${HOME}/.config/music-restoration"
CFG="${CFG_DIR}/config.env"
mkdir -p "$CFG_DIR"
if [[ -f "$CFG" ]]; then
  say "  ok      $CFG exists"
else
  cp config.env.example "$CFG"
  say "  created $CFG (all keys optional — phone and ~/Music are auto-detected)"
fi

say "== agent skills (${MODE} -> ${SKILLS_DIR}) =="
mkdir -p "$SKILLS_DIR"
for s in restore-dr add-lyrics phone-music split-flac-album; do
  target="${SKILLS_DIR}/${s}"
  if [[ -e "$target" && ! -L "$target" ]]; then
    say "  SKIP    $target exists and is not a symlink — move it away first if you want $MODE"
    continue
  fi
  rm -rf "$target"
  if [[ $MODE == symlink ]]; then
    ln -s "${REPO}/${s}" "$target"
  else
    cp -r "${REPO}/${s}" "$target"
  fi
  say "  ok      $s"
done

say
if [[ $fail -eq 0 ]]; then
  say "All set. Try: restore-dr/scripts/measure.py metrics <file.flac>"
else
  say "Some dependencies are missing — fix them, then re-run ./install.sh"
  exit 1
fi
