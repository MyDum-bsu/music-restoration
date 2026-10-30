#!/usr/bin/env bash
# music-restoration installer: dependency check + agent-skills installation.
#
#   ./install.sh                    # all skills -> ~/.agents/skills
#   ./install.sh release-restore   # any subset: name skills as arguments
#   ./install.sh release-restore add-lyrics phone-music
#   ./install.sh --skills-dir D     # symlink into a custom dir (any agent)
#   ./install.sh --copy             # copy instead of symlink (updates need re-run)
#   ./install.sh --venv             # also create .venv and pip-install requirements
#
# Skills: music-care (router), release-restore (album restoration), audio-measure,
#         tracker (search/compare/download),
#         add-lyrics (lyrics into tags), phone-music (MTP library care),
#         split-flac-album (image+cue splitting).
# Idempotent: re-run any time. Never touches your music files.
set -euo pipefail
cd "$(dirname "$0")"
REPO="$(pwd)"

ALL_SKILLS="music-care release-restore audio-measure tracker add-lyrics phone-music split-flac-album"
SKILLS_DIR="${HOME}/.agents/skills"
MODE=symlink
VENV=0
SKILLS=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --skills-dir) SKILLS_DIR="$2"; shift 2 ;;
    --copy) MODE=copy; shift ;;
    --venv) VENV=1; shift ;;
    -h|--help) grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    -*) echo "unknown option: $1"; exit 1 ;;
    *) SKILLS="$SKILLS $1"; shift ;;
  esac
done
SKILLS="${SKILLS:-${ALL_SKILLS}}"
for s in $SKILLS; do
  [[ -d "$REPO/skills/$s" ]] || { echo "unknown skill: $s (available:$ALL_SKILLS)"; exit 1; }
done

has() { [[ " $SKILLS " == *" $1 "* ]]; }

fail=0
say() { printf '%s\n' "$*"; }
have() { command -v "$1" >/dev/null 2>&1; }

say "== dependencies =="
for cmd in ffmpeg ffprobe flac metaflac curl; do
  if have "$cmd"; then say "  ok      $cmd"; else say "  MISSING $cmd  (install it with your package manager)"; fail=1; fi
done
if has phone-music; then
  if have gio; then say "  ok      gio (phone over MTP)"; else say "  NOTE    gio missing — phone/MTP features unavailable (local files unaffected)"; fi
fi
if python3 -c "import numpy, scipy, matplotlib, mutagen, PIL, bs4, requests, sklearn, joblib" 2>/dev/null; then
  say "  ok      python3 + numpy/scipy/matplotlib/mutagen/pillow/bs4/requests/sklearn/joblib"
else
  say "  MISSING python3 deps: pip install -r requirements.txt (or use --venv)"
  fail=1
fi

if [[ $VENV -eq 1 ]]; then
  python3 -m venv .venv
  # shellcheck disable=SC1091
  source .venv/bin/activate
  pip install -q -r requirements.txt
  pip install -q -e .
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
for s in $SKILLS; do
  target="${SKILLS_DIR}/${s}"
  if [[ -e "$target" && ! -L "$target" ]]; then
    say "  SKIP    $target exists and is not a symlink — move it away first if you want $MODE"
    continue
  fi
  rm -rf "$target"
  if [[ $MODE == symlink ]]; then
    ln -s "${REPO}/skills/${s}" "$target"
  else
    cp -r "${REPO}/skills/${s}" "$target"
  fi
  say "  ok      $s"
done

say
if [[ $fail -eq 0 ]]; then
  say "All set. Restore an album:"
  say "  python3 scripts/restore_one.py --src <dir-or-image.flac> --out <dir> --auto-policy --adaptive"
  say "Full release (tags/covers/lyrics/phone): skill release-restore."
else
  say "Some dependencies are missing — fix them, then re-run ./install.sh"
  exit 1
fi
