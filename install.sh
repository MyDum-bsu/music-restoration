#!/usr/bin/env bash
# music-restoration installer: dependency check + agent-skills installation.
#
#   ./install.sh                    # all four skills -> ~/.agents/skills
#   ./install.sh restore-dr         # any subset: name skills as arguments
#   ./install.sh restore-dr add-lyrics phone-music
#   ./install.sh --skills-dir D     # symlink into a custom dir (any agent)
#   ./install.sh --copy             # copy instead of symlink (updates need re-run)
#   ./install.sh --venv             # also create .venv and pip-install requirements
#
# Skills: restore-dr (DR restoration), add-lyrics (lyrics into tags),
#         phone-music (phone library over MTP), split-flac-album (cue splitting).
# Idempotent: re-run any time. Never touches your music files.
set -euo pipefail
cd "$(dirname "$0")"
REPO="$(pwd)"

ALL_SKILLS="restore-dr add-lyrics phone-music split-flac-album"
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
  [[ -d "$REPO/$s" ]] || { echo "unknown skill: $s (available:$ALL_SKILLS)"; exit 1; }
done

has() { [[ " $SKILLS " == *" $1 "* ]]; }

fail=0
say() { printf '%s\n' "$*"; }
have() { command -v "$1" >/dev/null 2>&1; }

say "== dependencies =="
for cmd in ffmpeg ffprobe flac metaflac curl; do
  if have "$cmd"; then say "  ok      $cmd"; else say "  MISSING $cmd  (install it with your package manager)"; fail=1; fi
done
if has phone-music || has restore-dr; then
  if have gio; then say "  ok      gio (phone over MTP)"; else say "  NOTE    gio missing — phone/MTP features unavailable (local files unaffected)"; fi
fi
if python3 -c "import numpy, scipy, matplotlib, mutagen" 2>/dev/null; then
  say "  ok      python3 + numpy/scipy/matplotlib/mutagen"
else
  say "  MISSING python3 deps: pip install -r requirements.txt (or use --venv)"
  fail=1
fi

if has restore-dr; then
  if [[ -x "restore-dr/bin/stereo_tool_cmd_64" || -x "restore-dr/bin/stereo_tool_cmd_64.exe" ]]; then
    say "  ok      Stereo Tool CLI (restore-dr/bin)"
  else
    say "  NOTE    Stereo Tool CLI not found — ND2 punch stage disabled until you add it."
    say "          Free for non-commercial: https://www.stereo-tool.com -> command line"
    say "          version for your OS -> restore-dr/bin/ (see docs/stereo-tool.md)."
    say "          Everything else (diagnose/measure/contour pipeline) works without it."
  fi
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
for s in $SKILLS; do
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

if has phone-music && ! has restore-dr; then
  say "  NOTE    phone-music borrows helper code from restore-dr (same repo, found automatically);"
  say "          install restore-dr too if you want its skill visible to the agent."
fi
if { has phone-music || has restore-dr; } && ! has add-lyrics; then
  say "  NOTE    lyrics features (--lyrics, embed_lyrics) use the add-lyrics skill;"
  say "          add it to the install list if you plan to embed lyrics."
fi

say
if [[ $fail -eq 0 ]]; then
  say "All set. Try: restore-dr/scripts/measure.py metrics <file.flac>"
else
  say "Some dependencies are missing — fix them, then re-run ./install.sh"
  exit 1
fi
