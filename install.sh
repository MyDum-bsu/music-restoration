#!/usr/bin/env bash
# music-restoration installer: dependency check + agent-skills installation.
#
#   ./install.sh                    # all skills -> ~/.agents/skills
#   ./install.sh release-restore   # any subset: name skills as arguments
#   ./install.sh release-restore add-lyrics phone-music
#   ./install.sh --skills-dir D     # symlink into a custom dir (any agent)
#   ./install.sh --copy             # copy instead of symlink (updates need re-run)
#   ./install.sh --venv             # also create .venv and pip-install requirements
#   ./install.sh --install-tools    # try to install MISSING system tools via the detected
#                                   # package manager (no sudo: brew runs directly; for
#                                   # apt-get/dnf/pacman the ready command is printed)
#
# Skills: music-care (router), release-restore (album restoration), audio-measure,
#         tracker (search/compare/download),
#         add-lyrics (lyrics into tags), phone-music (MTP library care),
#         split-flac-album (image+cue splitting).
# Secrets: rutracker cookies/UA live in ~/.config/music-restoration/secrets (bootstrap
#          here; chmod 700 dir, 600 files — see README.md there). The tracker skill
#          requires a rutracker account (registration is invite-only).
# Idempotent: re-run any time. Never touches your music files.
set -euo pipefail
cd "$(dirname "$0")"
REPO="$(pwd)"

ALL_SKILLS="music-care release-restore audio-measure tracker add-lyrics phone-music split-flac-album"
SKILLS_DIR="${HOME}/.agents/skills"
MODE=symlink
VENV=0
INSTALL_TOOLS=0
SKILLS=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --skills-dir) SKILLS_DIR="$2"; shift 2 ;;
    --copy) MODE=copy; shift ;;
    --venv) VENV=1; shift ;;
    --install-tools) INSTALL_TOOLS=1; shift ;;
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

# System package manager, detected by command -v; first match wins.
# Prints the manager name (apt-get=Debian/Ubuntu, dnf=Fedora, pacman=Arch,
# brew=macOS) or an empty string when none is found. Always exits 0.
pm_detect() {
  local p
  for p in apt-get dnf pacman brew; do
    if have "$p"; then echo "$p"; return 0; fi
  done
  echo ""
}

# Package that provides a tool; names hold across apt-get/dnf/pacman/brew.
# cuetag comes from the cuetools package everywhere (Debian ships it as
# /usr/bin/cuetag, other distros as cuetag.sh — see have_tool).
pkg_for() {
  case "$1" in
    ffprobe)  echo ffmpeg ;;
    metaflac) echo flac ;;
    cuetag)   echo cuetools ;;
    *)        echo "$1" ;;
  esac
}

# cuetag is shipped as cuetag.sh by some distros (dnf/pacman/brew cuetools
# packages) — either name means the tool is present.
have_tool() {
  case "$1" in
    cuetag) have cuetag || have cuetag.sh ;;
    *)      have "$1" ;;
  esac
}

# Ready-to-run install command for packages via a manager. Root-requiring
# managers are never executed by this script (no sudo) — the command is
# printed for the user instead.
manager_install_cmd() {  # args: pm pkgs...
  case "$1" in
    apt-get) echo "sudo apt-get install ${*:2}" ;;
    dnf)     echo "sudo dnf install ${*:2}" ;;
    pacman)  echo "sudo pacman -S ${*:2}" ;;
    brew)    echo "brew install ${*:2}" ;;
  esac
}

TOOLS="ffmpeg ffprobe flac metaflac curl shntool cuetag"
MISSING_TOOLS=""
pm="$(pm_detect)"
say "== dependencies =="
for cmd in $TOOLS; do
  if have_tool "$cmd"; then
    say "  ok      $cmd"
  else
    if [[ -n $pm ]]; then
      say "  MISSING $cmd  (install: $(manager_install_cmd "$pm" "$(pkg_for "$cmd")"))"
    else
      say "  MISSING $cmd  (no apt-get/dnf/pacman/brew found — install it manually)"
    fi
    MISSING_TOOLS="$MISSING_TOOLS $cmd"
  fi
done

# --install-tools: try to install what is MISSING via the detected manager.
# This script never uses sudo: brew is driven directly (it refuses sudo anyway);
# apt-get/dnf/pacman need root, so their ready command is printed to run yourself.
if [[ $INSTALL_TOOLS -eq 1 ]]; then
  if [[ -z "${MISSING_TOOLS// }" ]]; then
    say "  ok      --install-tools: nothing to install"
  elif [[ -z $pm ]]; then
    say "  --      --install-tools: no apt-get/dnf/pacman/brew found — install${MISSING_TOOLS} manually"
  else
    pkgs=""
    for t in $MISSING_TOOLS; do
      p="$(pkg_for "$t")"
      case " $pkgs " in *" $p "*) ;; *) pkgs="$pkgs $p" ;; esac
    done
    pkgs="${pkgs# }"
    run="$(manager_install_cmd "$pm" $pkgs)"
    if [[ $pm == brew ]]; then
      say "  --      --install-tools: running: $run"
      brew install $pkgs || say "  MISSING brew install failed — run it yourself: $run"
    else
      say "  --      --install-tools: $pm needs root and this script never uses sudo — run:"
      say "          $run"
    fi
  fi
fi

# verdict on the system tools (the --install-tools attempt above may have fixed them)
still=""
for cmd in $MISSING_TOOLS; do
  have_tool "$cmd" || still="$still $cmd"
done
if [[ -n "${still// }" ]]; then
  fail=1
  if [[ $INSTALL_TOOLS -eq 1 ]]; then
    say "  MISSING still missing after install attempt:${still} — install, then re-run ./install.sh"
  fi
elif [[ -n "${MISSING_TOOLS// }" && $INSTALL_TOOLS -eq 1 ]]; then
  say "  ok      --install-tools: everything installed"
fi
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

# secrets: rutracker cookies/UA etc. — machine-local, owner-only, never in the repo.
# Bootstrap only creates the dir + README; existing secret files are never touched.
say "== secrets =="
SEC_DIR="${CFG_DIR}/secrets"
mkdir -p "$SEC_DIR"
chmod 700 "$SEC_DIR"
if [[ ! -f "$SEC_DIR/README.md" ]]; then
  cat > "$SEC_DIR/README.md" <<'EOF'
# secrets — machine-local credentials for music-restoration

Everything here stays on this machine (directory chmod 700, files chmod 600)
and is never copied into the repository. install.sh only creates this README;
it never writes or rewrites any other file in this directory.

## rutracker_cookies.txt — session cookies for the tracker skill

What: rutracker.org session cookies, one `name=value` per line; empty lines and
lines starting with `#` are ignored. Expected at minimum:

    bb_session=<...>
    bb_ssl=<...>
    bb_t=<...>
    cf_clearance=<...>

`bb_data` (old login scheme) is picked up too if present.

Why: the tracker skill (search/compare/download on rutracker) authenticates
with these cookies instead of a login+password, so the password itself is
never stored anywhere.

## rutracker_ua.txt — the browser User-Agent bound to those cookies

What: one line — the exact `User-Agent` string of the browser the cookies were
taken from (must start with `Mozilla/`, e.g. `Mozilla/5.0 ... Chrome/...`).

Why: rutracker sits behind Cloudflare; the `cf_clearance` cookie is bound to
one IP and one User-Agent. Requests that send cookies with a mismatched UA are
rejected, so both files must come from the same browser session.

## transmission_rpc.json (optional)

`{"host": ..., "port": ..., "user": ..., "password": ...}` — the Transmission
RPC endpoint the agent queues torrents into.
EOF
  say "  created $SEC_DIR/README.md (what belongs here)"
else
  say "  ok      $SEC_DIR/README.md exists (left untouched)"
fi
find "$SEC_DIR" -maxdepth 1 -type f -exec chmod 600 {} +
say "  WARNING tracker skill requires a rutracker account (registration is invite-only);"
say "          its cookies/UA live in this dir — see $SEC_DIR/README.md"

# workspace.json: machine-local (gitignored); create from the tracked example,
# data_root from RESTORE_DATA_ROOT or ~/music-restoration-data. Never overwrite.
WRK="$REPO/config/workspace.json"
if [[ -f "$WRK" ]]; then
  say "  ok      config/workspace.json exists (left untouched)"
elif [[ ! -f "$REPO/config/workspace.example.json" ]]; then
  say "  MISSING config/workspace.example.json — cannot create config/workspace.json"
  fail=1
else
  DATA_ROOT="${RESTORE_DATA_ROOT:-${HOME}/music-restoration-data}"
  python3 - "$REPO/config/workspace.example.json" "$WRK" "$DATA_ROOT" <<'PY'
import json, sys
src, dst, data_root = sys.argv[1:4]
with open(src, encoding="utf-8") as f:
    cfg = json.load(f)
cfg["data_root"] = data_root
with open(dst, "w", encoding="utf-8") as f:
    json.dump(cfg, f, indent=2)
    f.write("\n")
PY
  say "  created config/workspace.json (data_root=${DATA_ROOT})"
  python3 - "$WRK" <<'PY' || fail=1
import json, os, sys
with open(sys.argv[1], encoding="utf-8") as f:
    cfg = json.load(f)
rc = 0
for kind, rel in sorted(cfg.get("dirs", {}).items()):
    d = os.path.join(cfg["data_root"], str(rel))
    try:
        os.makedirs(d, exist_ok=True)
        probe = os.path.join(d, ".install_write_probe")
        with open(probe, "x") as pf:
            pf.write("ok\n")
        os.remove(probe)
        print(f"  ok      {kind}: {d} writable")
    except OSError as e:
        print(f"  MISSING {kind}: {d} not writable: {e}")
        rc = 1
sys.exit(rc)
PY
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
