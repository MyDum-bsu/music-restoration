# Setup: from a fresh clone to a working restoration run

The product is agent-native: an agent session with the skills in `skills/` drives
`scripts/`. Everything below is what those scripts need; nothing is installed
into the system except the external tools.

## 1. Requirements

- Python 3.10+ and the packages from `requirements.txt`
  (`numpy`, `scipy`, `matplotlib`, `mutagen`, `pillow`, `beautifulsoup4`,
  `requests`, `scikit-learn`, `joblib`, `pytest`);
- `ffmpeg`, `flac`, `metaflac`, `shntool`, `cuetag` for decoding, tagging and
  image+cue splitting;
- Transmission with the RPC endpoint enabled, if the agent is to queue downloads.

```bash
git clone <this repo> && cd music-restoration
./install.sh            # dependency check + skills into ~/.agents/skills
./install.sh --venv     # optional: a local .venv with the requirements
python3 -m pytest tests/ -q
```

`python3 scripts/doctor.py` re-checks all of this locally at any time (no
network): exit 0 — all green, 1 — a HARD check failed (the fix command is
printed per package family), 2 — only WARN.

### Platforms

| platform | status | notes |
|---|---|---|
| Linux — Debian/Ubuntu/Mint (`apt`) | supported | `install.sh` and `doctor.py` print ready `sudo apt install …` commands |
| Linux — Fedora/RHEL (`dnf`) | supported | same via `sudo dnf install …` |
| Linux — Arch (`pacman`) | supported | same via `sudo pacman -S …` |
| macOS (`brew`) | supported | system tools via brew; Python packages via `python3 -m pip install -r requirements.txt` — brew has no formulas for them |
| Windows | WSL only | no native Windows support: package detection covers apt/dnf/pacman/brew only, so run everything inside WSL |
| `phone-music` (MTP delivery) | Linux desktop with `gio` | `gio` is a WARN check in `doctor.py`; without it local work is unaffected, phone delivery is not. Android delivery over ADB (`adb`) also works on macOS |

## 2. Secrets: tracker cookies, UA, Transmission

Nothing secret is stored in the repository. Secrets live in
`~/.config/music-restoration/secrets/` (`rnext/secrets.py`):

| file | content |
|---|---|
| `rutracker_cookies.txt` | one `name=value` per line: `bb_session`, `bb_ssl`, `bb_t`, `cf_clearance` |
| `rutracker_ua.txt` | the exact `User-Agent` of the browser those cookies came from |
| `transmission_rpc.json` | `{"host": ..., "port": ..., "user": ..., "password": ...}` |

Permissions are owner-only: `install.sh` creates the directory with `chmod
700` and keeps every file in it at `chmod 600`. When adding files by hand,
match that:

```bash
mkdir -p ~/.config/music-restoration/secrets
chmod 700 ~/.config/music-restoration/secrets
chmod 600 ~/.config/music-restoration/secrets/*
```

## 3. `cf_clearance`: why it breaks and how to refresh it

`rutracker.org` sits behind Cloudflare. The `cf_clearance` cookie is bound to
**one IP address and one User-Agent** and expires after a while. The symptom is
unmistakable: `rnext/rtclient.py` raises `ChallengeError` ("Cloudflare
challenge; browser fallback needed") or `NotLoggedError` on every request,
while the same pages open fine in a browser.

Refresh the cookie pair from the browser that already passes the challenge:

```bash
python3 scripts/rt_browser_refresh.py --help          # all flags
python3 scripts/rt_browser_refresh.py --check         # are the stored cookies complete?
python3 scripts/rt_browser_refresh.py                 # the only Firefox profile found
python3 scripts/rt_browser_refresh.py --profile <firefox-profile-dir>
```

The script copies `cookies.sqlite` out of the profile (the live database is
never touched), keeps the `rutracker.org` cookies, writes
`rutracker_cookies.txt` + `rutracker_ua.txt` into the secrets directory and
reads them back to verify. Cookie values are never printed.

Two rules that save an hour of guessing:

1. the UA must be the one of the browser that minted `cf_clearance` — a
   different UA gets a fresh challenge;
2. Chromium-family profiles are not read: their cookie values are encrypted
   with an OS-keyring key. Export them from DevTools
   (Application → Cookies → `rutracker.org`) instead.

## 4. Workspace: where releases and temp data live

`config/workspace.json` holds the only machine-specific settings:

```json
{
  "data_root": "<a writable directory, ideally a separate drive>",
  "budget_bytes": 42949672960,
  "dirs": {"torrents": "torrents", "corpus": "corpus", "abx": "abx", "cache": "cache"}
}
```

Start from the portable template `config/workspace.example.json`, copy it to
`config/workspace.json` and set `data_root`. `rnext/workspace.py` refuses to
work with a missing `data_root` or an unknown directory kind, and the 40 GiB
budget is enforced before every download.

`~/.config/music-restoration/config.env` (see `config.env.example`) holds
optional overrides such as `MR_PHONE_DIR`.

## 5. First run

```bash
python3 scripts/audio_measure.py "album/"                  # what is this file, really
python3 scripts/restore_safe.py --src "album/" --out "out/album [restored]"
python3 scripts/qc_gates.py "out/album [restored]"/*.flac  # TP, clipping, spikes
python3 scripts/qc_gates.py --orig "album/" "out/album [restored]"/01*.flac
python3 scripts/finish_release.py --restored "out/album [restored]" --source "album/"
```

`qc_gates.py --orig <source dir>` compares the output against the source take:
the applied gain becomes measurable and the spike gate FAILs only when the
output has more click-like jumps than the source (a hot master already carries
hundreds). A missing source take is an explicit failure, never a silent
fallback to the absolute gate.

Inspect every status in the safe runner's `report.json`: `processed_candidate`,
`kept_original`, `qc_fallback_original`, or `error`. A copied original can retain
its original high peak or other defects; external QC does not turn that copy
into a repaired result. The legacy `restore_per_track.py` remains available for
historical reproduction, but is no longer the preferred automatic path.

## 6. Checks before a pull request

```bash
python3 -m pytest tests/ -q
```

CI runs the same command on every push and pull request
(`.github/workflows/ci.yml`). Coding conventions are in
[`CODING_STANDARDS.md`](../CODING_STANDARDS.md).

## 7. Troubleshooting

| symptom | cause | command |
|---|---|---|
| `doctor.py` exits 1, a tool row says `не найден в PATH` | a system tool is missing (ffmpeg, flac, shntool, cuetools…) | run the install command printed in the fix column, e.g. `sudo apt install ffmpeg flac shntool cuetools` |
| `doctor.py` exits 1, `не импортируются: …` | Python packages missing | `python3 -m pip install -r requirements.txt` (or `./install.sh --venv`) |
| `doctor.py` exits 1, `файл не найден: config/workspace.json` | workspace config not created yet | `./install.sh` creates it from `config/workspace.example.json`; or copy it yourself and set `data_root` |
| `doctor.py` exits 1, `каталог не существует` / `нет прав на запись` for `data_root` | data root missing or unwritable | `mkdir -p <data_root> && chmod u+w <data_root>` |
| `doctor.py` exits 1, `свободно … GiB из требуемых 1.00 GiB` | less than 1 GiB free on `data_root` | `df -h <data_root>` — free up space or point `data_root` elsewhere in the config |
| tracker: `NotLoggedError` / `ChallengeError` on every request | cookies missing/expired, or `cf_clearance` bound to another IP/UA | `python3 scripts/rt_browser_refresh.py --check`, then refresh — section 3 above |
| tracker: topic page ~1.5 KB, forum "temporarily disabled for maintenance" | rutracker maintenance stub, not an authorization failure | use a local `.torrent` from `~/Downloads` until maintenance ends |
| `doctor.py` exits 2, `RPC 127.0.0.1:9091 не отвечает` | Transmission RPC is down | start `transmission-gtk` (or `transmission-daemon`), then `python3 -c "from rnext.transm import check; print(check())"` |
| `doctor.py` exits 2, `gio не найден` | MTP delivery unavailable | install `gio` with the printed command (`sudo apt install libglib2.0-bin` on apt) |
