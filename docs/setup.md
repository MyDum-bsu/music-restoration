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

## 2. Secrets: tracker cookies, UA, Transmission

Nothing secret is stored in the repository. Secrets live in
`~/.config/music-restoration/secrets/` (`rnext/secrets.py`):

| file | content |
|---|---|
| `rutracker_cookies.txt` | one `name=value` per line: `bb_session`, `bb_ssl`, `bb_t`, `cf_clearance` |
| `rutracker_ua.txt` | the exact `User-Agent` of the browser those cookies came from |
| `transmission_rpc.json` | `{"host": ..., "port": ..., "user": ..., "password": ...}` |

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
python3 scripts/restore_per_track.py --src "album/" --out "out/album [restored]"
python3 scripts/qc_gates.py "out/album [restored]"/*.flac  # TP, clipping, spikes
python3 scripts/qc_gates.py --orig "album/" "out/album [restored]"/01*.flac
python3 scripts/finish_release.py --restored "out/album [restored]" --source "album/"
```

`qc_gates.py --orig <source dir>` compares the output against the source take:
the applied gain becomes measurable and the spike gate FAILs only when the
output has more click-like jumps than the source (a hot master already carries
hundreds). A missing source take is an explicit failure, never a silent
fallback to the absolute gate.

## 6. Checks before a pull request

```bash
python3 -m pytest tests/ -q
```

CI runs the same command on every push and pull request
(`.github/workflows/ci.yml`). Coding conventions are in
[`CODING_STANDARDS.md`](../CODING_STANDARDS.md).
