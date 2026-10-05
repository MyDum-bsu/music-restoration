# music-restoration

Undo the loudness war on your music library. A transparency-first dynamic-range
restoration pipeline (DR4–6 masters → DR8–12) with hard verification gates,
plus library tooling: lyrics, covers, tags, DR scanning — for local folders or
an Android phone over MTP.

Built around a reverse-engineered professional restoration chain, reimplemented
on free software (ffmpeg + numpy/scipy + Stereo Tool's free CLI features), and
battle-tested on a ~3000-file library where every delivery passes an md5-verified
gate.

**Works two ways:**

1. **With an AI agent** — `./install.sh` symlinks four agent skills
   (`restore-dr`, `add-lyrics`, `phone-music`, `split-flac-album`) into your
   skills directory; then just ask your agent: *"реставрируй этот альбом"*,
   *"add lyrics to this folder"*, *"audit my phone library"*.
2. **By hand** — plain CLI scripts, everything documented:

```bash
# diagnose a crushed master, then restore one track
python3 restore-dr/scripts/diagnose.py album/*.flac
python3 restore-dr/scripts/master.py track in.flac out.flac          # DR4 → DR9-grade

# one command per album: restore + covers + tags + gate + push to the phone
python3 restore-dr/scripts/restore_batch.py "Artist - Album" --lyrics --mb

# metrics that match dr.loudness-war.info + before/after graphs
python3 restore-dr/scripts/measure.py compare in.flac out.flac --outdir report
python3 restore-dr/scripts/scan_dr.py            # whole-library DR scan (phone or dir)
```

No hardcoded paths: the phone music dir is auto-detected from gvfs MTP mounts,
the local library defaults to `~/Music`, staging lives under
`~/.local/state/music-restoration/` — all overridable in
`~/.config/music-restoration/config.env` or env vars.

## Install

```bash
git clone <this repo> && cd music-restoration
./install.sh                 # checks deps, installs agent skills, creates config template
```

Dependencies: `python3` + `pip install -r requirements.txt`, and
`ffmpeg flac metaflac curl gio` from your package manager. Optional:
[Stereo Tool CLI](docs/stereo-tool.md) (free for non-commercial, not
redistributable — enables the ND2 punch stage).

## The four skills

| Skill | What it does |
|---|---|
| `restore-dr` | diagnose → restore (v2 chain) → verify → deliver, with TT DR/LUFS metrics and graphs |
| `add-lyrics` | Genius → lrclib lyrics embedded into tags (FLAC/MP3/M4A/Ogg/Opus) |
| `phone-music` | audit a phone/local library; bulk lyrics, covers, track numbers over MTP |
| `split-flac-album` | split a single-FLAC album by CUE into tagged tracks with covers |

## Docs

- [docs/pipeline.md](docs/pipeline.md) — the chain, parameter choices, verification
- [docs/graphs-guide.md](docs/graphs-guide.md) / [RU](docs/graphs-guide.ru.md) — how to read the before/after graphs
- [docs/mtp-notes.md](docs/mtp-notes.md) — MTP/phone delivery: the hard-won rules
- [docs/stereo-tool.md](docs/stereo-tool.md) — Stereo Tool setup, free-vs-paid safety

## Typical results

Crushed 2000s masters come out of the loudness-war zone at matched fidelity:
spectrum within ±1 dB, true peak ≤ −0.5 dBTP, zero clipped samples, duration
bit-identical, DR roughly doubled. See `docs/graphs-guide.md` for what the
numbers mean — and remember the project rule: **ears decide, metrics lead.**

## Legal note

Restoration is for music you legally own. The pipeline reads your files,
writes new files next to them (never in place), and ships no third-party
binaries. Genius/lrclib/MusicBrainz/Cover Art Archive are used politely
(sequential requests, caching).

## License

MIT — see [LICENSE](LICENSE).
