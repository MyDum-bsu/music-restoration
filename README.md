# music-restoration

Undo the loudness war on your music library: take crushed DR4–6 masters back
to a healthy DR8–12 without touching the timbre. A dynamic-range restoration
pipeline where every delivery step is verified, plus library care — lyrics,
covers, tags, DR scanning — for local folders and an Android phone over MTP.

At its core is a professional restoration chain, reverse-engineered from
reference restorations and rebuilt from free software: ffmpeg, numpy/scipy,
and the free CLI features of Stereo Tool. The tool was proven on a ~3000-file
library; every album passes integrity/tag/cover/duration checks before it is
written and a byte-level md5 comparison after.

Two ways to use it:

1. **With an AI agent.** `./install.sh` links four agent skills
   (`restore-dr`, `add-lyrics`, `phone-music`, `split-flac-album`) into your
   skills directory; then just ask your agent: "реставрируй этот альбом",
   "add lyrics to this folder", "audit my phone library".
2. **By hand.** Plain scripts with a readable CLI:

```bash
# diagnose a crushed master, then restore one track
python3 restore-dr/scripts/diagnose.py album/*.flac
python3 restore-dr/scripts/master.py track in.flac out.flac

# one command per album: restore, covers, tags, verification, push to the phone
python3 restore-dr/scripts/restore_batch.py "Artist - Album" --lyrics --mb

# metrics in dr.loudness-war.info terms, plus before/after graphs
python3 restore-dr/scripts/measure.py compare in.flac out.flac --outdir report
python3 restore-dr/scripts/scan_dr.py            # whole-library DR scan
```

No hardcoded paths anywhere. The phone music dir is found automatically
(gvfs MTP), the local library defaults to `~/Music`, working files go to
`~/.local/state/music-restoration/` — all of it overridable in
`~/.config/music-restoration/config.env` or via environment variables.

## Platforms

| Task | Windows / macOS | Linux |
|---|---|---|
| Restoration core: diagnose, master, measure, scan_dr, restore_batch `--out-local` | works | works |
| Phone delivery over MTP, the phone-music skill | no (needs gvfs; WSL does not help) | works |
| split-flac-album (shntool/cuetag) | builds exist but are finicky | works |

Dependencies: Python 3.10+ with `pip install -r requirements.txt`, plus
`ffmpeg`, `flac` (installs `metaflac` alongside) and `curl`. MusicBrainz
tracklists and Cover Art Archive covers go through curl — Windows 10+ ships
it. `install.sh` is bash; on Windows, run the same steps by hand — yourself
or through your agent (each SKILL.md lists its requirements explicitly).

## Install

```bash
git clone <this repo> && cd music-restoration
./install.sh
```

Optional: the [Stereo Tool CLI](docs/stereo-tool.md). It is free for
non-commercial use and cannot be redistributed, so it is not bundled; it
enables the ND2 punch stage. Everything else works without it — restore with
`--nd2 0`.

## Docs

- [docs/pipeline.md](docs/pipeline.md) — the chain, parameter choices, verification
- [docs/graphs-guide.md](docs/graphs-guide.md) / [RU](docs/graphs-guide.ru.md) — how to read the before/after graphs
- [docs/mtp-notes.md](docs/mtp-notes.md) — phone/MTP delivery rules
- [docs/stereo-tool.md](docs/stereo-tool.md) — Stereo Tool setup, free-vs-paid safety

## Typical results

Crushed 2000s masters leave the loudness-war zone with the timbre intact:
spectrum within ±1 dB, true peak at or below −0.5 dBTP, zero clipped samples,
duration identical to the millisecond, DR roughly doubled. What the numbers
mean — in the [graphs guide](docs/graphs-guide.md). House rule: metrics lead,
ears decide.

## Legal note

Use it on music you legally own. The scripts read your files and write new
ones next to them — sources are never overwritten. The repo ships no
third-party binaries. Genius, lrclib, MusicBrainz and Cover Art Archive are
used politely: sequential requests, caching.

## License

MIT — see [LICENSE](LICENSE).
