# music-restoration

Undo the loudness war on your music library: crushed DR4–6 masters come back
to DR8–12 with the timbre intact. One agent phrase — **"restore this"** —
covers everything: the agent finds the release, downloads the best quality,
measures it, restores every track individually, checks the result, applies
tags/cover/lyrics, and delivers the album to your collection with a
master-style DR report.

This is a **skills-first** project: instead of one universal trained model,
the system is a set of agent skills over a deterministic DSP core.

## Agent story

```bash
git clone https://github.com/MyDum-bsu/music-restoration.git
cd music-restoration
./install.sh                 # dependency check + skills into ~/.agents/skills
```

Then just say to your agent:

> "restore this" — with a path, a link, or just an album name

The agent will:
1. find the best-quality release on the tracker and download it (if the file
   is not on disk), verifying it is real lossless;
2. measure the source (DR, LUFS, TP, clipping, fake hi-res, lossy transcode);
3. split image+cue if needed;
4. restore **every track individually** — a candidate sweep of `flatten`/
   `expand` policies per track, the winner verified on the full track and
   automatically re-processed if Δcrest leaves the target window;
5. run QC gates (TP −0.5 dBFS, zero clipping, zero spikes, gain smoothness);
6. transfer tags, cover art and lyrics, copy the album into the collection;
7. finish with a **master-style DR Report** (`DR Report.txt`, the exact table
   format restoration masters post in their release topics) and **spectra** —
   per-track PNG pages (waveform, original/restored spectrograms, average
   spectrum, crest over time) plus animated GIFs alternating the two.

## Skills

| skill | purpose |
|---|---|
| `music-care` | router: any music task → the right skill |
| `release-restore` | "restore this": find/download → per-track restore → QC → dressing → collection |
| `audio-measure` | measure DR/loudness/authenticity of files or folders |
| `tracker` | search, compare and download releases |
| `restore-status` | project state: dataset, downloads, processes, models |
| `add-lyrics` | lyrics into tags (genius.com + lrclib) |
| `phone-music` | inventory, lyrics and covers on the phone over MTP |
| `split-flac-album` | splitting image+cue into tracks |

Skills are thin wrappers over `scripts/`; every one also works without an agent.

## The restoration pipeline

`rnext/dsp/pipeline.py`:

1. **HPF 25 Hz** — zero-phase rumble/DC removal.
2. **Declip** — only when real clipping exists: AR extrapolation,
   forward+backward blend.
3. **EQ** — spectral delta versus a reference (off by default).
4. **Contour** — the main stage: per-band, per-frame level above a smoothed
   body is stretched; transients survive thanks to WOLA and a final TP clip.
5. **TP limit** — 4× oversampled true-peak measurement, soft bring-down to
   −0.5 dBFS.
6. **Output** — 16/24-bit FLAC (`--bits auto` keeps the source depth),
   TPDF dither.

**Per-track approach** (`scripts/restore_per_track.py`): a candidate sweep of
`flatten` (slow multiband modulation with a gain cap — larger DR, less timbre
damage on crushed masters) and `expand` policies, scored by Δcrest (+6.3 dB
target), timbre error and over/under-expansion; the winner is verified on the
full track and re-processed with neighboring parameters if it misses the
window. Every run writes `per_track_report.json` and `PER_TRACK_REPORT.md`.

Details: [skills/release-restore/references/quality.md](skills/release-restore/references/quality.md)
and [docs/dataset.md](docs/dataset.md), [docs/ml.md](docs/ml.md).

## Reports

- `DR Report.txt` — the classic DR-meter table (`Analyzed:`, per-track
  DR/Peak/RMS/duration, `Official DR value` computed by pooling blocks of all
  tracks exactly like the meter). Byte-compatible with the format restoration
  masters post on trackers; parsed by `rnext.rttopic.parse_dr_table`.
- `spectra/*.png` + `*.gif` — MasVis-style per-track pages and animated
  original/restored spectrogram comparisons (`scripts/spectrum_report.py`).
- `per_track_report.json`, `PER_TRACK_REPORT.md`, `restore_report.jsonl`,
  `_measure.json` — machine-readable metrics.

## Manual CLI (no agent)

```bash
python3 scripts/audio_measure.py "album/"                  # DR/LUFS/TP/clipping
python3 scripts/restore_per_track.py --src "album/" \
        --out out/album\ [restored]                        # per-track restore
python3 scripts/finish_release.py --restored out/album\ [restored] \
        --source "album/" --dest ~/Music/album\ [restored] # tags/cover/lyrics/DR report
python3 scripts/dr_report.py out/album\ [restored]         # DR-meter table
python3 scripts/spectrum_report.py --orig "album/" \
        --restored out/album\ [restored] --out out/spectra # spectra + GIFs
```

## Data and configuration

- Corpus and releases: `config/workspace.json` → `data_root` (default flash
  drive `/media/<user>/Data/Music/restore`). Override with `RESTORE_DATA_ROOT`.
- Personal config: `~/.config/music-restoration/config.env` (sample:
  `config.env.example`). All keys optional.
- Secrets (rutracker cookies, Transmission RPC): `~/.config/music-restoration/secrets/`.
- State and caches: `~/.local/state/music-restoration/`.
- Models: `models/snapshots/` (local artifacts, not committed); registry and
  leaderboard — `models/registry.jsonl`, `models/leaderboard.jsonl`.

## Corpus and ML (optional)

The DSP core is self-sufficient. The optional ML layer learns a Δcrest target
estimator from "original → professional restoration" pairs of two independent
restoration masters; a dataset is content-versioned (`ds-<hash>`), models are
promoted only when they beat the best baseline with a 95% CI. See
[docs/dataset.md](docs/dataset.md) and [docs/ml.md](docs/ml.md).

```bash
python3 scripts/status_report.py --deep    # project state
python3 scripts/ml_auto.py                 # version → train → evals → summary
```

## Blind listening

```bash
python3 scripts/listen_pack.py ...   # LUFS-aligned variant pack
cd app && npm install && npm start   # Electron A/B/C player
```

Local corpus/pipeline overview: `python3 lab/server.py` →
http://127.0.0.1:8765 (see [lab/README.md](lab/README.md)).

## Tests

```bash
python3 -m pytest tests/ -q
```

## History

- `v1.0` — the old Stereo Tool CLI engine. Preserved as a git tag.
- `v2.0` — own DSP core, corpus, ML, release cycle.
- `v2.1` — agent-native skills, per-track restoration, master-style DR and
  spectrum reports.

## License

MIT — see [LICENSE](LICENSE).
