---
name: phone-music
description: Work with a phone music library over MTP (or any local library) — audit it, bulk-add lyrics, embed covers and track numbers, fix untagged albums. Deterministic scripts, chunked and verify-everything. Use when the user asks to do anything with the "phone", the "phone library", phone albums/songs, or asks for covers/track info/lyrics on the phone. Do NOT use for one-off lyrics or split/tag jobs on desktop files (that's add-lyrics and split-flac-album), nor for quality measurement or restoration.
---

# Phone Music Library

Bulk operations on a phone music library over MTP (Samsung/Android via gvfs;
the music root is auto-detected, or set `MR_PHONE_DIR` in
`~/.config/music-restoration/config.env`). All scripts are deterministic; every
ambiguity is logged to a stragglers list for a human decision, never guessed.
With `--root /path` the same scripts audit and enrich any local library.

## Hard rules (each was learned the hard way — do not relearn them)

1. **The gvfs fuse mount is read-only in practice.** cp/touch/mutagen-save to
   it fail with "Operation not supported". Every write goes through
   `gio copy -p <local> "<mtp://… URL-encoded path>"` (see `rnext/phone/mtp.py`).
2. **Stage → edit locally → copy back → verify.** Never edit in place, and
   never trust a copy-back without reading the tag back from the phone.
3. **MTP transfers are serial.** Parallelizing gio does not help; parallelize
   the network phase instead (but see the Genius rule below).
4. **Genius rate-limits hard.** Sustained parallel requests put the IP into a
   1–2 h quarantine where even the browser-visible site returns 429 to
   scripts. `embed_lyrics.py` runs Genius strictly sequential (~1 req/s) with
   a 60 s cooldown and a persistent cache. Do not raise workers above 1–2.
5. Bulk sequential reads from MTP are fast (~20 MB/s, albums copy in minutes)
   but per-file opens through fuse are ~100× slower — always bulk-copy to
   local disk before per-file processing.
6. When deriving album identity from a staging path, remember the staged tree
   mirrors only the SUBTREE: `"Mountain" in str(path)` style checks silently
   mis-tag. Carry the album name as explicit data, never re-derive it from
   staged paths.
7. Phone player apps cache tags in their own DB — after overwriting files the
   player must rescan its library, else stale lyrics/covers show. Cure when a
   rescan is not enough: delete + re-add the file (a fresh file event forces
   re-index), or move the album to a NEW folder path. Never re-tag again.
8. MIXED discnumber presence across an album's files splits it into
   "Disk 0"/"Disk 1" in players — always set discnumber=1+disctotal on every file.

## Workflow 1 — Audit

```bash
S=~/.agents/skills/phone-music/scripts
python3 $S/inventory.py                        # whole library (~5 min / 2800 files)
python3 $S/inventory.py --dir "Place Vendome"  # one group
python3 $S/inventory.py --missing cover --missing tracknumber
```

Per file: lyrics, embedded cover, tracknumber, artist/title/album. Full TSV in
`~/.local/state/music-restoration/inventory.tsv`; summary prints per top-level
dir. Always start here and show the user the gaps before running bulk fixes.

## Workflow 2 — Lyrics (bulk, resume-safe)

```bash
S=~/.agents/skills/phone-music/scripts
python3 $S/embed_lyrics.py                     # everything missing lyrics
python3 $S/embed_lyrics.py --dir "Helloween"
python3 $S/embed_lyrics.py --force             # re-embed even if present
```

Chunked (~4 GB per chunk): stage → transplant from the local library
(`MUSIC_DIR`, normalized artist+title match — fast, no network) → Genius with
lrclib fallback for the rest → write tags locally → decode spot-check → copy
back only changed files → verify each on the phone. Run in background; watch
`status.tsv`. Afterwards sweep `stage/stragglers.tsv` and the
NOMATCH/AMBIGUOUS/ERROR lines:
- instrumentals (Genius page exists, no lyrics section; lrclib says
  instrumental) — leave without lyrics, report;
- AMBIGUOUS — show candidates, ask the user, re-run with the add-lyrics
  skill's `tag --url <URL> --force` on the staged copy;
- songs absent everywhere — report, don't invent.

## Workflow 3 — Covers + track numbers (album enrichment)

```bash
S=~/.agents/skills/phone-music/scripts
python3 $S/embed_covers.py --dir "Place Vendome" --dry-run
python3 $S/embed_covers.py --dir "Place Vendome"
python3 $S/embed_covers.py --album "<album dir name>"
python3 $S/embed_covers.py --all
```

Picks the cover from the album folder (Cover.jpg > Front.jpg > ...), downscales
scans > 600 KB to 1200 px via ffmpeg before embedding, sets TRACKNUMBER from
the filename's leading number plus TRACKTOTAL. Complete files are not touched.

## Workflow 4 — Untagged albums (needs judgment, no script)

When files have no artist/title at all: stage the folder, decide the correct
artist/album/date from folder names, filenames and the user, tag with mutagen,
embed lyrics via the add-lyrics script, copy back with `rnext.phone.mtp`, verify. Ask
the user when the album identity is unclear — do not guess.

Adding a NEW album from the computer: `gio mkdir -p` the album dir, embed
lyrics/covers into the local copies first (add-lyrics skill), then `gio copy
-p` each file and verify on the phone.

## Requirements

- Phone connected in MTP mode (`ls` on the detected root must work), or `--root`.
- `python3` + `mutagen`, `gio`, `flac`, `metaflac`, `ffmpeg`; the add-lyrics
  skill from this repo must be installed alongside.
- Free disk space ≥ chunk size (default 4 GB) under the state dir.
