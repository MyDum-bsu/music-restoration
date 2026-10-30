---
name: split-flac-album
description: Split a single-FLAC album (one file per album) into individual tracks using an existing .cue sheet. Tags each track with number/title/artist/album/year/genre and embeds cover art (auto-detected in the album folder, or asked from the user if ambiguous). Use when the user says "split album", "split into tracks", or hands you an album folder with .flac + .cue + cover image. Do NOT use for albums already split into per-track files (plain re-tagging), for quality measurement (audio-measure), or for restoration (release-restore).
---

# Split FLAC Album

Split a single-FLAC-per-album into individual tagged tracks with cover art.

## Requirements

- `shntool`, `cuetag`, `flac`, `metaflac`, `ffmpeg` — all must be installed.
  shntool/cuetag are Unix tools: easy on Linux, finicky on Windows/macOS —
  there an agent can substitute `ffmpeg -ss/-to` per CUE index for splitting
  and write tags with mutagen; the steps below assume the standard tools.
- Album directory must contain:
  - One `.flac` file (whole album)
  - One `.cue` file (track listing)
  - Some cover image (any name/place — see Step 5)

## Process

### 1. Split by CUE

```bash
cd "/path/to/album/dir"
shntool split -f "*.cue" -o flac -t "%n - %t" "*.flac"
```

This creates `01 - Song Title.flac`, `02 - Song Title.flac` etc.

### 2. Fix filenames with problematic characters

Some track titles contain `/` (e.g. "Gary Moore feat. Ozzy Osbourne / Led Clones"). Replace with ` - `:

```bash
for f in */*.flac; do
  dir=$(dirname "$f")
  base=$(basename "$f")
  newbase=$(echo "$base" | sed 's:/: - :g')
  if [ "$base" != "$newbase" ]; then
    mv "$f" "$dir/$newbase"
  fi
done
```

If files ended up in subdirs (due to `/`), move them up:

```bash
for d in */; do
  if [ -d "$d" ]; then
    mv "$d"*.flac . 2>/dev/null || true
    rmdir "$d" 2>/dev/null || true
  fi
done
```

### 3. Tag from CUE (track number, title, artist)

```bash
cuetag "*.cue" *.flac
```

### 4. Set album-level tags (album, year, genre)

Parse from the CUE file or set manually:

```bash
album=$(grep '^TITLE ' "*.cue" | head -1 | sed 's/^TITLE "\(.*\)"/\1/')
year=$(grep 'REM DATE' "*.cue" | sed 's/REM DATE \(.*\)/\1/')
genre=$(grep 'REM GENRE' "*.cue" | sed 's/REM GENRE "\(.*\)"/\1/')

for f in [0-9]*.flac; do
  metaflac --set-tag="ALBUM=$album" \
           --set-tag="DATE=$year" \
           --set-tag="GENRE=$genre" "$f"
done
```

### 5. Embed cover art

**Locate the cover image — don't assume a fixed name.** Search the album directory and its `Covers/` subfolder for a likely front-cover image. Use the first match of this priority list:

1. `Covers/front_.jpg`
2. `folder.jpg` or `cover.jpg`
3. `Covers/front.jpg`
4. `front.jpg`, `Front.jpg`, `cover.jpg`, `Cover.jpg`
5. Any `*.jpg` in `Covers/` that looks like a front cover (prefer `front*`, then a single small file; avoid `back*`, `cd*`, `disc*`, `in*`, `mx*`, and numbered booklet scans like `1 001.jpg`)

If the candidate list is ambiguous (multiple plausible front covers, e.g. several `front*` variants), **ask the user** which file to embed (use the Question tool), showing the candidates.

After the cover file is chosen:

```bash
# skip tracks that were already tagged in Step 4 or Step 3
for f in [0-9]*.flac; do
  metaflac --import-picture-from="$cover_file" "$f"
done
```

Notes:
- Prefer a small image (~100-200 KB) — `front_.jpg` or `folder.jpg` — over multi-MB scans; large images bloat every track. If only big scans exist, either use the smallest, optionally downscale with `ffmpeg` first, or ask the user.
- If a track already has embedded art (e.g. from a prior run), you can re-embed the chosen cover with `--import-picture-from`, it replaces the first picture block.

### 6. Clean up original single-file sources

After split is complete, remove the original single-file FLAC and its associated files (cue, log, accurip, txt, dr):

```bash
rm -f "$flac_file" "$cue_file" "${flac_file%.flac}.log" "${flac_file%.flac}.accurip" "${flac_file%.flac}.txt" foo_dr.txt
```

### 7. Verify

```bash
ffprobe -v quiet -show_entries format_tags -of default=noprint_wrappers=1 "01 - Song Title.flac"
```

## Notes

- Original single-FLAC, CUE, LOG, ACCURIP, TXT and `foo_dr.txt` **are deleted** after successful split
- Only the per-track `NN - Title.flac` files and the cover image remain
- Tracks with no INDEX 00 pregap: INDEX 01 is the audio start (correct)
- Tracks with INDEX 00 + INDEX 01: shntool handles the pregap automatically
- Prefer a small cover thumbnail (~100-200 KB) for embedding; ask the user if only large scans are available
