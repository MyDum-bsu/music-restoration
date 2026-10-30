---
name: add-lyrics
description: Fetch song lyrics from genius.com and embed them into the audio files' tags (not a separate .txt/.lrc file), so a phone player can display them. Works for one song, an album folder, or any selection of files. Supports FLAC, MP3, M4A, Ogg, Opus. Use when the user says "add lyrics", "embed lyrics into the tags", "put the song text into the metadata", or asks to enrich album/song info with lyrics. Do NOT use for bulk phone-library operations (that's phone-music) or for splitting image+cue albums (that's split-flac-album).
---

# Add Lyrics

Fetch lyrics from genius.com and embed them into the audio files' tags. No separate lyrics files are created.

Tags written (for maximum phone-player compatibility):
- FLAC / Ogg / Opus: `LYRICS` + `UNSYNCEDLYRICS` (vorbis comments)
- MP3: `USLT` frame
- M4A / MP4: `©lyr` atom

## Requirements

- `python3` with `mutagen` (check: `python3 -c "import mutagen"`)
- Network access to genius.com (plain curl with a browser User-Agent passes; if it returns 403, try again, then fall back — lrclib.net has a free JSON API and the script uses it automatically)

## Script

`scripts/genius_lyrics.py` (next to this file). Subcommands:

```bash
S=<path-to-repo>/skills/add-lyrics/scripts/genius_lyrics.py
# if the skill is installed via install.sh:
# S=~/.agents/skills/add-lyrics/scripts/genius_lyrics.py

# 1. Dry run: resolve matches for everything, write nothing
python3 $S tag --dir "/path/to/album" --dry-run
python3 $S tag "01 - Song.flac" --dry-run

# 2. Embed (skips files that already have lyrics)
python3 $S tag --dir "/path/to/album"
python3 $S tag --force "01 - Song.flac"     # --force overwrites existing lyrics

# 3. Resolve a single ambiguous file with a pinned page
python3 $S search "artist title"            # list candidates
python3 $S tag --url "https://genius.com/Artist-song-lyrics" "01 - Song.flac"

# 4. Just print lyrics
python3 $S fetch "Artist" "Song title"
python3 $S fetch --url "https://genius.com/Artist-song-lyrics"
```

Per-file status lines (`STATUS<TAB>file<TAB>detail`):
- `OK` — embedded (detail: genius URL or "lrclib.net fallback", char count, tags written)
- `SKIP` — file already has lyrics (use `--force`)
- `AMBIGUOUS` — several equally good candidates; resolve, don't guess
- `NOMATCH` — nothing matched (detail lists what was found)
- `ERROR` — unreadable tags, network failure, unsupported file

## Process

1. Resolve the target: single file, or every audio file in a folder.
2. Always run `--dry-run` first and read the statuses.
3. Embed with `tag` (add `--force` when replacing).
4. Verify the result, e.g. for FLAC:
   ```bash
   metaflac --show-tag=LYRICS "01 - Song.flac" | head -3   # text is there
   flac -t --totally-silent "01 - Song.flac"               # decode check
   ```

## Ambiguity and matching rules

- Before searching, `[bracketed]` annotations in tag titles (`[bonus track]`, `[cover]`, `[Michael Kiske version]`) are stripped — Genius titles don't carry them, and the live/remix/bonus demotion heuristic would otherwise kill every bonus-track match.
- The script only auto-picks a candidate when the artist matches exactly and the title matches after normalization (case/punctuation/parentheses-insensitive). Live / remix / demo / acoustic / instrumental versions are demoted automatically.
- `AMBIGUOUS`: run `search`, show the candidate URLs, ask the user to choose (or pick the obvious one from context — e.g. the album's studio version), then re-run with `tag --url <URL> <file> --force`.
- `NOMATCH`: the script automatically tries lrclib.net as a fallback source (plain lyrics only, strict artist + near-exact title match, typo-tolerant queries, instrumentals skipped). Detail lines say which source produced the text.
- Covers: if the covering artist has no Genius page, search the original artist (`search "Pantera P.S.T. 88"`) and pin with `--url`.
- Alternate/radio/live versions of the same song: pin the album-version URL with `--url` (lyrics are the same).
- Songs that stay `NOMATCH` with detail `page had no usable lyrics: <URL>` are usually instrumentals (Genius page exists, no lyrics section) — verify once, then report them as instrumentals instead of forcing wrong text.
- Track titles with typos in tags (e.g. `Dont't Stop`) are handled by the lrclib fallback's query variants; Genius side may still need a manual `search` + `--url`.
- Some artists simply have no Genius page (obscurities, many prog/melodic-rock bands) — lrclib may still have them; both empty = report, never invent.

## Notes

- Extracted lyrics keep Genius section markers like `[Chorus]` / `[Verse 1]` — they aid reading; don't strip them.
- The extractor reads `data-lyrics-container` divs, tracks nested divs properly, and suppresses the Genius header/bio block. A result under 40 chars is treated as a failed extraction (reported `NOMATCH`), never embedded.
- Genius HTML can change: if everything starts reporting NOMATCH with fetched pages, re-inspect one page's HTML and fix `LyricsExtractor` first.
- Multi-line tag values in FLAC are legal; mutagen writes them correctly. `metaflac --show-tag=LYRICS` prints them line by line.
- Embedding reuses FLAC metadata padding, so file size usually does not grow.
- Rate limits: keep bulk runs sequential (~1 req/s); see the phone-music skill's hard rules.
