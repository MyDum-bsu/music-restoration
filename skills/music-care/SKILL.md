---
name: music-care
description: Router for every music skill of the music-restoration project — find and download the best release, measure DR/authenticity, restore an album or track, embed lyrics, split image+cue, fix covers, sync to the phone, build blind listening packs. Use when the user asks for anything music-related — even if they never name a skill: "download this album", "how's the quality", "embed lyrics", "check my phone", "restore this", "restoration status". Do NOT use for non-music torrents (tracker handles movies/books itself — hand over to it) or anything outside audio.
---

# music-care: one entry point for music

Users almost never phrase requests in skill names — identify the task yourself
and hand it to the right skill. Never do a skill's work by hand: each one ships
battle-tested scripts and a list of learned-the-hard-way rules.

| user task | skill | example phrases |
|---|---|---|
| find / compare / download a release, pick best quality | **tracker** | "find this album on the tracker", "which release is better", "download in the best quality" |
| measure DR/LUFS/TP, verify lossless authenticity | **audio-measure** | "how's the quality", "is this real lossless", "measure DR" |
| **restore an album/track, raise DR, full release** | **release-restore** | "restore this", "raise the DR", "save this crushed master" — including when the file is not on disk |
| split image+cue into tracks | **split-flac-album** | "split the image", "cut by cue" |
| song lyrics into tags | **add-lyrics** | "embed lyrics", "add lyrics" |
| phone library: inventory, covers, lyrics, delivery | **phone-music** | "check my phone", "push to the phone", "covers on the phone" |
| restoration project status | **restore-status** | "how are things", "status", "how many tracks" |
| blind comparison of restoration variants | **release-restore** (+ `listen_pack.py`) | "compare the variants", "build a listening pack" |

## The golden path

"Restore this" is the most common phrase, and it always means the full
**release-restore** cycle: source (a local file OR finding and downloading the
best quality via tracker) → verification → per-track restoration → QC → tags/
cover/lyrics → collection → phone on request → a report with numbers.
