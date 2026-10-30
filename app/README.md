# restore-listen — blind comparison player

An Electron wrapper over the excerpt sets: it compares variants of processing for
the same spot in a track with **loudness automatically matched** (LUFS, not RMS),
with blind A/B/C labeling and recorded ratings.

## Running

```bash
cd app
npm install            # once (electron)
npm start              # opens the latest pack from dataset/listen/*
```

A pack is specified by a folder: `--pack=/path/to/pack` or the "Open pack…" button.

## How a pack is prepared

```bash
# excerpts for the pair are already in dataset/pairs/<key>/excerpts (dataset_extract --excerpts 1)
python3 scripts/listen_pack.py --all --limit 8 --out /media/.../dataset/listen/pack-002

# if we have our own run (restore_one), a third variant "ours" is added
python3 scripts/listen_pack.py --key 5220623/rel2 --ours-dir out/silver-side-up \
        --out /media/.../dataset/listen/pack-003
```

The generator normalizes every file to `--target-lufs` (−20 LUFS by default) via
ebur128 and lowers the gain if a peak approaches clipping. The assignment of
variants to A/B/C is random (seed is fixed); the answer key is in `pack.json`.

## What the player can do

* switch variants in sync **at the same playback position** (essential for A/B);
* hotkeys: `Space` — play/stop, `1/2/3` — variant, `←/→` — ±2 s,
  `L` — loop, `R` — show answer;
* rating "A is better / B is better / equal", written to `<pack>/ratings.json`,
  export to `ratings.csv`;
* item list with progress, a "N of M rated" counter.

## Notes

* On Linux without a properly configured `chrome-sandbox`, Electron starts with
  `--no-sandbox` (that is what `npm start` passes). This is a local app with no
  network access.
* Audio is served to the page over a custom `pack://` protocol — otherwise
  Chromium will not let a `file://` page read files from another folder.
* The pack lives on the flash drive (`dataset/listen/`) and never enters the repo.
