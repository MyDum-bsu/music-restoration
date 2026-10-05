# MTP / phone delivery — the hard-won rules

These are baked into `drlib/mtp.py`, `drlib/gate.py` and `restore_batch.py`;
this document explains *why*, for anyone debugging a transfer or writing new
tooling on top.

## The mount

- Android + Linux = MTP through gvfs. The mount appears at
  `/run/user/<uid>/gvfs/mtp:host=<DEVICE>/`. music-restoration auto-detects
  the Music dir on it; `MR_PHONE_DIR` / `MR_MTP_BASE` override.
- **The fuse mount is read-only in practice**: `cp`/`touch`/mutagen-save fail
  with "Operation not supported". Writes go through
  `gio copy -p <local> "mtp://<device>/<encoded path>"`.
- gio can only **overwrite** existing files, not create new ones — create dirs
  with `gio mkdir -p` first and push onto existing targets.
- URL-encode **each path segment separately** (`urllib.parse.quote(seg)`),
  never the whole path — encoding `/` as `%2F` breaks the URI. (This once made
  every nested album fail to push while top-level ones passed.)
- MTP/libmtp cannot rename files or directories. Rename/move = create new +
  copy + verify + delete old (verify by file count before deleting).
- Reads: fuse `cp` works, but `gio copy` reads are ~2× faster for bulk pulls.
  Per-file opens by ffmpeg through fuse are ~100× slower than bulk copies —
  always stage to local disk before per-file processing (this is what
  `scan_dr.py` and `restore_batch.py` do).

## Verification

- **Verify pushes by reading bytes back** (`gio cat` → md5), not by trusting
  fuse metadata: right after an overwrite the fuse view can serve stale data,
  and `metaflac` reads through the fuse can return stale tags too.
- `subprocess` + `text=True` on binary output (`gio cat` of a FLAC) crashes
  with UnicodeDecodeError; capture bytes and decode with `errors="replace"`
  (see `drlib/util.py:run`). This exact bug hit four different scripts once.
- Kill rules for background pipelines: killing the wrapper shell does NOT kill
  the python child — re-check `ps aux` for the python PID itself. A
  "stopped" batch once ran on for hours, overwriting albums in place.

## The player side

- Phone players cache tags in their own database. After overwriting files in
  place, the player shows stale info until a **full library rescan**.
- Cures that work without rescan: delete + re-add the file (a fresh file
  event forces re-index), or move the album to a NEW folder path (players
  index by path).
- Ghost duplicates (old + new path both indexed) only clear via a full media
  index rebuild (Android: clear Media Storage data, reboot) or the new-folder
  move.
- MIXED discnumber presence across an album splits it into "Disk 0"/"Disk 1"
  in players — set discnumber=1 + disctotal on every file.

## Process rules (why restore_batch.py looks the way it does)

- **Never restore in place.** One batch once read back post-overwrite (empty)
  tags and shipped 8 releases tagless. Always separate src/ and dst/ dirs.
- **Hard gate before every push**: flac -t, artist/album/title/tracknumber
  non-empty, embedded cover, duration + sample rate equal to source, and
  per-file md5 after push. Any failure = loud quarantine of the whole album,
  never a silent skip.
- **Staging dies right after a verified delivery** — the library is the
  archive; quarantine/ survives for inspection.
- When files get mixed in a folder, split by the `album` tag (source of
  truth), not by filename patterns.
