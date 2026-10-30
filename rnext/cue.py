"""CUE parser: track boundaries in integer samples (75 fps)."""


def parse_cue(path: str, file_key: str | None = None) -> list[dict]:
    """List of cue tracks: {n, title, start_s (in 75fps seconds), performer}.

    INDEX 01 mm:ss:ff — track start; ff — CD frames (1/75 s).
    When file_key is given, other FILE blocks are ignored: only tracks of the
    same file are taken. file_key is the basename of the image-format file.
    """
    out = []
    cur_file = None
    n = 0
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            lsp = line.strip()
            if lsp.startswith("FILE "):
                rest = lsp[5:].rsplit(" WAVE", 1)[0].strip().strip('"')
                cur_file = rest
                continue
            if file_key and cur_file and cur_file != file_key:
                continue
            if lsp.startswith("TRACK "):
                try:
                    n = int(lsp.split()[1])
                except ValueError:
                    n += 0
                out.append({"n": n, "title": "", "start_frames": 0, "performer": ""})
            elif lsp.startswith("INDEX 01") and out:
                mm, ss_, ff = lsp.split()[2].split(":")
                out[-1]["start_frames"] = int(mm) * 60 * 75 + int(ss_) * 75 + int(ff)
            elif lsp.startswith("TITLE"):
                if out:
                    out[-1]["title"] = lsp[6:].strip().rstrip('"').strip('"')
            elif lsp.startswith("PERFORMER") and out:
                out[-1]["performer"] = lsp[10:].strip().rstrip('"').strip('"')
    return out


def track_bounds(cue_tracks: list[dict], total_samples: int, sr: int = 44100) -> list[tuple[int, int]]:
    """Slices [start, end) in samples (end = start of the next track, up to end of file)."""
    starts = [int(round(t["start_frames"] / 75.0 * sr)) for t in cue_tracks]
    ends = starts[1:] + [total_samples]
    return list(zip(starts, ends))
