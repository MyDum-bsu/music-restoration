"""CUE-парсер: границы треков в целых семплах (75 fps)._ всі"""


def parse_cue(path: str, file_key: str | None = None) -> list[dict]:
    """Список треков cue: {n, title, start_s (в 75fps секундах), performer}.

    INDEX 01 mm:ss:ff — старт трека; ff — CD-кадры (1/75 s).
    Баланс FILE-строк игнорируется, если file_key задан: берём только того же файла.
    file_key пути (baseнame) файла формата image.
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
    """Слайды [start, end) в семплах (end = старта следующего, до конца файла)."""
    starts = [int(round(t["start_frames"] / 75.0 * sr)) for t in cue_tracks]
    ends = starts[1:] + [total_samples]
    return list(zip(starts, ends))
