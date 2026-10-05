#!/usr/bin/env python3
"""genius_lyrics.py — fetch song lyrics from genius.com and embed them into audio tags.

Subcommands:
  search "<artist> <title>"     list top Genius candidates (title | artist | url)
  fetch --url <URL>             print cleaned lyrics for a known Genius song page
  fetch "<artist>" "<title>"    search, auto-pick best match, print lyrics
  tag FILE [FILE...]            read artist/title tags, fetch lyrics, embed them
      --dir FOLDER              process all *.flac *.mp3 *.m4a *.ogg *.opus in folder
      --force                   overwrite lyrics even if the file already has some
      --dry-run                 search and report decisions, write nothing
      --url URL                 pin the Genius page (skip search) — for resolving ambiguity

Tags written:
  FLAC / Ogg / Opus  -> LYRICS + UNSYNCEDLYRICS (vorbis comments)
  MP3                -> USLT frame
  M4A / MP4          -> ©lyr atom

Exit status per file is reported as: OK | SKIP | AMBIGUOUS | NOMATCH | ERROR.
On ambiguity the script never guesses: it lists candidate URLs and moves on,
so the caller can resolve them (ask the user) and re-run with --url.
"""

import argparse
import json
import re
import sys
import time
import urllib.parse
import urllib.request
from html.parser import HTMLParser
from pathlib import Path

UA = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/json;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}


# Politeness for genius.com: minimum spacing between requests and a global
# cooldown after any 429 (Cloudflare bans hard when hammered).
_last_genius_req = 0.0
_cooldown_until = 0.0
GENIUS_MIN_INTERVAL = 1.1
GENIUS_429_COOLDOWN = 60.0


def http_get(url, timeout=30, retries=4):
    global _last_genius_req, _cooldown_until
    last = None
    is_genius = "genius.com" in urllib.parse.urlparse(url).netloc
    for attempt in range(retries + 1):
        try:
            if is_genius:
                wait = _cooldown_until - time.time()
                if wait > 0:
                    time.sleep(wait)
                gap = GENIUS_MIN_INTERVAL - (time.time() - _last_genius_req)
                if gap > 0:
                    time.sleep(gap)
                _last_genius_req = time.time()
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read().decode("utf-8", "replace")
        except urllib.error.HTTPError as e:
            last = e
            if e.code == 429 and attempt < retries:
                if is_genius:
                    _cooldown_until = max(_cooldown_until, time.time() + GENIUS_429_COOLDOWN)
                ra = e.headers.get("Retry-After") if e.headers else None
                delay = int(ra) if ra and ra.isdigit() else min(10 * 2 ** attempt, 60)
                time.sleep(delay)
                continue
            raise
        except Exception as e:  # noqa: BLE001 - report any network failure
            last = e
            time.sleep(1.5)
    raise RuntimeError(f"fetch failed: {url} ({last})")


# ---------------------------------------------------------------- search ----

def search_songs(query):
    """Return a deduplicated list of song candidates from the public search API."""
    url = "https://genius.com/api/search/multi?q=" + urllib.parse.quote(query)
    data = json.loads(http_get(url))
    out, seen = [], set()
    for sec in data.get("response", {}).get("sections", []):
        for hit in sec.get("hits", []):
            if hit.get("type") != "song":
                continue
            r = hit["result"]
            c = {
                "title": r.get("title", ""),
                "artist": r.get("primary_artist", {}).get("name", ""),
                "url": r.get("url", ""),
            }
            if c["url"] and c["url"] not in seen:
                seen.add(c["url"])
                out.append(c)
    return out


def norm(s):
    s = s.casefold()
    s = re.sub(r"[()[\]{}]", " ", s)
    s = re.sub(r"[^a-z0-9а-яё]+", " ", s, flags=re.IGNORECASE)
    return " ".join(s.split())


DEMOTED = re.compile(
    r"\b(live|remix|demo|acoustic|instrumental|version|mix|edit|bonus|"
    r"re-?record\w*|rehearsal|session|radio|alternate|feat)\b"
)


def score_candidate(cand, artist, title):
    """Higher is better; 0 means unusable. Exact artist + title required for full score."""
    ca, ct = norm(cand["artist"]), norm(cand["artist"] + " " + cand["title"])
    want_a, want_t = norm(artist), norm(title)
    if not ca or ca != want_a:
        return 0
    c_title = norm(cand["title"])
    s = 0
    if c_title == want_t:
        s = 3
    elif re.sub(r"\b(and|&)\b", "", c_title) == re.sub(r"\b(and|&)\b", "", want_t):
        s = 2
    elif c_title.startswith(want_t) or want_t.startswith(c_title):
        s = 1
    if s and DEMOTED.search(c_title) and not DEMOTED.search(want_t):
        s -= 5
    return max(s, 0)


# ------------------------------------------------------- lrclib fallback ----

def lrclib_fetch(artist, title):
    """Fallback source: lrclib.net public API. Returns plain lyrics or None.

    Strict on purpose: artist must match (apostrophe-insensitive) and the
    track title must match exactly or near-exactly (typo tolerance ~0.85);
    instrumentals and empty texts are skipped. Query variants cover tag
    typos, and if none of them return a match the artist's whole track
    list is fetched once and matched locally, so a mistyped tag like
    "Dont't Stop" (extra letter) still finds "Don't Stop".
    """
    import difflib

    def query_variants(t):
        out = [t]
        stripped = re.sub(r"['’`]", "", t)
        if stripped != t:
            out.append(stripped)
        no_brackets = re.sub(r"[([].*[)\]]", "", stripped).strip()
        if no_brackets and no_brackets not in out:
            out.append(no_brackets)
        # collapse doubled letters: repairs typos like "Dont't" -> "Dont Stop"
        collapsed = re.sub(r"(.)\1+", r"\1", no_brackets)
        if collapsed and collapsed not in out:
            out.append(collapsed)
        return out

    def collect(query_params, pool):
        q = urllib.parse.urlencode(query_params)
        try:
            data = json.loads(http_get("https://lrclib.net/api/search?" + q, timeout=20))
        except Exception:  # noqa: BLE001 - fallback must never raise
            return
        if isinstance(data, list):
            for r in data:
                if isinstance(r, dict) and r.get("artistName") and r.get("trackName"):
                    pool[(r["artistName"], r["trackName"])] = r

    pool = {}
    for q_title in query_variants(title):
        collect({"artist_name": artist, "track_name": q_title}, pool)
    if not pool:
        collect({"artist_name": artist}, pool)

    wa = norm(artist).replace("'", "")
    wt = norm(title).replace("'", "")
    best = None
    for r in pool.values():
        if r.get("instrumental"):
            continue
        plain = (r.get("plainLyrics") or "").strip()
        if len(plain) < 40:
            continue
        if norm(r.get("artistName", "")).replace("'", "") != wa:
            continue
        rt = norm(r.get("trackName", "")).replace("'", "")
        if rt == wt:
            score = 3.0
        else:
            ratio = difflib.SequenceMatcher(None, wt, rt).ratio()
            if ratio < 0.85:
                continue
            score = 2.0 + ratio
        if best is None or score > best[0]:
            best = (score, plain)
    return best[1] if best else None


def clean_search_title(t):
    """Drop [bracketed] annotations (bonus track, cover, version...) from a tag
    title before searching/comparing — Genius titles don't carry them."""
    t = re.sub(r"\[[^\]]*\]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def decide(cands, artist, title):
    """-> ("MATCH", url) | ("AMBIGUOUS", [cand, ...]) | ("NOMATCH", top_cands)"""
    scored = sorted(
        ((score_candidate(c, artist, title), c) for c in cands),
        key=lambda t: t[0], reverse=True,
    )
    top = [c for s, c in scored if s > 0]
    if not top:
        return "NOMATCH", cands[:5]
    best = [c for s, c in scored if s == scored[0][0]]
    if len(best) > 1:
        return "AMBIGUOUS", best
    return "MATCH", top[0]


# --------------------------------------------------------------- lyrics -----

VOID_TAGS = {"br", "img", "meta", "input", "hr", "link", "source", "wbr", "area", "base", "col", "embed", "param", "track"}


class LyricsExtractor(HTMLParser):
    """Collect text inside <div data-lyrics-container="true">; <br>/</p> become newlines.

    - Tracks a stack of open <div>s, so nested divs inside a lyrics container
      cannot end the capture early.
    - Suppresses the page header ("N Contributors", "{Title} Lyrics", the bio
      preview) which Genius nests inside the first lyrics container with
      LyricsHeader* classes.
    """

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.div_stack = []   # per open <div>: is it a lyrics container?
        self.supp_stack = []  # per open element: does it suppress capture?
        self.parts = []
        self.containers = 0

    def _inside(self):
        return any(self.div_stack)

    def _suppressed(self):
        return any(self.supp_stack)

    def handle_starttag(self, tag, attrs):
        d = dict(attrs)
        if tag not in VOID_TAGS:
            self.supp_stack.append("LyricsHeader" in (d.get("class") or ""))
        if tag == "div":
            is_container = d.get("data-lyrics-container") == "true"
            self.div_stack.append(is_container)
            if is_container:
                self.containers += 1
                self.parts.append("\n")
        elif tag == "br":
            if self._inside() and not self._suppressed():
                self.parts.append("\n")
        elif tag == "p":
            if self._inside() and not self._suppressed():
                self.parts.append("\n")

    def handle_startendtag(self, tag, attrs):
        if tag == "br" and self._inside() and not self._suppressed():
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag == "div" and self.div_stack:
            self.div_stack.pop()
            self.parts.append("\n")
        elif tag == "p" and self._inside() and not self._suppressed():
            self.parts.append("\n")
        if tag not in VOID_TAGS and self.supp_stack:
            self.supp_stack.pop()

    def handle_data(self, data):
        if self._inside() and not self._suppressed():
            self.parts.append(data)


HEADER_JUNK = re.compile(r"^\d+\s+Contributors?\b")
BRACKET_HEADER = re.compile(r"^\[?lyrics to\b.*$", re.IGNORECASE)
TAIL_JUNK = {"embed", "you might also like", "how to format lyrics:", "copy", "edit", "translate"}


def clean_lyrics_lines(lines):
    # page header: "21 Contributors...To Be Loved Lyrics", "Lyrics", "{Title} Lyrics",
    # "[Lyrics to «Title»]" (some templates), trailing "Read More"
    while lines:
        first = lines[0].strip()
        if (HEADER_JUNK.match(first) or BRACKET_HEADER.match(first)
                or first.lower() in ("lyrics", "text")
                or first.endswith(("Lyrics", "Read More")) and len(first) <= 200):
            lines.pop(0)
        else:
            break
    while lines:
        last = lines[-1].strip()
        if (last.lower() in TAIL_JUNK or BRACKET_HEADER.match(last)
                or last.endswith(("Lyrics", "Read More")) and len(last) <= 200):
            lines.pop()
        else:
            break
    return [ln.rstrip() for ln in lines]


def extract_lyrics(html_text):
    p = LyricsExtractor()
    p.feed(html_text)
    raw = "".join(p.parts).replace("\r", "")
    lines, out, blank = raw.split("\n"), [], 0
    for ln in lines:
        ln = ln.strip()
        if ln:
            out.append(ln)
            blank = 0
        else:
            blank += 1
            if blank == 1 and out:
                out.append("")
    while out and out[-1] == "":
        out.pop()
    return "\n".join(clean_lyrics_lines(out))


def fetch_lyrics(url):
    return extract_lyrics(http_get(url))


# ---------------------------------------------------------------- embed -----

def embed(path, text):
    """Write lyrics into the file's tags. Returns the tag names written."""
    from mutagen import File as MutagenFile
    from mutagen.flac import FLAC
    from mutagen.mp3 import MP3
    from mutagen.mp4 import MP4
    from mutagen.oggopus import OggOpus
    from mutagen.oggvorbis import OggVorbis
    from mutagen.id3 import ID3, USLT

    ext = path.suffix.lower()
    if ext in (".flac", ".ogg", ".oga"):
        f = FLAC(path) if ext == ".flac" else OggVorbis(path)
        f["LYRICS"] = text
        f["UNSYNCEDLYRICS"] = text
        f.save()
        return "LYRICS + UNSYNCEDLYRICS"
    if ext == ".opus":
        f = OggOpus(path)
        f["LYRICS"] = text
        f["UNSYNCEDLYRICS"] = text
        f.save()
        return "LYRICS + UNSYNCEDLYRICS"
    if ext == ".mp3":
        audio = MP3(path)
        if audio.tags is None:
            audio.add_tags()
        audio.tags.setall("USLT", [USLT(encoding=3, lang="eng", desc="", text=text)])
        audio.save()
        return "USLT"
    if ext in (".m4a", ".m4b", ".mp4"):
        audio = MP4(path)
        audio["\xa9lyr"] = [text]
        audio.save()
        return "©lyr"
    # last resort: let mutagen figure it out (vorbis-style comment if supported)
    f = MutagenFile(path)
    if f is None or not hasattr(f, "tags") or f.tags is None or not hasattr(f.tags, "keys"):
        raise RuntimeError(f"unsupported file type: {ext}")
    f["LYRICS"] = text
    f.save()
    return "LYRICS"


def read_artist_title(path):
    from mutagen import File as MutagenFile
    f = MutagenFile(path, easy=True)
    if f is None:
        raise RuntimeError("mutagen cannot read this file")
    artist = (f.get("artist") or [""])[0]
    title = (f.get("title") or [""])[0]
    has_lyrics = bool(f.get("lyrics") or f.get("unsyncedlyrics"))
    return artist.strip(), title.strip(), has_lyrics


# ------------------------------------------------------------------ cli -----

def report(status, path, detail=""):
    print(f"{status}\t{path}\t{detail}", flush=True)


def cmd_tag(files, force=False, dry_run=False, pinned_url=None):
    for path in files:
        path = Path(path)
        try:
            artist, title, has = read_artist_title(path)
        except Exception as e:  # noqa: BLE001
            report("ERROR", path, f"read tags: {e}")
            continue
        if not artist or not title:
            report("ERROR", path, "missing artist/title tags")
            continue
        if has and not force:
            report("SKIP", path, "already has lyrics (use --force to overwrite)")
            continue
        try:
            title = clean_search_title(title)
            artist = clean_search_title(artist) or artist
            if pinned_url:
                status, payload = "MATCH", {"url": pinned_url, "title": title, "artist": artist}
            else:
                cands = search_songs(f"{artist} {title}")
                status, payload = decide(cands, artist, title)
            if status == "NOMATCH" and not pinned_url:
                fb = lrclib_fetch(artist, title)
                if fb:
                    if dry_run:
                        report("DRYRUN", path, f"lrclib.net fallback ({len(fb)} chars)")
                        continue
                    tags = embed(path, fb)
                    report("OK", path, f"lrclib.net fallback ({len(fb)} chars, {tags})")
                    continue
            if status == "NOMATCH":
                names = "; ".join(f"{c['full_title'] if 'full_title' in c else c['title']} ({c['url']})" for c in payload)
                report("NOMATCH", path, names or "no candidates at all")
                continue
            if status == "AMBIGUOUS":
                urls = " ;; ".join(c["url"] for c in payload)
                report("AMBIGUOUS", path, urls)
                continue
            lyrics = fetch_lyrics(payload["url"])
            if len(lyrics) < 40:
                report("NOMATCH", path, f"page had no usable lyrics: {payload['url']}")
                continue
            if dry_run:
                report("DRYRUN", path, f"{payload['url']} ({len(lyrics)} chars)")
                continue
            tags = embed(path, lyrics)
            report("OK", path, f"{payload['url']} ({len(lyrics)} chars, {tags})")
        except Exception as e:  # noqa: BLE001
            report("ERROR", path, str(e))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("search")
    s.add_argument("query")

    f = sub.add_parser("fetch")
    f.add_argument("artist", nargs="?")
    f.add_argument("title", nargs="?")
    f.add_argument("--url")

    t = sub.add_parser("tag")
    t.add_argument("files", nargs="*")
    t.add_argument("--dir")
    t.add_argument("--force", action="store_true")
    t.add_argument("--dry-run", action="store_true")
    t.add_argument("--url")

    a = ap.parse_args()
    if a.cmd == "search":
        for c in search_songs(a.query):
            print(f"{c['title']}\t{c['artist']}\t{c['url']}")
    elif a.cmd == "fetch":
        if a.url:
            print(fetch_lyrics(a.url))
            return
        if not (a.artist and a.title):
            ap.error("fetch needs --url, or both artist and title")
        cands = search_songs(f"{a.artist} {a.title}")
        status, payload = decide(cands, a.artist, a.title)
        if status != "MATCH":
            names = "; ".join(c["url"] for c in (payload if isinstance(payload, list) else [payload]))
            print(f"{status}: {names}", file=sys.stderr)
            sys.exit(2 if status == "AMBIGUOUS" else 3)
        print(fetch_lyrics(payload["url"]))
    else:
        files = list(a.files)
        if a.dir:
            exts = {".flac", ".mp3", ".m4a", ".m4b", ".mp4", ".ogg", ".oga", ".opus"}
            files += sorted(
                p for p in Path(a.dir).rglob("*")
                if p.is_file() and p.suffix.lower() in exts
            )
        if not files:
            ap.error("tag needs files or --dir")
        cmd_tag(files, force=a.force, dry_run=a.dry_run, pinned_url=a.url)


if __name__ == "__main__":
    main()
