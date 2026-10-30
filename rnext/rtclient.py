"""rutracker HTTP client: cookie-jar session, Cloudflare challenge detection.

If a challenge appears, the call fails with a specific error; the upper layer
decides what to do (fallback: browser MCP, where Chrome passes CF).
"""
import re

import requests

UA = ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/139.0 Safari/537.36")

BASE = "https://rutracker.org"
FORUM = BASE + "/forum/"


class ChallengeError(RuntimeError):
    """Cloudflare "Just a moment..." — anonymous curl does not pass."""


class NotLoggedError(RuntimeError):
    """bb_data cookie missing/expired — no access to closed pages."""


class RTClient:
    def __init__(self, cookies: dict[str, str] | None = None, ua: str | None = None):
        if cookies is None:
            from rnext import secrets
            cookies = secrets.rutracker_cookies()
            ua = ua or secrets.rutracker_ua()
        if not cookies:
            raise NotLoggedError(
                "no cookies: put rutracker_cookies.txt and rutracker_ua.txt in "
                "~/.config/music-restoration/secrets/"
            )
        self.s = requests.Session()
        self.s.headers.update({"User-Agent": ua or UA})
        for name, value in cookies.items():
            self.s.cookies.set(name, value, domain="rutracker.org")

    # --- core ---------------------------------------------------------------
    def get(self, url: str, referer: str | None = None) -> requests.Response:
        h = {}
        if referer:
            h["Referer"] = referer
        r = self.s.get(url, headers=h, timeout=30)
        self._guard(r)
        return r

    def post(self, url: str, **kw) -> requests.Response:
        r = self.s.post(url, timeout=30, **kw)
        self._guard(r)
        return r

    def download_torrent(self, dl_url: str, to_path: str) -> str:
        """.torrent via the dl.php link (requires a logged-in session)."""
        r = self.get(dl_url, referer=FORUM)
        ct = r.headers.get("content-type", "")
        if "bittorrent" not in ct and not r.content.startswith(b"d8:"):
            raise RuntimeError(f"not a .torrent: {ct!r} len {len(r.content)}")
        open(to_path, "wb").write(r.content)
        return to_path

    # --- pages --------------------------------------------------------------
    def view_topic(self, topic_id: str):
        return self.get(FORUM + f"viewtopic.php?t={topic_id}")

    def poster_topics(self, pid: str, page: int = 1):
        return self.get(FORUM + f"tracker.php?pid={pid}&start={50*(page-1)}")

    def search(self, nm: str, forum_id: str | None = None, page: int = 1):
        q = f"tracker.php?nm={requests.utils.quote(nm)}&start={50*(page-1)}"
        if forum_id:
            q += f"&f={forum_id}&fonly=1"
        return self.get(FORUM + q)

    # --- guards -------------------------------------------------------------
    @staticmethod
    def _guard(resp: requests.Response) -> None:
        body = resp.text[:4000]
        if resp.status_code == 403 and "Just a moment" in body:
            raise ChallengeError("Cloudflare challenge; browser fallback needed")
        # ru markers: "Log in" / "You must log in" (login-page detection)
        if resp.url.startswith(FORUM) and re.search(
                r"\u0412\u0445\u043E\u0434|\u0412\u044B \u0434\u043E\u043B\u0436\u043D\u044B \u0432\u043E\u0439\u0442\u0438|log in",
                body[:2000]):
            raise NotLoggedError("no rutracker login (expired bb_data?)")
