"""Transmission RPC client: session, adding .torrent files, queue status."""
import base64
import json
from pathlib import Path

import requests

from rnext import secrets


class Transm:
    def __init__(self):
        self.rpc = secrets.transmission_rpc()
        self.url = self.rpc["url"]
        self.auth = (self.rpc["username"], self.rpc["password"])
        self.sid = None

    def _call(self, method: str, arguments: dict | None = None) -> dict:
        body = {"method": method, "arguments": arguments or {}}
        headers = {}
        if self.sid:
            headers["X-Transmission-Session-Id"] = self.sid
        r = requests.post(self.url, json=body, auth=self.auth, headers=headers, timeout=30)
        if r.status_code == 409:
            self.sid = r.headers["X-Transmission-Session-Id"]
            headers["X-Transmission-Session-Id"] = self.sid
            r = requests.post(self.url, json=body, auth=self.auth, headers=headers, timeout=30)
        r.raise_for_status()
        j = r.json()
        if j.get("result") != "success":
            raise RuntimeError(f"transmission rpc error: {j.get('result')}")
        return j["arguments"]

    # --- high level ---------------------------------------------------------
    def session(self) -> dict:
        return self._call("session-get")

    def add_torrent_file(self, path: str, download_dir: str, stopped: bool = False) -> dict:
        from .workspace import data_root, dir_of
        _ = data_root(), dir_of  # keep the import warm for logging elsewhere
        b64 = base64.b64encode(open(path, "rb").read()).decode()
        return self._call("torrent-add", {
            "metainfo": b64,
            "download-dir": download_dir,
            "paused": stopped,
        })

    def torrents(self, fields: list[str] | None = None) -> list[dict]:
        fields = fields or ["id","name","status","downloadedEver","sizeWhenDone",
                            "percentDone","totalSize","downloadDir","rateDownload","eta","error","errorString"]
        out = self._call("torrent-get", {"fields": fields})
        return out.get("torrents", [])

    def stats(self) -> tuple[int, int]:
        ts = self.torrents(["downloadedEverBytes", "sizeWhenDone"])
        return sum(t["downloadedEverBytes"] for t in ts), sum(t["sizeWhenDone"] for t in ts)


def check() -> dict:
    try:
        s = Transm().session()
        return {"ok": True, "version": s.get("version"), "download_dir": s.get("download-dir")}
    except Exception as e:
        return {"ok": False, "err": f"{e.__class__.__name__}: {e}"}
