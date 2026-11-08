"""Corpus preparation refuses corrupted caches and mismatched downloads."""
import hashlib
import io

import pytest

from research import fetch_declipping as corpus


def catalog(raw=b"test-waveform"):
    spec = {"path": "reference/example.wav", "bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(), "url": "https://example.org/audio.wav"}
    paired = {"path": spec["path"], "sha256": spec["sha256"]}
    return {"source": "test fixture", "files": [spec],
            "cases": [{"id": "example", "clean": paired, "degraded": paired}]}


def test_cached_corpus_is_verified_without_network(tmp_path, monkeypatch):
    source = tmp_path / "reference/example.wav"
    source.parent.mkdir()
    source.write_bytes(b"test-waveform")
    monkeypatch.setattr(corpus, "urlopen", lambda *a, **k: pytest.fail("unexpected network"))
    manifest = corpus.prepare(catalog(), tmp_path, cached_only=True)
    assert manifest["cases"][0]["clean"]["path"] == str(source)
    source.write_bytes(b"bad!-waveform")
    with pytest.raises(ValueError, match="hash mismatch"):
        corpus.prepare(catalog(), tmp_path, cached_only=True)
    assert source.read_bytes() == b"bad!-waveform"


@pytest.mark.parametrize("payload", [b"wrong-content", b"test-waveform-extra"])
def test_invalid_download_is_not_installed(tmp_path, monkeypatch, payload):
    monkeypatch.setattr(corpus, "urlopen", lambda *a, **k: io.BytesIO(payload))
    with pytest.raises(ValueError, match="hash mismatch"):
        corpus.prepare(catalog(), tmp_path)
    assert not (tmp_path / "reference/example.wav").exists()


def test_corpus_paths_cannot_escape_destination(tmp_path):
    with pytest.raises(ValueError):
        corpus._target(tmp_path, "../outside.wav")
