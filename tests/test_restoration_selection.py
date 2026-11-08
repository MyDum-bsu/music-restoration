"""Regression scenarios for conservative delivery and PCM preservation."""
import json
import subprocess
import sys

import numpy as np
import pytest

from rnext import features as F
from rnext.dsp.pipeline import Policy, restore_track, hpf25
from rnext.dsp.declip import clip_positions
from rnext.dsp.selection import measure, select_track, compare, output_violations
from scripts.restore_safe import main


def test_declip_counts_negative_fullscale():
    assert clip_positions(np.full((10, 2), -32768, dtype=np.int16)) == 20


def test_declip_preserves_unrepaired_high_precision_and_hpf():
    sr = 44100
    t = np.arange(sr) / sr
    x = np.column_stack([.1 * np.sin(2*np.pi*90*t), .1 * np.sin(2*np.pi*130*t)]).astype('float32')
    x[200:215, 0] = .99999
    # Right channel contains no clipped samples: repair must preserve all bits.
    y, stages = restore_track(x, sr, Policy(auto_policy=False, contour_enabled=False),
                             return_stages=True)
    np.testing.assert_array_equal(stages['declip'][0][:, 1], x[:, 1])
    np.testing.assert_allclose(stages['hpf'], hpf25(stages['declip'][0], sr), atol=1e-7)


def test_disabled_hpf_preserves_low_frequency_content():
    sr = 44100
    t = np.arange(sr)/sr
    x = np.column_stack([.1*np.sin(2*np.pi*15*t)]*2).astype('float32')
    y = restore_track(x, sr, Policy(auto_policy=False, hpf_hz=0., contour_enabled=False, declip=False))
    np.testing.assert_array_equal(x,y)


def test_short_identity_is_exact_and_reports_no_listening_claim():
    rng = np.random.default_rng(6)
    x = (rng.normal(size=(10000, 2))*.02).astype('float32')
    y, report = select_track(x, 44100)
    np.testing.assert_array_equal(x, y)
    assert report['status'] == 'kept_original'
    assert report['listening_verdict'] == 'not_evaluated'


def test_one_peak_cannot_win_without_block_dynamics_improvement():
    original = {'crest_db': 9., 'dr_exact': 5., 'shape': [0., 0.]}
    boosted = {'crest_db': 15., 'dr_exact': 5., 'shape': [0., 0.]}
    assert compare(boosted, original, 5.3)['score'] == compare(original, original, 5.3)['score']


def test_matched_level_boundary_artifact_fails():
    sr = 44100
    x = np.full((sr, 2), .05, dtype='float32')
    y = x.copy()
    y[10] = .2
    orig, rest = measure(x, sr), measure(y, sr)
    assert 'boundary_peak_growth_over_3_db_at_matched_rms' in output_violations(x, y, sr, orig, rest)


@pytest.mark.parametrize('value', [np.nan, np.inf, -np.inf])
def test_nonfinite_audio_rejected(value):
    x = np.ones((20, 2), dtype='float32')
    x[5, 0] = value
    with pytest.raises(ValueError, match='finite'):
        measure(x, 44100)


def test_strict_decode_rejects_corrupt_input(tmp_path):
    path = tmp_path / 'broken.flac'
    path.write_bytes(b'fLaC' + b'broken'*40)
    with pytest.raises((ValueError, subprocess.CalledProcessError)):
        F.decode_f32(str(path), strict=True)


def test_runner_failure_is_persisted_and_nonzero(tmp_path, monkeypatch):
    source = tmp_path / 'broken.flac'
    source.write_bytes(b'bad audio')
    out = tmp_path / 'out'
    monkeypatch.setattr(sys, 'argv', ['restore_safe', '--src', str(source), '--out', str(out)])
    assert main() == 1
    report = json.loads((out/'report.json').read_text())
    assert report[0]['status'] == 'error'
    assert not (out/source.name).exists()


def test_runner_keeps_short_source_byte_for_byte(tmp_path):
    from scripts.restore_per_track import write_24
    from scripts.restore_safe import process
    sr=44100
    t=np.arange(sr)/sr
    x=np.column_stack([.1*np.sin(2*np.pi*440*t)]*2).astype('float32')
    source,dest=tmp_path/'source.flac',tmp_path/'output.flac'
    write_24(x,sr,source,1)
    report=process(source,dest)
    assert report['status']=='kept_original'
    assert source.read_bytes()==dest.read_bytes()


@pytest.mark.parametrize('reject_encoded', [False, True])
def test_encoded_roundtrip_and_failed_encoder_fallback(tmp_path, monkeypatch, reject_encoded):
    from scripts.restore_per_track import write_24
    from scripts import restore_safe as runner
    sr=44100
    t=np.arange(sr)/sr
    x=np.column_stack([.1*np.sin(2*np.pi*440*t)]*2).astype('float32')
    source,dest=tmp_path/'source.flac',tmp_path/'output.flac'
    write_24(x,sr,source,1)
    def proposed(pcm,rate):
        y=pcm*.5
        return y, {'status':'processed_candidate','orig':measure(pcm,rate),
                   'rest':measure(y,rate),'plan':{'target':1.},'violations':[]}
    monkeypatch.setattr(runner,'select_track',proposed)
    if reject_encoded:
        monkeypatch.setattr(runner,'output_violations',lambda *args: ['encoded_qc_failed'])
    report=runner.process(source,dest)
    if reject_encoded:
        assert report['status']=='qc_fallback_original'
        assert source.read_bytes()==dest.read_bytes()
        assert report['rest']==report['orig']
    else:
        assert report['status']=='processed_candidate', report['violations']
        decoded,rate=F.decode_f32(str(dest),strict=True)
        original,_=F.decode_f32(str(source),strict=True)
        assert rate==sr and decoded.shape==original.shape
        np.testing.assert_allclose(decoded, original*.5, atol=3/8388608, rtol=0)
        assert F.ffprobe_audio(str(dest))['bps']==24


def test_listening_pack_uses_one_common_peak_safe_loudness(monkeypatch):
    from scripts import listen_pack as pack
    from pathlib import Path
    import random
    readings = {'a': {'lufs': -10., 'tp_peak_dbfs': 0.},
                'b': {'lufs': -20., 'tp_peak_dbfs': -2.}}
    monkeypatch.setattr(pack.F, 'lufs', lambda p: readings[p])
    monkeypatch.setattr(pack.F, 'ffprobe_audio', lambda p: {'dur': 10.})
    targets = []
    def capture(src, dst, target):
        targets.append(target)
        return {}
    monkeypatch.setattr(pack, 'gain_to_lufs', capture)
    pack.build_item('test', {'orig': Path('a'), 'ours': Path('b')}, Path('/unused'), -14., random.Random(1))
    assert targets == pytest.approx([-19.1, -19.1])


def test_listening_rejects_whole_song_against_excerpt(monkeypatch, tmp_path):
    from scripts import listen_pack as pack
    import random
    a,b=tmp_path/'a.flac',tmp_path/'b.flac'
    monkeypatch.setattr(pack.F, 'ffprobe_audio', lambda p: {'dur': 30. if p==str(a) else 240.})
    with pytest.raises(ValueError, match='equal-duration'):
        pack.build_item('item', {'original': a, 'ours': b}, tmp_path, -20, random.Random(1))
