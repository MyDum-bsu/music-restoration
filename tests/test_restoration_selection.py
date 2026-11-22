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
@pytest.mark.parametrize('processor', ['measure','sparse_repair'])
def test_nonfinite_audio_rejected(value,processor):
    x = np.ones((20, 2), dtype='float32')
    x[5, 0] = value
    with pytest.raises(ValueError, match='finite'):
        if processor=='measure':
            measure(x, 44100)
        else:
            from rnext.dsp.sparse import repair_clipping
            repair_clipping(x,44100,np.zeros(x.shape,dtype=bool))


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
    def proposed(pcm,rate,**options):
        y=pcm*.5
        return y, {'status':'processed_candidate','orig':measure(pcm,rate),
                   'rest':measure(y,rate),'plan':{'target':1.},'violations':[]}
    monkeypatch.setattr(runner,'select_restoration',proposed)
    if reject_encoded:
        monkeypatch.setattr(runner,'output_violations',lambda *args,**kw: ['encoded_qc_failed'])
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


def test_encoded_qc_uses_dynamics_plan_ceiling(tmp_path, monkeypatch):
    """Decode-back QC must use the plan's own ceiling, not a fixed -1.0 dBTP."""
    from scripts.restore_per_track import write_24
    from scripts import restore_safe as runner
    sr=44100
    t=np.arange(sr)/sr
    x=np.column_stack([.1*np.sin(2*np.pi*440*t)]*2).astype('float32')
    y=(x*.5).astype('float32')
    source,dest=tmp_path/'source.flac',tmp_path/'output.flac'
    write_24(x,sr,source,1)
    seen=[]
    real=runner.output_violations
    monkeypatch.setattr(runner,'output_violations',
                        lambda *a,**kw: seen.append(kw) or real(*a,**kw))
    def proposed(pcm,rate,**options):
        return y, {'status':'processed_candidate','orig':measure(pcm,rate),
                   'rest':measure(y,rate),
                   'plan':{'class':'crushed','target':1.,'tp_ceiling':-0.3},
                   'violations':[]}
    monkeypatch.setattr(runner,'select_restoration',proposed)
    report=runner.process(source,dest)
    assert report['status']=='processed_candidate', report['violations']
    assert report['delivery_qc']['tp_ceiling_dbfs']==-0.3
    assert seen and seen[0]['tp_ceiling']==-0.3


def test_peak_between_ceilings_judged_by_effective_ceiling():
    """A delivery at -0.6 dBTP violates the -1.0 Preserve ceiling but fits the
    -0.3 'crushed' material ceiling: one limit, one tolerance, everywhere."""
    sr=44100
    t=np.arange(3*sr)/sr
    x=np.column_stack([.1*np.sin(2*np.pi*440*t)]*2).astype('float32')
    y=(x/np.abs(x).max()*10**(-.6/20)).astype('float32')
    orig=measure(x,sr); rest=measure(y,sr)
    strict=output_violations(x,y,sr,orig,rest,tp_ceiling=-1.0)
    relaxed=output_violations(x,y,sr,orig,rest,tp_ceiling=-0.3)
    assert 'true_peak_above_ceiling' in strict
    assert 'true_peak_above_ceiling' not in relaxed


def test_cli_declip_delivery_contract(tmp_path, monkeypatch):
    """Real FLAC through the CLI: defective Preserve at -1 dBTP, hashes and
    source integrity recorded."""
    from scripts.restore_per_track import write_24
    import hashlib
    sr=44100
    t=np.arange(20000)/sr
    clean=np.column_stack([.9*np.sin(2*np.pi*440*t)]*2).astype('float32')
    clipped=np.clip(clean,-.5,.5)
    src=tmp_path/'in'
    out=tmp_path/'out'
    src.mkdir()
    write_24(clipped,sr,src/'track.flac',0)
    before=hashlib.sha256((src/'track.flac').read_bytes()).hexdigest()
    monkeypatch.setattr(sys,'argv',['restore_safe','--src',str(src),'--out',str(out)])
    assert main()==0
    rows=json.loads((out/'report.json').read_text())
    row=rows[0]
    assert row['status']=='processed_candidate' and row['mode']=='declip'
    assert row['delivery_qc']=={'tp_ceiling_dbfs':-1.0,'tp_tolerance_db':0.05,
                                'tp_measurer':'polyphase_4x'}
    assert row['encoded_metrics']['tp_dbfs']<=-1.0+0.05
    assert row['source_sha256']==before
    assert hashlib.sha256((src/'track.flac').read_bytes()).hexdigest()==before
    assert row['output_sha256']==hashlib.sha256((out/'track.flac').read_bytes()).hexdigest()


def test_cli_e2e_all_delivery_classes(tmp_path, monkeypatch):
    """One batch over the active material classes: defective Preserve, kept
    identity classes and the dynamics branch all share the QC contract."""
    from scripts.restore_per_track import write_24
    sr=44100
    t=np.arange(30000)/sr
    src=tmp_path/'in'; out=tmp_path/'out'
    src.mkdir()
    # hard clip -> defective Preserve
    write_24(np.clip(np.column_stack([.9*np.sin(2*np.pi*440*t)]*2),-.5,.5).astype('float32'),
             sr,src/'clipped.flac',0)
    # quiet/healthy -> kept original (no supported defect, no dynamics need)
    write_24(np.column_stack([.03*np.sin(2*np.pi*220*t)]*2).astype('float32'),
             sr,src/'healthy.flac',0)
    # short -> short_or_silent identity
    write_24(np.column_stack([.2*np.sin(2*np.pi*330*t[:4000])]*2).astype('float32'),
             sr,src/'short.flac',0)
    # crushed dynamics material (limiter-saturated noise, no clip plateau)
    long_t=np.arange(7*sr)/sr
    rng=np.random.default_rng(5)
    write_24((.7*np.tanh(np.column_stack([rng.normal(size=7*sr)]*2)*2.5)).astype('float32'),
             sr,src/'crushed.flac',0)
    monkeypatch.setattr(sys,'argv',['restore_safe','--src',str(src),'--out',str(out)])
    assert main()==0
    rows={r['source'].split('/')[-1]:r for r in json.loads((out/'report.json').read_text())}
    assert set(rows)=={'clipped.flac','healthy.flac','short.flac','crushed.flac'}
    for name,row in rows.items():
        assert row['status'] in ('processed_candidate','kept_original',
                                 'qc_fallback_original'), row
        assert row['delivery_qc']['tp_tolerance_db']==0.05
        assert (out/name).read_bytes()
    assert rows['clipped.flac']['mode']=='declip'
    assert rows['clipped.flac']['delivery_qc']['tp_ceiling_dbfs']==-1.0
    crushed=rows['crushed.flac']
    if crushed['status']=='processed_candidate':
        assert crushed['delivery_qc']['tp_ceiling_dbfs']==-0.3
        assert crushed['encoded_metrics']['tp_dbfs']<=-0.3+0.05


def _clipped_burst_flac(tmp_path):
    """High-crest declip candidate: quiet track with clipped loud bursts.

    After repair the true peak sits at the -1 dBTP ceiling while integrated
    loudness stays low, so a -8 LUFS request cannot be met under the ceiling."""
    from scripts.restore_per_track import write_24
    sr=44100
    x=np.zeros((6*sr,2),dtype='float32')
    for start in range(0,6*sr,sr):
        tt=np.arange(2000)/sr
        burst=np.clip(.95*np.sin(2*np.pi*880*tt),-.6,.6)
        x[start:start+2000]=np.column_stack([burst]*2)
    source=tmp_path/'burst.flac'
    write_24(x,sr,source,0)
    return source


def test_loudness_target_unreachable_reports_limited_by_peak(tmp_path):
    """Requested -8 LUFS is louder than the -1 dBTP ceiling admits: the run
    reports limited_by_peak instead of pretending or failing the candidate."""
    from scripts.restore_safe import process
    source=_clipped_burst_flac(tmp_path)
    dest=tmp_path/'out.flac'
    report=process(source,dest,loudness_target=-8.)
    assert report['status']=='processed_candidate', report.get('violations')
    loud=report['loudness']
    assert loud['requested_lufs']==-8. and loud['result']=='limited_by_peak'
    assert loud['achieved_lufs']<-8.-.2
    assert loud['applied_gain_db']<loud['requested_gain_db']
    assert report['encoded_metrics']['tp_dbfs']<=-1.+.05


def test_loudness_target_within_headroom_is_met(tmp_path):
    """A reachable attenuation request reports met within +/-0.2 LU."""
    from scripts.restore_per_track import write_24
    from scripts.restore_safe import process
    source=_clipped_burst_flac(tmp_path)
    dest=tmp_path/'out.flac'
    base=process(source,tmp_path/'base.flac')
    assert base['status']=='processed_candidate'
    from scripts import audio_measure
    lufs,_=audio_measure.lufs_tp(tmp_path/'base.flac')
    assert lufs is not None
    target=lufs-2.  # 2 dB of attenuation always fits under the ceiling
    report=process(source,dest,loudness_target=target)
    loud=report['loudness']
    assert loud['result']=='met', loud
    assert abs(loud['achieved_lufs']-target)<=.2


def test_write_error_leaves_no_accepted_artifact(tmp_path):
    """A failed encode writes no output and no temp residue; nothing on disk
    can be mistaken for an accepted result."""
    import os
    from scripts.restore_per_track import write_24
    from scripts.restore_safe import process
    sr=44100
    t=np.arange(20000)/sr
    clipped=np.clip(np.column_stack([.9*np.sin(2*np.pi*440*t)]*2),-.5,.5).astype('float32')
    source=tmp_path/'clip.flac'
    write_24(clipped,sr,source,0)
    outdir=tmp_path/'out'
    outdir.mkdir()
    outdir.chmod(0o555)
    try:
        with pytest.raises(OSError):
            process(source,outdir/'clip.flac')
        leftovers=[p.name for p in outdir.iterdir()]
    finally:
        outdir.chmod(0o755)
    assert leftovers==[]


def test_listening_rejects_whole_song_against_excerpt(monkeypatch, tmp_path):
    from scripts import listen_pack as pack
    import random
    a,b=tmp_path/'a.flac',tmp_path/'b.flac'
    monkeypatch.setattr(pack.F, 'ffprobe_audio', lambda p: {'dur': 30. if p==str(a) else 240.})
    with pytest.raises(ValueError, match='equal-duration'):
        pack.build_item('item', {'original': a, 'ours': b}, tmp_path, -20, random.Random(1))
