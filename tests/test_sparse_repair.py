"""Observable consistency, identity and reconstruction properties."""
import numpy as np
import pytest
import json

from rnext.dsp.sparse import hard_clip_mask, repair_clipping, _ParsevalSTFT


def test_parseval_reconstruction_on_supported_interval():
    transform = _ParsevalSTFT(4096, 512)
    x = np.random.default_rng(7).normal(size=4096)
    y = transform.synthesis(transform.analysis(x))
    np.testing.assert_allclose(y[512:-512], x[512:-512], atol=1e-12)


@pytest.mark.parametrize('bits',[16,24])
@pytest.mark.parametrize('kind',['sine','noise'])
def test_quantized_clean_controls_are_not_clip_plateaus(bits,kind):
    t=np.arange(44100)/44100
    clean=(np.column_stack([.9*np.sin(2*np.pi*110*t)]*2) if kind=='sine'
           else np.random.default_rng(13).normal(size=(len(t),2))*.1)
    clean=np.round(clean*2**(bits-1))/2**(bits-1)
    mask,_=hard_clip_mask(clean.astype('float32'),bits)
    assert not mask.any()


def test_dithered_plateau_keeps_the_whole_clip_region():
    rng=np.random.default_rng(9)
    t=np.arange(20000)/44100
    clean=np.column_stack([.9*np.sin(2*np.pi*440*t)]*2)
    clipped=np.clip(clean,-.5,.5)
    q=2**23
    dither=rng.random(clipped.shape)+rng.random(clipped.shape)-1
    encoded=(np.round(clipped*q+dither)/q).astype('float32')
    mask,_=hard_clip_mask(encoded,24)
    truth=np.abs(clean)>.5
    assert np.mean(mask[truth])>.99
    assert not np.any(mask & (np.abs(clean)<.49))


def test_detector_finds_clips_below_fullscale_and_both_signs():
    t = np.arange(20000) / 44100
    source = np.column_stack([1.2*np.sin(2*np.pi*440*t)]*2).astype('float32')
    clipped = np.clip(source, -.6, .6)
    mask, diagnosis = hard_clip_mask(clipped, 24)
    json.dumps(diagnosis,allow_nan=False)
    assert np.count_nonzero(mask & (clipped > 0)) > 100
    assert np.count_nonzero(mask & (clipped < 0)) > 100
    assert not np.any(mask & (np.abs(clipped) < .59))


def test_identity_without_mask_is_exact():
    x = np.random.default_rng(1).normal(size=(5000, 2)).astype('float32') * .01
    y, _ = repair_clipping(x, 44100, np.zeros(x.shape, dtype=bool), frame=512, iterations=2)
    np.testing.assert_array_equal(x,y)


def test_repair_improves_known_sine_and_keeps_reliable_samples():
    t = np.arange(10000) / 44100
    clean = np.column_stack([.9*np.sin(2*np.pi*440*t)]*2)
    clipped = np.clip(clean,-.5,.5)
    mask = np.abs(clipped) >= .5
    repaired, report = repair_clipping(clipped,44100,mask,frame=1024,iterations=80)
    np.testing.assert_array_equal(repaired[~mask],clipped[~mask])
    assert np.all(repaired[mask & (clipped>0)] >= clipped[mask & (clipped>0)])
    assert np.all(repaired[mask & (clipped<0)] <= clipped[mask & (clipped<0)])
    assert np.mean((repaired-clean)**2) < np.mean((clipped-clean)**2)
    assert report['reliable_samples_preserved']


def test_chunk_seams_preserve_consistency():
    t=np.arange(9000)/8000
    clean=np.column_stack([np.sin(2*np.pi*220*t)]*2).astype('float32')
    clipped=np.clip(clean,-.6,.6)
    mask=np.abs(clipped)>=.6
    y,report=repair_clipping(clipped,8000,mask,frame=256,iterations=10,chunk_seconds=.3)
    assert len(report['blocks'])>2
    np.testing.assert_array_equal(y[~mask],clipped[~mask])
    assert np.isfinite(y).all() and y.shape==clipped.shape
    assert np.all(np.abs(y[mask])>=np.abs(clipped[mask]))


@pytest.mark.parametrize('channels',[1,2])
def test_auto_delivery_repairs_short_clipped_flac_and_preserves_native_format(tmp_path,channels):
    from rnext import features as F
    from scripts.restore_per_track import write_24
    from scripts.restore_safe import process
    sr=44100
    t=np.arange(8000)/sr
    clean=np.column_stack([.9*np.sin(2*np.pi*440*t)]*channels).astype('float32')
    clipped=np.clip(clean,-.5,.5)
    source,dest=tmp_path/'clipped.flac',tmp_path/'repaired.flac'
    write_24(clipped,sr,source,0)
    report=process(source,dest)
    json.dumps(report,allow_nan=False)
    assert report['status']=='processed_candidate'
    assert report['mode']=='declip'
    y,rate=F.decode_f32(str(dest),strict=True,channels=None)
    x,_=F.decode_f32(str(source),strict=True,channels=None)
    assert rate==sr and y.shape==x.shape
    assert F.ffprobe_audio(str(dest))['bps']==24
    assert F.ffprobe_audio(str(dest))['ch']==channels
    mask,_=hard_clip_mask(x,24)
    gain=10**(report['repair']['output_gain_db']/20)
    np.testing.assert_allclose(y[~mask],x[~mask]*gain,atol=2/8388608,rtol=0)
    assert report['encoded_metrics']['tp_dbfs']<=-.95
    assert report['repair']['changed_masked_samples']>0


def test_auto_preserves_quiet_natural_tone_without_a_diagnosed_defect():
    from rnext.dsp.repair import select_restoration
    sr=44100;t=np.arange(7*sr)/sr
    clean=np.column_stack([.03*np.sin(2*np.pi*440*t)]*2).astype('float32')
    y,report=select_restoration(clean,sr,mode='auto')
    np.testing.assert_array_equal(y,clean)
    assert report['status']=='kept_original'


def test_extreme_delivery_attenuation_falls_back_to_source(monkeypatch):
    from rnext.dsp import repair
    sr=44100;t=np.arange(3000)/sr
    x=np.column_stack([np.clip(.9*np.sin(2*np.pi*440*t),-.5,.5)]*2).astype('float32')
    mask,_=hard_clip_mask(x,24)
    proposed=x.copy();proposed[mask]=np.sign(x[mask])*100
    monkeypatch.setattr(repair,'periodic_ar_proposal',lambda *args:(x,{'accepted':False}))
    monkeypatch.setattr(repair,'repair_clipping',lambda *args,**kwargs:(proposed,{'method':'test_candidate','iterations_limit':160}))
    y,report=repair.select_restoration(x,sr,mode='declip')
    np.testing.assert_array_equal(x,y)
    assert report['status']=='qc_fallback_original'
    assert 'headroom_attenuation' in report['violations'][0]


@pytest.mark.parametrize('frequency',[55,110,440,1703])
def test_periodic_repair_does_not_regress_known_tones(frequency):
    from rnext.dsp.repair import select_restoration
    sr=44100;t=np.arange(sr)/sr
    clean=np.column_stack([.9*np.sin(2*np.pi*frequency*t)]*2).astype('float32')
    clipped=np.clip(clean,-.5,.5)
    y,report=select_restoration(clipped,sr,mode='declip',bits=24)
    gain=float(np.sum(clean.astype(float)*y)/np.sum(y.astype(float)**2))
    repaired_error=np.mean((clean-y*gain)**2)
    gain0=float(np.sum(clean.astype(float)*clipped)/np.sum(clipped.astype(float)**2))
    original_error=np.mean((clean-clipped*gain0)**2)
    assert repaired_error < original_error
    assert report['status']=='processed_candidate'
