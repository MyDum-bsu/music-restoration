"""The evaluation must reject wrong inputs before producing quality scores."""
import hashlib

import numpy as np
import pytest
from scipy.io import wavfile

from research.bench_declipping import _read, evaluate_case, reconstruction_metrics
from rnext import features as F


def test_reference_hash_is_verified_before_decode(tmp_path):
    source=tmp_path/'wrong.wav';source.write_bytes(b'not the pinned audio')
    with pytest.raises(ValueError,match='hash mismatch'):
        _read({'path':str(source),'sha256':'0'*64})


def test_wrong_severity_label_cannot_produce_a_benchmark_result(tmp_path):
    source=tmp_path/'clean.wav'
    time=np.arange(8000)/8000
    wavfile.write(source,8000,(.1*np.sin(2*np.pi*440*time)).astype('float32'))
    spec={'path':str(source),'sha256':hashlib.sha256(source.read_bytes()).hexdigest()}
    case={'id':'incorrect_severity','clean':spec,'degraded':spec,'expected_input_sdr_db':15}
    with pytest.raises(ValueError,match='wrong degradation'):
        evaluate_case(case,tmp_path)
    assert not (tmp_path/'incorrect_severity.wav').exists()


def test_scalar_headroom_is_separate_from_reconstruction_error():
    clean=np.random.default_rng(4).normal(size=(10000,2))*.1
    metrics=reconstruction_metrics(clean,clean*.5)
    assert metrics['relative_error']<1e-12
    assert metrics['gain_to_clean']==pytest.approx(2.)
    assert metrics['fixed_gain_sdr_db']==pytest.approx(6.0206,abs=.001)


@pytest.mark.parametrize('dtype,bits',[(np.int16,16),(np.float32,32)])
def test_pcm_wave_bit_depth_is_not_lost_from_probe(dtype,bits,tmp_path):
    source=tmp_path/'pcm.wav'
    wavfile.write(source,8000,np.zeros(8000,dtype=dtype))
    assert F.ffprobe_audio(str(source))['bps']==bits
