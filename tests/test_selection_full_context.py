"""Selection must assess the same contextual waveform that it can deliver."""
import numpy as np

from rnext.dsp import selection as S
from rnext.dsp.pipeline import Policy


def _compressed_fixture():
    sr=4000
    clean=(np.random.default_rng(713).normal(size=(24*sr,2))*.12).astype('float32')
    clean[:sr//4]*=.1
    clean[-sr//4:]*=.1
    compressed=(.5*np.tanh(4*clean)).astype('float32')
    return compressed,clean,sr


def test_full_context_repair_is_not_vetoed_by_cut_processing(monkeypatch):
    source,clean,sr=_compressed_fixture()
    orig=S.measure(source,sr);rest=S.measure(clean,sr)
    target=S.material_plan(orig,len(source)/sr)['target']
    assert target>0
    assert not S.output_violations(source,clean,sr,orig,rest)
    assert S.compare(rest,orig,target)['score']<target
    monkeypatch.setattr(S,'candidate_policies',lambda plan:{'original':None,'context_repair':Policy()})
    def processor(pcm,*args,**kwargs):
        if len(pcm)==len(source):
            return clean.copy()
        # Standalone cuts have different calibration/boundary behavior.
        bad=pcm.copy();bad[0]=.98
        return bad
    monkeypatch.setattr(S,'restore_track',processor)
    result,report=S.select_track(source,sr)
    assert report['status']=='processed_candidate'
    np.testing.assert_array_equal(result,clean)


def test_no_better_candidate_is_unresolved_not_a_clean_bill(monkeypatch):
    source,_,sr=_compressed_fixture()
    monkeypatch.setattr(S,'candidate_policies',lambda plan:{'original':None,'no_change':Policy()})
    monkeypatch.setattr(S,'restore_track',lambda pcm,*args,**kwargs:pcm.copy())
    result,report=S.select_track(source,sr)
    np.testing.assert_array_equal(result,source)
    assert report['status']=='kept_original'
    assert report['restoration_state']=='no_acceptable_candidate'
    assert report['needs_review'] is True


def test_invalid_full_candidate_does_not_hide_a_valid_alternative(monkeypatch):
    source,clean,sr=_compressed_fixture()
    monkeypatch.setattr(S,'candidate_policies',lambda plan:{'original':None,'invalid':Policy(k=.1),'valid':Policy(k=.2)})
    def processor(pcm,rate,pol,**kwargs):
        if pol.k==.1:
            result=pcm.copy();result[len(result)//2]=np.nan
            return result
        return clean.copy() if len(pcm)==len(source) else pcm.copy()
    monkeypatch.setattr(S,'restore_track',processor)
    result,report=S.select_track(source,sr)
    assert report['status']=='processed_candidate'
    assert report['chosen']=='valid'
    np.testing.assert_array_equal(result,clean)
