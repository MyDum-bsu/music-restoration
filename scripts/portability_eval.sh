#!/usr/bin/env bash
# portability_eval.sh — deterministic portability test for music-restoration.
#
# Simulates a fresh machine, end to end:
#   1. git clone of the repository into a temp dir;
#   2. ./install.sh there (no venv: system python3 already has the deps);
#      config/workspace.json must be CREATED by the bootstrap with the default
#      data_root and must NOT show up in git status;
#   3. python3 scripts/doctor.py --json — every HARD check green;
#   4. generate a synthetic killed track (sine + acompressor/alimiter:
#      DR <= 8, clipping present) as $TMP/in/track.flac;
#   5. python3 scripts/restore_per_track.py --src ... --out ... (release-restore
#      skill, step 4);
#   6. qc_gates.py + audio_measure.py on the result: DR grew by >= 2,
#      TP <= -0.3, clipping 0;
#   7. git status in the clone still clean after the install;
#   8. print PORTABILITY-OK and exit 0. Any failure: diagnostics + exit 1.
#
# Usage: bash scripts/portability_eval.sh [path-to-repo]
#        (default: /home/mydum/Desktop/projects/music-restoration)
#
# Implementation notes:
#   * The clone is `git clone` followed by a checkout of a snapshot commit of
#     the CURRENT working tree (built through a temporary index; the user's
#     real index and refs are untouched). This tests the tree as it lies on
#     disk right now — once the changes are committed, the snapshot equals
#     HEAD and the checkout is a no-op.
#   * The synthetic track is 100 s: scripts/restore_per_track.py hardcodes a
#     30 s excerpt start in gain_check(), so anything shorter crashes there
#     (verified: ValueError from sosfiltfilt on an empty excerpt).
#   * qc_gates.py runs without --orig: the skill's --orig mode matches source
#     takes by track number ("07 - ..."), while the track here is named
#     track.flac -> restored as "00. track.flac" with no numbered source take.
#     The absolute gate (TP, clipping, click-spikes) still applies in full.
set -uo pipefail

REPO="${1:-/home/mydum/Desktop/projects/music-restoration}"

say()  { printf '%s\n' "$*"; }
die()  { say "PORTABILITY-FAIL: $*"; KEEP_TMP=1; exit 1; }

have() { command -v "$1" >/dev/null 2>&1; }
for t in git python3 ffmpeg ffprobe; do
  have "$t" || die "required tool not found: $t"
done
[[ -d "$REPO/.git" ]] || die "not a git repository: $REPO"
REPO="$(cd "$REPO" && pwd)" || die "cannot enter $REPO"

TMP="$(mktemp -d)"
KEEP_TMP=0
trap '[[ $KEEP_TMP -eq 1 ]] && say "temp dir kept for inspection: $TMP" || rm -rf "$TMP"' EXIT
LOG="$TMP/logs"; mkdir -p "$LOG" "$TMP/in"
# deterministic defaults: the bootstrap must use the default data_root
unset RESTORE_DATA_ROOT
unset PYTHONPATH
export PYTHONDONTWRITEBYTECODE=1

say "== [1/7] clone $REPO -> $TMP/mr-fresh =="
# snapshot of the current working tree via a TEMPORARY index (real index untouched)
SNAP=$(GIT_INDEX_FILE="$TMP/portability.index" git -C "$REPO" add -A &&
       TREE=$(GIT_INDEX_FILE="$TMP/portability.index" git -C "$REPO" write-tree) &&
       git -C "$REPO" -c user.name=portability-eval -c user.email=portability@eval \
           commit-tree "$TREE" -p HEAD -m "portability-eval: working-tree snapshot") \
  || die "could not build a working-tree snapshot commit"
git clone --quiet "$REPO" "$TMP/mr-fresh" || die "git clone failed"
git -C "$TMP/mr-fresh" checkout --quiet "$SNAP" \
  || die "snapshot commit $SNAP missing in the clone"
CLONE="$TMP/mr-fresh"
say "   cloned at snapshot $(git -C "$CLONE" rev-parse --short HEAD)"

say "== [2/7] install.sh (no venv, skills into $TMP/agent-skills) =="
( cd "$CLONE" && env -u RESTORE_DATA_ROOT ./install.sh --skills-dir "$TMP/agent-skills" ) \
  > "$LOG/install.log" 2>&1 || { tail -20 "$LOG/install.log"; die "install.sh failed (full log: $LOG/install.log)"; }
grep -E "created config/workspace.json|ok      python3|All set" "$LOG/install.log" | sed 's/^/   /'
[[ -f "$CLONE/config/workspace.json" ]] || die "bootstrap did not create config/workspace.json"
python3 -I -c "
import json,sys
cfg = json.load(open('$CLONE/config/workspace.json'))
assert cfg['data_root'] == '$HOME/music-restoration-data', cfg['data_root']
print('   workspace.json data_root =', cfg['data_root'])
" || die "config/workspace.json has no default data_root"

say "== [3/7] doctor --json (hard checks must be green) =="
DOCTOR_RC=0
( cd "$CLONE" && python3 scripts/doctor.py --json ) > "$LOG/doctor.json" 2>"$LOG/doctor.err" || DOCTOR_RC=$?
# exit 0 = all green, exit 2 = only WARN checks failed (allowed); 1 = hard failure
[[ $DOCTOR_RC -eq 0 || $DOCTOR_RC -eq 2 ]] || { cat "$LOG/doctor.err"; die "scripts/doctor.py --json exited $DOCTOR_RC"; }
python3 -I - "$LOG/doctor.json" <<'PY' || die "doctor: HARD checks failed (see above)"
import json, sys
doc = json.load(open(sys.argv[1], encoding="utf-8"))
hard_fail = [c for c in doc["checks"] if c["severity"] == "hard" and c["status"] != "ok"]
for c in hard_fail:
    print(f"   FAIL [{c['id']}] {c['name']}: {c['detail']}  fix: {c['fix']}")
if hard_fail:
    sys.exit(1)
print(f"   all {sum(1 for c in doc['checks'] if c['severity']=='hard')} hard checks ok "
      f"(exit_code {doc['exit_code']}: warns allowed)")
PY

say "== [4/7] synthetic killed track -> $TMP/in/track.flac =="
ffmpeg -v error \
  -f lavfi -i "sine=frequency=440:duration=100" \
  -f lavfi -i "sine=frequency=4000:duration=100" \
  -filter_complex "[0:a]pan=stereo|c0=c0|c1=c0[b];[1:a]pan=stereo|c0=c0|c1=c0,volume='if(lt(mod(t,5),0.03),1,0)':eval=frame[tk];[b][tk]amix=inputs=2:duration=first:normalize=0,acompressor=threshold=0.1:ratio=20:attack=5:release=200:makeup=1,volume=22dB,alimiter=limit=1:level=false,volume=0.8dB" \
  -c:a pcm_f32le -y "$TMP/in/src.wav" \
  || die "ffmpeg: failed to render the synthetic track"
ffmpeg -v error -i "$TMP/in/src.wav" -c:a flac -y "$TMP/in/track.flac" \
  || die "ffmpeg: failed to encode the synthetic track to FLAC"
( cd "$CLONE" && python3 scripts/audio_measure.py --json "$TMP/logs/src.json" --quick "$TMP/in/track.flac" ) \
  > "$LOG/measure_src.log" 2>&1 || { cat "$LOG/measure_src.log"; die "audio_measure.py failed on the source"; }
python3 -I - "$LOG/src.json" <<'PY' || die "synthetic source misses the brief (DR<=8, clipping>0)"
import json, sys
r = json.load(open(sys.argv[1], encoding="utf-8"))[0]
print(f"   source: DR {r['dr']}, TP {r['tp_dbfs']} dBFS, clipped {r['clipped_samples']}, crest {r['crest_tt']} dB")
ok = r["dr"] <= 8 and r["clipped_samples"] > 0
sys.exit(0 if ok else 1)
PY

say "== [5/7] restore_per_track.py (release-restore skill, step 4; ~15 min) =="
( cd "$CLONE" && python3 scripts/restore_per_track.py --src "$TMP/in/track.flac" --out "$TMP/out" ) \
  > "$LOG/restore.log" 2>&1 || { tail -25 "$LOG/restore.log"; die "restore_per_track.py failed (full log: $LOG/restore.log)"; }
OUTFLAC=$(ls "$TMP/out"/*.flac 2>/dev/null | head -1)
[[ -n "$OUTFLAC" ]] || { tail -10 "$LOG/restore.log"; die "restore produced no FLAC"; }
grep -E "final|-> excerpt" "$LOG/restore.log" | tail -2 | sed 's/^/   /'

say "== [6/7] QC gates on the result =="
( cd "$CLONE" && python3 scripts/qc_gates.py "$TMP/out"/*.flac ) > "$LOG/qc.log" 2>&1
QC_RC=$?
grep -o '"verdict": "[A-Z]*"' "$LOG/qc.log" | sed 's/^/   qc_gates /'
[[ $QC_RC -eq 0 ]] || { cat "$LOG/qc.log"; die "qc_gates.py failed (exit $QC_RC)"; }
( cd "$CLONE" && python3 scripts/audio_measure.py --json "$LOG/after.json" --quick "$TMP/out" ) \
  > "$LOG/measure_after.log" 2>&1 || { cat "$LOG/measure_after.log"; die "audio_measure.py failed on the result"; }
python3 -I - "$LOG/src.json" "$LOG/after.json" <<'PY' || die "quality gates: DR must grow >= 2, TP <= -0.3, clipping 0"
import json, sys
src = json.load(open(sys.argv[1], encoding="utf-8"))[0]
res = json.load(open(sys.argv[2], encoding="utf-8"))[0]
delta = res["dr"] - src["dr"]
print(f"   result: DR {src['dr']} -> {res['dr']} (delta {delta:+d}), "
      f"TP {res['tp_dbfs']} dBFS, clipped {res['clipped_samples']}")
ok = delta >= 2 and res["tp_dbfs"] <= -0.3 and res["clipped_samples"] == 0
sys.exit(0 if ok else 1)
PY

say "== [7/7] git status in the clone must be clean =="
DIRTY=$(git -C "$CLONE" status --porcelain)
if [[ -n "$DIRTY" ]]; then
  say "$DIRTY" | sed 's/^/   /'
  die "git status in the clone is not clean after the install"
fi
say "   clean"

say "PORTABILITY-OK"
exit 0
