#!/usr/bin/env bash
# Resume work after a reboot: mount the flash drive, tune Transmission,
# start the enrichment loop, training and the player. Idempotent — safe to
# run any number of times.
#
#   scripts/resume.sh             # bring up everything possible (NO player)
#   scripts/resume.sh --with-app  # plus the player for listening
set -u
cd "$(dirname "$0")/.."
ROOT="$PWD"
DATA="/media/mydum/Data"
DATA_ROOT="$DATA/Music/restore"

say() { printf '\n\033[1m%s\033[0m\n' "$*"; }

say "1/5 flash drive"
if mountpoint -q "$DATA"; then
  echo "already mounted: $DATA"
else
  echo "not mounted — trying udisksctl"
  if ! udisksctl mount -b /dev/sda5 2>&1 | tail -2; then
    cat <<'MSG'
MOUNT FAILED. Most likely NTFS is flagged "dirty" after a crash.
Run this once (password required):

    sudo ntfsfix -d /dev/sda5
    udisksctl mount -b /dev/sda5

If that does not help either:

    sudo mount -t ntfs3 -o force,rw /dev/sda5 /media/mydum/Data

then run this script again.
MSG
    exit 1
  fi
fi
[ -d "$DATA_ROOT" ] || { echo "no $DATA_ROOT — is the partition mounted elsewhere?"; exit 1; }

say "2/5 Transmission"
pgrep -f "[t]ransmission-gtk" >/dev/null || {
  echo "starting transmission-gtk"; setsid nohup transmission-gtk >/tmp/transmission.log 2>&1 </dev/null &
  sleep 12
}
python3 scripts/transmission_tune.py --apply | tail -3

say "3/5 data integrity"
python3 scripts/dataset_audit.py --integrity | tail -3

say "4/5 enrichment loop"
if pgrep -f "[d]ataset_loop.py" >/dev/null; then
  echo "loop already running"
else
  # same parameters as watchdog.sh: aggressive ones (24 pairs / 16 GB) filled
  # up the partition — keep the conservative set, capped by live peers and free space
  setsid nohup python3 -u scripts/dataset_loop.py --interval 420 --batch-pairs 10 \
    --batch-gb 6 --keep-window 6 --stall-passes 1 --min-rate-mbps 2.0 \
    --min-free-gb 15 --max-inflight-gb 12 --max-master-gb 6 \
    >/tmp/dataset_loop.out 2>&1 </dev/null &
  sleep 5; echo "loop started (log /tmp/dataset_loop.out)"
fi

say "5/5 training"
python3 scripts/ml_auto.py --force --with-audio 2>&1 | tail -20

# The player only starts on explicit request: it is needed at the final stage
# and gets in the way during a normal work resumption.
if [ "${1:-}" = "--with-app" ] && [ -d app/node_modules ]; then
  pgrep -f "music-restoration/app/node_modules/[e]lectron/dist" >/dev/null || {
    echo "starting the player"
    (cd app && setsid nohup ./node_modules/.bin/electron --no-sandbox --disable-gpu . \
       >/tmp/electron.log 2>&1 </dev/null &)
  }
fi

say "done"
python3 scripts/status.py | tail -3
