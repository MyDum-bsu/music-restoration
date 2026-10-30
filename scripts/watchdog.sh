#!/usr/bin/env bash
# Unattended-run watchdog: if the enrichment loop dies, brings it back up.
# Run detached:  setsid nohup scripts/watchdog.sh > /tmp/watchdog.log 2>&1 &
cd "$(dirname "$0")/.."
LOOP_ARGS="--interval 420 --batch-pairs 10 --batch-gb 6 --keep-window 6 --stall-passes 1 --min-rate-mbps 2.0 --min-free-gb 15 --max-inflight-gb 12 --max-master-gb 6"
while true; do
  if ! pgrep -f "[d]ataset_loop.py" >/dev/null; then
    echo "[$(date '+%d.%m %H:%M:%S')] loop not found — restarting"
    setsid nohup python3 -u scripts/dataset_loop.py $LOOP_ARGS >> /tmp/dataset_loop.out 2>&1 </dev/null &
    sleep 20
  fi
  # flash drive still mounted?
  if ! mountpoint -q /media/mydum/Data; then
    echo "[$(date '+%d.%m %H:%M:%S')] flash drive unmounted — trying to remount"
    mount /media/mydum/Data 2>&1 | tail -1
  fi
  sleep 300
done
