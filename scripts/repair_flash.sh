#!/usr/bin/env bash
# Restore access to the flash drive after a hard power-off.
#
# What it does: finds the NTFS partition by label ("Data" by default), checks
# whether it is already mounted, clears the "dirty" flag (ntfsfix -d) and mounts.
# If the partition is not in fstab and udisks refuses — falls back to mount with force.
#
# Privileges: one privileged step is needed. The script picks pkexec itself
# (graphical password prompt) or sudo (when run from a terminal).
#
#   scripts/repair_flash.sh              # find "Data" and repair
#   scripts/repair_flash.sh --label MD   # a different partition
#   scripts/repair_flash.sh --check      # diagnostics only
set -u

LABEL="Data"
MOUNT="/media/mydum/Data"
MODE="repair"
while [ $# -gt 0 ]; do
  case "$1" in
    --label) LABEL="$2"; shift 2;;
    --mount) MOUNT="$2"; shift 2;;
    --check) MODE="check"; shift;;
    *) echo "unknown argument: $1"; exit 2;;
  esac
done

say() { printf '\n\033[1m%s\033[0m\n' "$*"; }
have() { command -v "$1" >/dev/null 2>&1; }

say "partitions of this disk"
lsblk -o NAME,SIZE,FSTYPE,LABEL,MOUNTPOINT 2>/dev/null | sed 's/^/  /'

DEV="$(lsblk -pnro NAME,LABEL | awk -v L="$LABEL" '$2==L {print $1; exit}')"
case "$DEV" in /dev/*) ;; *) DEV="/dev/$DEV";; esac
if [ -z "$DEV" ]; then
  echo "no partition with label \"$LABEL\" found"; exit 1
fi
echo "partition: $DEV (label $LABEL), target mount point: $MOUNT"

if mountpoint -q "$MOUNT" 2>/dev/null; then
  echo "already mounted at $MOUNT — nothing to do"
  ls "$MOUNT" | head -5
  exit 0
fi
[ "$MODE" = "check" ] && exit 0

say "attempt 1: plain mount via udisks"
if have udisksctl && udisksctl mount -b "$DEV" 2>&1 | tail -2; then
  if mountpoint -q "$MOUNT"; then echo "mounted"; ls "$MOUNT" | head -5; exit 0; fi
fi

# Privileged step: ntfsfix -d clears the "dirty" flag (the volume was not
# unmounted because of the power-off) — after that the ntfs3 kernel driver mounts.
priv() {
  if have pkexec && pgrep -f "[p]olkit.*agent|gnome-shell" >/dev/null 2>&1; then
    echo "→ requesting privileges via pkexec (a password window will appear)"
    pkexec "$@"
  elif have sudo; then
    echo "→ requesting privileges via sudo (enter the password)"
    sudo "$@"
  else
    echo "neither pkexec nor sudo available — run manually: $*"; return 1
  fi
}

say "attempt 2: ntfsfix -d (clear the \"dirty\" flag)"
if have ntfsfix; then
  priv ntfsfix -d "$DEV" 2>&1 | tee /tmp/ntfsfix.out | tail -6
  if grep -qi "corrupt\|chkdsk" /tmp/ntfsfix.out; then
    say "attempt 2b: ntfsfix -b (boot sector) — the volume is flagged as damaged"
    priv ntfsfix -b -d "$DEV" 2>&1 | tee -a /tmp/ntfsfix.out | tail -6
  fi
else
  echo "ntfsfix not installed (ntfs-3g package)"
fi

say "attempt 2c: read-only mount (ntfs-3g) — rescue the data"
[ -d "$MOUNT" ] || priv mkdir -p "$MOUNT" >/dev/null 2>&1 || true
if priv mount -t ntfs-3g -o ro "$DEV" "$MOUNT" 2>&1 | tail -3; then
  if mountpoint -q "$MOUNT"; then
    echo "mounted READ-ONLY — data can be copied off"
    ls "$MOUNT" | head -5
    exit 0
  fi
fi

say "attempt 3: mount after the repair"
if have udisksctl && udisksctl mount -b "$DEV" 2>&1 | tail -2; then
  if mountpoint -q "$MOUNT"; then echo "mounted"; ls "$MOUNT" | head -5; exit 0; fi
fi

say "attempt 4: forced ntfs3 mount (force)"
[ -d "$MOUNT" ] || priv mkdir -p "$MOUNT" >/dev/null 2>&1 || true
if priv mount -t ntfs3 -o force,rw "$DEV" "$MOUNT" 2>&1 | tail -3; then
  if mountpoint -q "$MOUNT"; then echo "mounted (force)"; ls "$MOUNT" | head -5; exit 0; fi
fi

echo
echo "FAILED. What to try manually:"
cat <<EOF
  sudo ntfsfix -d $DEV
  sudo mount -t ntfs3 -o force,rw $DEV $MOUNT
  # or open "Disks" (gnome-disks) and press "Restore Filesystem"
EOF
exit 1
