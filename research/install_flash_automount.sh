#!/usr/bin/env bash
# One-time flash drive setup: mounts WITHOUT a password and survives hard
# power-offs. Run once as root:
#
#     sudo ~/Desktop/projects/music-restoration/scripts/install_flash_automount.sh
#
# What it does:
#   1) adds an /etc/fstab line for the "Data" partition with the ntfs-3g driver
#      and the `user` option — a regular user can mount/unmount without a
#      password (ntfs-3g replays the NTFS journal itself after a crash, so
#      ntfsfix is no longer needed);
#   2) installs a polkit rule allowing udisks (GNOME "Disks", Files,
#      udisksctl) to mount without a password prompt;
#   3) puts the mount point in order (user-owned directory).
set -eu

LABEL="${1:-Data}"
MOUNT="/media/mydum/Data"
USER_NAME="${SUDO_USER:-mydum}"
UID_N="$(id -u "$USER_NAME")"
GID_N="$(id -g "$USER_NAME")"

[ "$(id -u)" = "0" ] || { echo "root required: sudo $0"; exit 1; }

DEV="$(lsblk -pnro NAME,LABEL | awk -v L="$LABEL" '$2==L {print $1; exit}')"
[ -n "$DEV" ] || { echo "no partition with label \"$LABEL\" found"; exit 1; }
UUID="$(lsblk -pnro NAME,UUID | awk -v D="$DEV" '$1==D {print $2}')"
[ -n "$UUID" ] || { echo "no UUID found for $DEV"; exit 1; }
echo "partition $DEV (label $LABEL), UUID $UUID, user $USER_NAME ($UID_N:$GID_N)"

echo "→ editing /etc/fstab (backup kept alongside)"
cp -a /etc/fstab "/etc/fstab.bak.$(date +%s)"
FSTAB_LINE="UUID=$UUID $MOUNT ntfs-3g rw,user,auto,nofail,uid=$UID_N,gid=$GID_N,umask=022,iocharset=utf8,x-gvfs-show 0 0"
grep -v "$UUID" /etc/fstab | grep -v "[[:space:]]$MOUNT[[:space:]]" > /tmp/fstab.new || true
printf '%s\n' "$FSTAB_LINE" >> /tmp/fstab.new
install -m 644 -o root -g root /tmp/fstab.new /etc/fstab
echo "   added: $FSTAB_LINE"

echo "→ setuid on ntfs-3g (otherwise the fstab user option does not allow mounting)"
if [ -x /usr/bin/ntfs-3g ]; then
  chmod 4755 /usr/bin/ntfs-3g
  echo "   /usr/bin/ntfs-3g → $(stat -c %A /usr/bin/ntfs-3g)"
fi

echo "→ polkit rule (udisks without a password)"
mkdir -p /etc/polkit-1/rules.d
cat > /etc/polkit-1/rules.d/49-udisks-mount.rules <<EOF
// Allow user $USER_NAME to mount/unmount disks via udisks (GNOME Files/Disks,
// udisksctl) without a password prompt.
polkit.addRule(function(action, subject) {
    if (action.id.indexOf("org.freedesktop.udisks2.") === 0 &&
        subject.user === "$USER_NAME" && subject.local && subject.active) {
        return polkit.Result.YES;
    }
});
EOF
systemctl restart polkit 2>/dev/null || true

echo "→ mount point"
if mountpoint -q "$MOUNT"; then
  echo "   $MOUNT is already mounted"
else
  # remove a possible "Data1" left over by udisks
  for extra in "$MOUNT"1 "$MOUNT"2; do
    mountpoint -q "$extra" && umount "$extra" && echo "   unmounted $extra"
  done
  rmdir "$MOUNT" 2>/dev/null || true
  mkdir -p "$MOUNT"
  chown "$UID_N:$GID_N" "$MOUNT"
  chmod 755 "$MOUNT"
fi

echo "→ check (as $USER_NAME, without a password)"
if mount "$MOUNT" 2>/dev/null || mountpoint -q "$MOUNT"; then
  echo "   mounted:"
  ls "$MOUNT" | head -5
  df -h "$MOUNT" | tail -1
else
  echo "   failed to mount — try: mount $MOUNT"
fi

cat <<'MSG'

Done. From now on:
  * mount/unmount without a password:  mount /media/mydum/Data | umount /media/mydum/Data
  * the partition mounts automatically at boot (nofail: if it can't, the system still boots)
  * after a hard power-off ntfs-3g replays the journal itself; ntfsfix is only needed
    if Windows left the volume hibernated (then: sudo mount -t ntfs-3g -o remove_hiberfile ...)
  * to resume work:  ~/Desktop/projects/music-restoration/research/resume.sh
MSG
