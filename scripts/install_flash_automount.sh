#!/usr/bin/env bash
# Одноразовая настройка флешки: монтируется БЕЗ пароля и переживает аварийные
# выключения. Запускать один раз с правами root:
#
#     sudo ~/Desktop/projects/restore-next/scripts/install_flash_automount.sh
#
# Что делает:
#   1) добавляет в /etc/fstab строку для раздела «Data» с драйвером ntfs-3g
#      и опцией `user` — монтировать/размонтировать может обычный пользователь
#      без пароля (ntfs-3g сам проигрывает журнал NTFS после сбоя, поэтому
#      ntfsfix больше не нужен);
#   2) ставит правило polkit, разрешающее udisks (GNOME «Диски», Файлы,
#      udisksctl) монтировать без запроса пароля;
#   3) приводит точку монтирования в порядок (каталог пользователя).
set -eu

LABEL="${1:-Data}"
MOUNT="/media/mydum/Data"
USER_NAME="${SUDO_USER:-mydum}"
UID_N="$(id -u "$USER_NAME")"
GID_N="$(id -g "$USER_NAME")"

[ "$(id -u)" = "0" ] || { echo "нужны права root: sudo $0"; exit 1; }

DEV="$(lsblk -pnro NAME,LABEL | awk -v L="$LABEL" '$2==L {print $1; exit}')"
[ -n "$DEV" ] || { echo "не нашёл раздел с меткой «$LABEL»"; exit 1; }
UUID="$(lsblk -pnro NAME,UUID | awk -v D="$DEV" '$1==D {print $2}')"
[ -n "$UUID" ] || { echo "не нашёл UUID для $DEV"; exit 1; }
echo "раздел $DEV (метка $LABEL), UUID $UUID, пользователь $USER_NAME ($UID_N:$GID_N)"

echo "→ правлю /etc/fstab (бэкап рядом)"
cp -a /etc/fstab "/etc/fstab.bak.$(date +%s)"
FSTAB_LINE="UUID=$UUID $MOUNT ntfs-3g rw,user,auto,nofail,uid=$UID_N,gid=$GID_N,umask=022,iocharset=utf8,x-gvfs-show 0 0"
grep -v "$UUID" /etc/fstab | grep -v "[[:space:]]$MOUNT[[:space:]]" > /tmp/fstab.new || true
printf '%s\n' "$FSTAB_LINE" >> /tmp/fstab.new
install -m 644 -o root -g root /tmp/fstab.new /etc/fstab
echo "   добавлено: $FSTAB_LINE"

echo "→ правило polkit (udisks без пароля)"
mkdir -p /etc/polkit-1/rules.d
cat > /etc/polkit-1/rules.d/49-udisks-mount.rules <<EOF
// Разрешить пользователю $USER_NAME монтировать/размонтировать диски через
// udisks (GNOME Файлы/Диски, udisksctl) без запроса пароля.
polkit.addRule(function(action, subject) {
    if (action.id.indexOf("org.freedesktop.udisks2.") === 0 &&
        subject.user === "$USER_NAME" && subject.local && subject.active) {
        return polkit.Result.YES;
    }
});
EOF
systemctl restart polkit 2>/dev/null || true

echo "→ точка монтирования"
if mountpoint -q "$MOUNT"; then
  echo "   $MOUNT уже смонтирован"
else
  # снять возможный «Data1», оставшийся от udisks
  for extra in "$MOUNT"1 "$MOUNT"2; do
    mountpoint -q "$extra" && umount "$extra" && echo "   отмонтирован $extra"
  done
  rmdir "$MOUNT" 2>/dev/null || true
  mkdir -p "$MOUNT"
  chown "$UID_N:$GID_N" "$MOUNT"
  chmod 755 "$MOUNT"
fi

echo "→ проверка (от имени $USER_NAME, без пароля)"
if mount "$MOUNT" 2>/dev/null || mountpoint -q "$MOUNT"; then
  echo "   смонтировано:"
  ls "$MOUNT" | head -5
  df -h "$MOUNT" | tail -1
else
  echo "   не смонтировалось — попробуйте: mount $MOUNT"
fi

cat <<'MSG'

Готово. Дальше:
  * монтирование/размонтирование без пароля:  mount /media/mydum/Data | umount /media/mydum/Data
  * раздел монтируется автоматически при загрузке (nofail: если не смог — система всё равно стартует)
  * после аварийного выключения ntfs-3g сам проигрывает журнал; ntfsfix нужен только
    если Windows оставил том в режиме гибернации (тогда: sudo mount -t ntfs-3g -o remove_hiberfile ...)
  * возобновление работы:  ~/Desktop/projects/restore-next/scripts/resume.sh
MSG
