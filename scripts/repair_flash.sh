#!/usr/bin/env bash
# Восстановление доступа к флешке после аварийного выключения.
#
# Что делает: находит NTFS-раздел по метке (по умолчанию «Data»), проверяет,
# не смонтирован ли он, снимает флаг «грязный» (ntfsfix -d) и монтирует.
# Если раздел не в fstab и udisks отказывается — падает на mount с опцией force.
#
# Права: нужен один привилегированный шаг. Скрипт сам выберет pkexec
# (графическое окно с паролем) или sudo (если запущен из терминала).
#
#   scripts/repair_flash.sh              # найти «Data» и восстановить
#   scripts/repair_flash.sh --label MD   # другой раздел
#   scripts/repair_flash.sh --check      # только диагностика
set -u

LABEL="Data"
MOUNT="/media/mydum/Data"
MODE="repair"
while [ $# -gt 0 ]; do
  case "$1" in
    --label) LABEL="$2"; shift 2;;
    --mount) MOUNT="$2"; shift 2;;
    --check) MODE="check"; shift;;
    *) echo "неизвестный аргумент: $1"; exit 2;;
  esac
done

say() { printf '\n\033[1m%s\033[0m\n' "$*"; }
have() { command -v "$1" >/dev/null 2>&1; }

say "разделы этого диска"
lsblk -o NAME,SIZE,FSTYPE,LABEL,MOUNTPOINT 2>/dev/null | sed 's/^/  /'

DEV="$(lsblk -pnro NAME,LABEL | awk -v L="$LABEL" '$2==L {print $1; exit}')"
case "$DEV" in /dev/*) ;; *) DEV="/dev/$DEV";; esac
if [ -z "$DEV" ]; then
  echo "не нашёл раздел с меткой «$LABEL»"; exit 1
fi
echo "раздел: $DEV (метка $LABEL), целевая точка: $MOUNT"

if mountpoint -q "$MOUNT" 2>/dev/null; then
  echo "уже смонтирован в $MOUNT — ничего делать не нужно"
  ls "$MOUNT" | head -5
  exit 0
fi
[ "$MODE" = "check" ] && exit 0

say "попытка 1: обычное монтирование через udisks"
if have udisksctl && udisksctl mount -b "$DEV" 2>&1 | tail -2; then
  if mountpoint -q "$MOUNT"; then echo "смонтировано"; ls "$MOUNT" | head -5; exit 0; fi
fi

# Привилегированный шаг: ntfsfix -d снимает флаг «грязный» (том не был
# размонтирован из-за аварийного выключения) — после этого ядро ntfs3 монтирует.
priv() {
  if have pkexec && pgrep -f "[p]olkit.*agent|gnome-shell" >/dev/null 2>&1; then
    echo "→ запрос прав через pkexec (появится окно с паролем)"
    pkexec "$@"
  elif have sudo; then
    echo "→ запрос прав через sudo (введите пароль)"
    sudo "$@"
  else
    echo "нет ни pkexec, ни sudo — выполните вручную: $*"; return 1
  fi
}

say "попытка 2: ntfsfix -d (снять флаг «грязный»)"
if have ntfsfix; then
  priv ntfsfix -d "$DEV" 2>&1 | tee /tmp/ntfsfix.out | tail -6
  if grep -qi "corrupt\|chkdsk" /tmp/ntfsfix.out; then
    say "попытка 2b: ntfsfix -b (загрузочный сектор) — том помечен повреждённым"
    priv ntfsfix -b -d "$DEV" 2>&1 | tee -a /tmp/ntfsfix.out | tail -6
  fi
else
  echo "ntfsfix не установлен (пакет ntfs-3g)"
fi

say "попытка 2c: монтирование только для чтения (ntfs-3g) — спасаем данные"
[ -d "$MOUNT" ] || priv mkdir -p "$MOUNT" >/dev/null 2>&1 || true
if priv mount -t ntfs-3g -o ro "$DEV" "$MOUNT" 2>&1 | tail -3; then
  if mountpoint -q "$MOUNT"; then
    echo "смонтировано ТОЛЬКО ДЛЯ ЧТЕНИЯ — можно скопировать данные"
    ls "$MOUNT" | head -5
    exit 0
  fi
fi

say "попытка 3: монтирование после ремонта"
if have udisksctl && udisksctl mount -b "$DEV" 2>&1 | tail -2; then
  if mountpoint -q "$MOUNT"; then echo "смонтировано"; ls "$MOUNT" | head -5; exit 0; fi
fi

say "попытка 4: принудительное монтирование ntfs3 (force)"
[ -d "$MOUNT" ] || priv mkdir -p "$MOUNT" >/dev/null 2>&1 || true
if priv mount -t ntfs3 -o force,rw "$DEV" "$MOUNT" 2>&1 | tail -3; then
  if mountpoint -q "$MOUNT"; then echo "смонтировано (force)"; ls "$MOUNT" | head -5; exit 0; fi
fi

echo
echo "НЕ УДАЛОСЬ. Что попробовать вручную:"
cat <<EOF
  sudo ntfsfix -d $DEV
  sudo mount -t ntfs3 -o force,rw $DEV $MOUNT
  # или откройте «Диски» (gnome-disks) и нажмите «Восстановить файловую систему»
EOF
exit 1
