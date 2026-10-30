#!/usr/bin/env bash
# Возобновление работы после перезагрузки: монтирование флешки, тюнинг
# Transmission, цикл обогащения, обучение и плеер. Идемпотентно — можно
# запускать сколько угодно раз.
#
#   scripts/resume.sh             # поднять всё, что можно (БЕЗ плеера)
#   scripts/resume.sh --with-app  # плюс плеер для прослушивания
set -u
cd "$(dirname "$0")/.."
ROOT="$PWD"
DATA="/media/mydum/Data"
DATA_ROOT="$DATA/Music/restore"

say() { printf '\n\033[1m%s\033[0m\n' "$*"; }

say "1/5 флешка"
if mountpoint -q "$DATA"; then
  echo "уже смонтирована: $DATA"
else
  echo "не смонтирована — пробую udisksctl"
  if ! udisksctl mount -b /dev/sda5 2>&1 | tail -2; then
    cat <<'MSG'
НЕ УДАЛОСЬ СМОНТИРОВАТЬ. Скорее всего NTFS помечен «грязным» после падения.
Выполните один раз (нужен пароль):

    sudo ntfsfix -d /dev/sda5
    udisksctl mount -b /dev/sda5

Если и это не помогло:

    sudo mount -t ntfs3 -o force,rw /dev/sda5 /media/mydum/Data

после чего запустите этот скрипт снова.
MSG
    exit 1
  fi
fi
[ -d "$DATA_ROOT" ] || { echo "нет $DATA_ROOT — раздел смонтирован не туда?"; exit 1; }

say "2/5 Transmission"
pgrep -f "[t]ransmission-gtk" >/dev/null || {
  echo "запускаю transmission-gtk"; setsid nohup transmission-gtk >/tmp/transmission.log 2>&1 </dev/null &
  sleep 12
}
python3 scripts/transmission_tune.py --apply | tail -3

say "3/5 целостность данных"
python3 scripts/dataset_audit.py --integrity | tail -3

say "4/5 цикл обогащения"
if pgrep -f "[d]ataset_loop.py" >/dev/null; then
  echo "цикл уже работает"
else
  # параметры те же, что у watchdog.sh: агрессивные (24 пары / 16 ГБ) забивали
  # раздел — держим консервативные, с капом по живым пирам и минимумом места
  setsid nohup python3 -u scripts/dataset_loop.py --interval 420 --batch-pairs 10 \
    --batch-gb 6 --keep-window 6 --stall-passes 1 --min-rate-mbps 2.0 \
    --min-free-gb 15 --max-inflight-gb 12 --max-master-gb 6 \
    >/tmp/dataset_loop.out 2>&1 </dev/null &
  sleep 5; echo "цикл запущен (лог /tmp/dataset_loop.out)"
fi

say "5/5 обучение"
python3 scripts/ml_auto.py --force --with-audio 2>&1 | tail -20

# Плеер запускается только по явному запросу: он нужен на последней стадии,
# при обычном возобновлении работы мешает.
if [ "${1:-}" = "--with-app" ] && [ -d app/node_modules ]; then
  pgrep -f "music-restoration/app/node_modules/[e]lectron/dist" >/dev/null || {
    echo "запускаю плеер"
    (cd app && setsid nohup ./node_modules/.bin/electron --no-sandbox --disable-gpu . \
       >/tmp/electron.log 2>&1 </dev/null &)
  }
fi

say "готово"
python3 scripts/status.py | tail -3
