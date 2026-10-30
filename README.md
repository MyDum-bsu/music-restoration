# music-restoration

Снятие «войны громкости» с музыкальной библиотеки: задавленные мастера DR4–6
возвращаются к DR8–12 без потери тембра. Плюс полный цикл релиза — разбиение
образа, теги, обложки, тексты, копия в коллекцию и доставка на телефон.

Это **v2** проекта. Старая версия (v1.0, Stereo Tool CLI + фиксированная
цепочка) сохранена тегом `v1.0` в истории репозитория; её движок больше не
используется и в текущем дереве отсутствует.

## Чем v2 отличается от v1

| | v1.0 | v2 |
|---|---|---|
| Движок | ffmpeg + Stereo Tool CLI + `sts`-профили | собственное DSP-ядро `rnext/dsp/` |
| Платный софт | нужен для стадии ND2 | не нужен |
| Настройка | фиксированная цепочка | калибровка по парам «оригинал → мастер-реставрация» |
| Цель | константа | модель предсказывает Δcrest по треку, ограничена физикой |
| Проверка | метрики | метрики + гейты (TP, внесённый клип, OOD) + слепые сравнения |

## Что умеет

- **Реставрация альбома** — образ+cue или каталог треков; авто-политика по
  классу материала, деклип только при реальном клипе, TP ≤ −0.5 dBFS.
- **Оформление релиза** — теги из источника/cue, обложка, тексты, копия в
  коллекцию.
- **Телефон** — замена альбома по MTP с побайтовой проверкой md5, инвентарь
  библиотеки, массовые тексты и обложки.
- **Измерения** — TT DR, LUFS, true peak, клиппинг, фейковый hi-res, lossy-транскод.
- **Корпус и ML** — пары «оригинал → реставрация мастера», версии датасета,
  обучение оценщика цели, снапшоты и гейт продвижения.
- **Слепые сравнения** — пак вариантов, выровненных по LUFS, и Electron-плеер.

## Установка

Нужны Python 3.10+, `ffmpeg`, `flac` (с `metaflac`), `curl`. Для телефона —
`gio` (gvfs MTP).

```bash
git clone https://github.com/MyDum-bsu/music-restoration.git
cd music-restoration
./install.sh                 # проверка зависимостей + симлинки скиллов в ~/.agents/skills
./install.sh --venv          # то же + .venv и pip install -r requirements.txt
```

Без агентских скиллов всё работает обычным CLI:

```bash
python3 scripts/audio_measure.py "album/"                  # DR/LUFS/TP/клиппинг
python3 scripts/restore_one.py --src "album/" --out out/   # реставрация
python3 scripts/finish_release.py --restored out/ --source "album/" --dest ~/Music/restored --lyrics
```

## Реставрация альбома

```bash
# образ + cue
python3 scripts/restore_one.py \
    --src "Artist - Album (1993)/image.flac" --cue "Artist - Album (1993)/image.cue" \
    --out "out/Artist - Album (1993)" \
    --auto-policy --adaptive --ml-snapshot auto --tp -0.5

# каталог треков
python3 scripts/restore_one.py \
    --src "Artist - Album (1993)/" --out "out/Artist - Album (1993)" \
    --auto-policy --adaptive --ml-snapshot auto --tp -0.5
```

`--auto-policy` выбирает режим по классу материала, `--adaptive` подбирает
силу контура под целевой Δcrest, `--ml-snapshot auto` подставляет лучшую
прошедшую гейт модель. Без корпуса и моделей работает классовая таблица:
DSP-ядро самодостаточно.

Полный цикл (разбиение → реставрация → теги/обложка/тексты → телефон):

```bash
python3 scripts/release_batch.py \
    --job "release_src/Artist/1993 - Album|Artist/Album (1993)|Artist - Album (1993)" \
    --log /tmp/batch.log
```

## Пайплайн

`rnext/dsp/pipeline.py`:

1. **HPF 25 Гц** — убирает рокот/DC (zero-phase).
2. **Деклип** — только при реальном клипе: AR-экстраполяция, forward+backward blend.
3. **EQ** — спектральная дельта относительно reference (по умолчанию off).
4. **Контур** — главный этап: per-band, per-frame уровень над сглаженным телом
   растягивается в `r` раз; транзиенты держит WOLA и TP-клип.
5. **TP-лимит** — измерение с 4× oversample, мягкое приведение к −0.5 dBFS.
6. **Вывод** — 16-бит FLAC, TPDF-дизер.

Адаптивный режим делает пробный проход и линейно инвертирует измеренный
Δcrest к цели (по умолчанию +4.4 дБ — уровень референсных мастеров корпуса).

Подробности и калибровка — в [docs/dataset.md](docs/dataset.md) и [docs/ml.md](docs/ml.md).

## Скиллы

`./install.sh` ставит в `~/.agents/skills`:

| скилл | для чего |
|---|---|
| `release-restore` | полный цикл релиза: источник → реставрация → оформление → коллекция |
| `audio-measure` | измерить DR/громкость/подлинность файла или каталога |
| `tracker` | поиск, сравнение и закачка раздач |
| `restore-status` | состояние проекта: датасет, качки, процессы, модели |
| `add-lyrics` | тексты в теги (genius.com + lrclib) |
| `phone-music` | инвентарь, тексты и обложки на телефоне по MTP |
| `split-flac-album` | разбиение image+cue на треки |

Скиллы — тонкие обёртки над `scripts/`; каждый работает и без агента.

## Данные и конфигурация

- Корпус и релизы: `config/workspace.json` → `data_root` (по умолчанию флешка
  `/media/<user>/Data/Music/restore`). Переопределяется `RESTORE_DATA_ROOT`.
- Личный конфиг: `~/.config/music-restoration/config.env` (образец —
  `config.env.example`). Ключи необязательны.
- Секреты (cookies rutracker, Transmission RPC): `~/.config/music-restoration/secrets/`.
- Состояние и кэши: `~/.local/state/music-restoration/`.
- Модели: `models/snapshots/` (локальные артефакты, в git не коммитятся),
  реестр и лидерборд — `models/registry.jsonl`, `models/leaderboard.jsonl`.

## Корпус и ML

Корпус — пары «оригинал → реставрация мастера» двух независимых
реставраторов (Лисной Йожыг и Alias Divas). Датасет версионируется по
содержимому (`ds-<хэш>`), паспорт версии кладётся в `dataset_manifests/`.

```bash
python3 scripts/dataset_plan.py topics --author 5618698,3048819
python3 scripts/dataset_select.py --limit 12
python3 scripts/dataset_fetch.py --go --max-gb 8
python3 scripts/dataset_extract.py --all --excerpts 1
python3 scripts/ml_auto.py                 # версия → обучение → эвалы → сводка
python3 scripts/status_report.py --deep    # состояние проекта
```

Модель выбирается по объёму (`table` → `ridge`/`conformal`/`knn` → `hgb`/`hgbq`)
и продвигается в рантайм только если лучше лучшей базы по медиане |err| при
95 % CI > 0 и выигрыше ≥ 0.10 дБ.

## Слепые сравнения

```bash
python3 scripts/listen_pack.py ...   # пак вариантов, выровненных по LUFS
cd app && npm install && npm start   # Electron-плеер A/B/C
```

Локальный обзор датасета и пайплайна: `python3 lab/server.py` →
http://127.0.0.1:8765 (см. [lab/README.md](lab/README.md)).

## Тесты

```bash
python3 -m pytest tests/ -q
```

## История

- `v1.0` — старый движок на Stereo Tool CLI. Сохранён тегом и историей git.
- `v2.0` — текущее дерево: собственное DSP-ядро, корпус, ML, релизный цикл.

## Лицензия

MIT — см. [LICENSE](LICENSE).
