# restore-next

Универсальный инструмент реставрации аудиоальбомов (динамика, деклип, спектр),
калибруемый на парах «оригинал → профессиональная реставрация».

## Цель (зафиксирована)

> Инструмент на входе принимает оригинал (CD 16/44), на выходе даёт
> реставрацию уровня референсных мастеров: ΔDR ≈ +4 дБ на сжатых
> материалах, TP ≤ −0.5 dBFS, спектр в допуске относительно
> reference-реставрации. Артефакты ниже порогов гейтов. Автономно
> работает на новых альбомах без учителя.

## Архитектура DSP-ядра

Пайплайн `rnext/dsp/pipeline.py`:

1. **HPF 25 Гц** — sosfiltfilt, убирает рокот/DC.
2. **Деклип** `rnext/dsp/declip.py` — только при наличии клипа:
   AR-экстраполяция (Levinson по контексту, forward+backward blend),
   стабильная рекурсия, edge-guard на мелких рывках.
3. **EQ** `rnext/dsp/eq.py` — спектральная дельта относительна
   reference (пока off по умолчанию; фазонейтрально через STFT-бины).
4. **Контур** `rnext/dsp/contour.py` — главный этап, два режима:
   - `expand` (по умолчанию): per-band, per-frame уровень над
     сглаженным телом растягивается `r_expand` раз; транзиенты в
     верхний TP не заедают благодаря перекрёстному WOLA и TP-клипу
     в конце; crest растёт, body остаётся.
   - `flatten`: медленная модуляция гейна вокруг аchoра полосы
     (k<1), bass_scale ослабляет НЧ.
   - Реализация: собственный chunked WOLA (Hann, hop=frame/4,
     кроссоверы между полосами 1/3-октавы по log-freq), паддинг
     краёв, two-pass (levels → gains → synthesis). Identity при
     r=1 точна (rel-err ~1e-8).
5. **TP-лимит** `rnext/dsp/limit.py` — статическое измерение 4×-oversample
   TP и мягкое приведение к потолку политики (по умолчанию −0.5 dBFS).
6. Вывод 16-бит FLAC (+ histogram позже — TPDF-дизер).

**Адаптивный r** (`rnext/dsp/adaptive.py`): пробный проход на полном
треке при r₀=1.5 → измеренный Δcrest → линейная инверсия к целевому
Δcrest (по умолчанию +4.4 дБ — уровень, наблюдаемый у референсных
мастеров корпуса).

## Калибровка на корпусе (май 2026, Michael Romeo пары)

`scripts/calibrate_pair.py`: наш пайплайн против master-restoration,
full-track метрики TT DR/crest + band deltas:

| альбом | r | crest_err (наши−мастер) | dr_err | bandΔ |
|---|---|---|---|---|
| WotW Pt.1 (2018) | 1.5 | −0.54 | −1.65 | ±0…2 дБ |
| WotW Pt.1 | 1.6 | +0.30 | −1.07 | +0…4 дБ |
| WotW Pt.2 CD1 (2022) | 1.5 | −0.37 | −1.15 | ±0…3 дБ |
| WotW Pt.2 | 1.6 | +0.47 | −0.48 | +0…4 дБ |

Адаптив (target Δcrest +4.4) на полном альбоме Pt.1 дал crest-ошибки
±0.43 дБ (среднее −0.06) на всех 9 треках.

## Датасет: два независимых автора

Полное описание — `docs/dataset.md`. Кратко:

В индексе 678 тем двух реставраторов: **Лисной Йожыг** (`pid=5618698`, 178 тем,
классический рок/хард, чаще `image+.cue`, публикует текстовую таблицу DR-метра)
и **Alias Divas** (`pid=3048819`, 500 тем, тяжёлые жанры, `tracks+.cue`).
Каталоги почти не пересекаются (3 общих артиста из 176 и 499) — это два
независимых вкуса, а не перезалив одного корпуса. Поэтому оценка обязана
держать 30 % альбомов КАЖДОГО автора вне настройки.

Артефакты: `corpus/authors/{pid}.jsonl` (каталоги), `corpus/restored/{tid}.json`
(разбор тем), `corpus/dataset_plan.jsonl` (релиз → оригинал), `corpus/master_targets.jsonl`
(цели мастера из его таблиц DR: релизов/треков), `dataset/` (пары, фичи, вырезки,
журнал загрузок, снапшоты моделей).

```bash
python3 scripts/dataset_plan.py topics --author 5618698,3048819   # разбор тем
python3 scripts/dataset_plan.py match                             # релиз → оригинал
python3 scripts/dataset_select.py --limit 12                      # выбор партии
python3 scripts/dataset_fetch.py --go --max-gb 8                  # загрузка (только нужное)
python3 scripts/dataset_extract.py --all --excerpts 1             # фичи + вырезки
python3 scripts/reference_audit.py                                # метки референсов
python3 scripts/dataset_audit.py                                  # не потерялось ли сырьё
python3 scripts/dataset_loop.py --interval 900                    # непрерывное обогащение
```

Скачивание только нужного: раздача добавляется на паузе, файлы сопоставляются с
трек-листом мастера, лишние снимаются (`files-unwanted`); из CD-дискографии
берётся только нужный диск. Одна тема-дискография обслуживает несколько пар —
wanted-набор считается объединением (`dataset_fetch.py --repair`).

## ML-слой: цель по треку вместо константы

Полное описание — `docs/ml.md`. Пайплайн (DSP) фиксирован, обучается **оценщик
цели** `Δcrest`; модель подставляется в физику и всегда проходит инварианты
(TP-потолок, отсутствие внесённого клипа, скобки).

Версионирование: `dataset/manifest.json` даёт версию `ds-<хэш>` по содержимому
(ключи пар + байты фич + метки), паспорт версии кладётся в `dataset_manifests/`;
снапшоты моделей — в `models/snapshots/<id>/` с метаданными (версия датасета,
признаки, сплит, метрики, git-коммит), реестр — `models/registry.jsonl`,
лидерборд — `models/leaderboard.jsonl`.

```bash
python3 scripts/ml_auto.py                  # версия → обучение → эвалы → сводка
python3 scripts/ml_auto.py --interval 1800  # то же в цикле
python3 scripts/ml_train.py --model ridge --by-author
python3 scripts/restore_one.py --src <img.flac> --cue <cue> --out out/ --ml-snapshot auto
```

Модели выбираются по объёму (`table` → `ridge`/`conformal`/`knn` → `hgb`/`hgbq`),
продвижение в рантайм — только если модель лучше лучшей базы по медиане |err| с
95 % CI > 0 и выигрышем ≥ 0.10 дБ (парный бутстрэп), иначе остаётся прежняя логика.

## Плеер слепых сравнений

`scripts/listen_pack.py` собирает пак (варианты выравниваются по LUFS, слепая
раскладка A/B/C), `app/` — Electron-приложение для прослушивания и оценок
(`cd app && npm install && npm start`). Полное описание — `app/README.md`.

## Команды

```bash
# реставрация целого альбома (image+cue или каталог треков)
python3 scripts/restore_one.py --src <img.flac> --cue <cue> \
        --out out/<name> --mode expand --adaptive --target 4.4 --tp -0.5

# сравнить наш прогон с master-restoration по корпусной паре
python3 scripts/compare_run.py --report out/<name>/restore_report.jsonl --key mrpt1-2018

# калибровка r по корпусной паре
python3 scripts/calibrate_pair.py --key mrpt1-2018 --tracks 1,2,3 \
        --r-list 1.4,1.5,1.6 --restored <dir> --orig-image <img> --cue <cue>

# фичи пары (TT DR/crest/band/clip) → corpus/feature_pairs/{key}.jsonl
python3 scripts/pair_features.py --key <key> --restored <dir> \
        --orig-image <img> --cue <cue>

# поиск оригинального издания под реставрацию мастера
python3 - <<'EOF'
from rnext.rtclient import RTClient
from rnext.pairsearch import find_originals
print(find_originals("Artist", "Album", 1993, client=RTClient()))
EOF

# статусы дашборда
python3 scripts/status.py
```

## Инфраструктура

- `rnext/rtclient.py` — rutracker cookie-jar клиент (dl.php и поиск);
- `rnext/transm.py` — Transmission RPC (batch загрузки;
  torrent-ы мастера не трогаем, наши id ≥ 4);
- `rnext/workspace.py` — бюджет 40 GiB на flash-разделе;
- корпус: `/media/.../restore/corpus/` (JSONL метрики, `master_topics.jsonl`);
- тесты: `tests/test_dsp.py` (pytest, 7).

## Планы

- деклип длинных ранов (cubic fallback >120 семплов) — для ререков
  клипогенных изданий;
- EQ-подгонка спектральной дельты (от corpus бланс) для calibrated mode;
- ABX-выписки 12/8-сек и артефакт-гейты (regression от гейтов корпуса);
- batch-002 (Aerosmith Get a Grip 1993): первый тест на чужой
  дискографии, эра клипов.
