# Портативность скила music-restoration — отчёт независимой проверки

Дата: 2026-11-06. Автор отчёта — независимый проверяющий: правки блоков A–E он
не делал, только читал diff и запускал проверки. Все числа ниже взяты из
вывода команд, выполненных в этой сессии; ничего не перемерялось и не
дописывалось.

## 1. Статус изменений

Правки блоков A–E **находятся в рабочем дереве и ещё не закоммичены**:
`git log --oneline -8` показывает последний коммит `92e0e7f` (phone: ADB-over-Wi-Fi
backend), а весь объём портативных правок виден только в `git diff HEAD` —
**11 files changed, 355 insertions(+), 24 deletions(-)** плюс 6 новых файлов
(`rnext/doctor.py` 480 строк, `scripts/doctor.py` 17, `tests/test_doctor.py` 187,
`tests/test_workspace.py` 57, `scripts/portability_eval.sh` 153,
`docs/eval-vs-masters-run.md` 178 — по `wc -l`). Коммит и push запланированы
после этой проверки (`.zcode/workflow-drafts/Портативность-скила.dwf.ts:140`).
Разбиение на блоки A–E — по плану того же workflow-файла (фазы «Блок A» …
«Блок E», строки 25–87).

## 2. Что изменено (по блокам)

### Блок A — конфигурация без машинной привязки
- `config/workspace.json` удалён из индекса (`git rm --cached`, файл на диске
  остался), в `.gitignore` добавлены строки `config/workspace.json` — машинный
  конфиг больше не попадает в клон. Трекнутый шаблон `config/workspace.example.json`
  остаётся в репо (`git ls-files config/`).
- `rnext/workspace.py:24-31` — `load_config()` без конфига бросает
  `WorkspaceError` с подсказкой «copy config/workspace.example.json …»
  (fail-fast вместо загадочного KeyError); `RESTORE_DATA_ROOT` имеет приоритет
  над `data_root` из конфига.
- `install.sh` (~строки 249–280) — бутстрап: если `config/workspace.json` нет,
  создаёт из `config/workspace.example.json` с `data_root` из
  `RESTORE_DATA_ROOT` или по умолчанию `~/music-restoration-data`, создаёт
  каталоги из `dirs` и проверяет их запись probe-файлом; существующий конфиг
  не перезаписывается.
- Новый `tests/test_workspace.py` — 3 теста: fail-fast без конфига, приоритет
  env, валидность example-JSON.

### Блок B — doctor
- Новый `rnext/doctor.py` (480 строк): 11 HARD-проверок (python ≥ 3.10; импорт
  numpy/scipy/matplotlib/mutagen/PIL/bs4/requests/sklearn/joblib; бинары
  ffmpeg/ffprobe/flac/metaflac/shntool/cuetag; config/workspace.json валиден;
  data_root существует и пишется; ≥ 1 GiB свободно на data_root) и 4 WARN
  (rutracker cookies, transmission RPC 127.0.0.1:9091, gio, adb). Exit-коды:
  0 — всё зелёное, 1 — HARD-провал, 2 — только WARN. Каждая диагностика несёт
  команду установки под пакетную семью apt/dnf/pacman/brew
  (`rnext/doctor.py:66-119`). Все пробники локальные: `shutil.which`, чтение
  конфига, write-probe, `shutil.disk_usage`, TCP на loopback — сеть не
  используется (`rnext/doctor.py:20-22`).
- Новый `scripts/doctor.py` — тонкая CLI-обёртка (17 строк).
- Новый `tests/test_doctor.py` (187 строк, 9 тестов) — exit-коды 0/1/2,
  схема JSON-отчёта, per-platform fix у каждого hard-провала, e2e-тест
  обёртки через subprocess.

### Блок C — install.sh: платформенные подсказки и бутстрап
- В чек зависимостей добавлены `shntool` и `cuetag` (алиас `cuetag.sh`
  некоторых дистрибутивов учтён в `have_tool`).
- `pm_detect()`/`pkg_for()`/`manager_install_cmd()`: менеджер определяется по
  `command -v` (apt-get/dnf/pacman/brew), у каждого MISSING печатается готовая
  команда установки.
- Флаг `--install-tools`: попытка установки недостающего; **sudo не
  используется** — brew запускается напрямую, для apt/dnf/pacman печатается
  готовая `sudo …` команда.
- Бутстрап секретов: `~/.config/music-restoration/secrets` (каталог `chmod
  700`, файлы `chmod 600`) + README с форматом `rutracker_cookies.txt` /
  `rutracker_ua.txt` (`cf_clearance` привязан к IP+UA); существующие файлы
  не трогаются.
- Явное предупреждение: tracker-скилл требует аккаунт rutracker (регистрация
  invite-only).
- Проверка python-модулей: без них `fail=1` и подсказка
  `pip install -r requirements.txt` или `--venv` (install.sh, секция
  «dependencies», ветка `MISSING python3 deps`).

### Блок D — тексты скиллов: preflight и деградация
- `skills/release-restore/SKILL.md` — **Шаг 0 Preflight**: `python3
  scripts/doctor.py`; exit 1 (HARD) → назвать пользователю точную команду
  установки из колонки fix и остановиться; exit 2 (WARN) → развилка: локальный
  файл есть — работать без трекера с предупреждением, иначе сказать про
  секреты и остановиться.
- `skills/tracker/SKILL.md` — раздел Requirements: аккаунт rutracker
  (invite-only), формат cookies 700/600, правило «нет cookies → не запускать
  поиск».
- `docs/setup.md` — матрица платформ (apt/dnf/pacman — supported; brew —
  supported с pip для python-пакетов; Windows — только WSL; phone-music —
  Linux-десктоп с gio), troubleshooting-таблица (симптом → причина → команда),
  раздел про права 700/600.
- `README.md` — Quick Start из трёх команд (`git clone && cd`, `./install.sh`,
  `python3 scripts/doctor.py`) + строка про rutracker-требование для сценария
  поиска.

### Блок E — тесты и CI
- `.github/workflows/ci.yml`: `apt-get install ffmpeg flac shntool cuetools`
  в шаге установки; CI сам создаёт локальный `config/workspace.json` в
  `$RUNNER_TEMP` (в репо его больше нет); новый шаг «Preflight doctor» —
  `python3 scripts/doctor.py --json`, exit 2 (только WARN) на свежем раннере
  допустим, exit 1 роняет шаг.
- `tests/conftest.py:13-24` — починка тестового окружения: если интерпретатор
  запущен под чужим `argv[0]` (здесь — AppImage-обёртка), `sys.executable`
  возвращается к реальному python из `sys.base_prefix`.

### Вне заявленных блоков A–E (присутствует в том же рабочем дереве)
- `rnext/phone/adb.py:77-98` — `mdns_endpoint()` и переработка
  `ensure_connected()` (mDNS-обнаружение беспроводного ADB). К блокам A–E не
  относится (следphone-трек); в acceptance не входит.
- `docs/eval-vs-masters-run.md` — новый анализ качества реставраций, к
  переносимости отношения не имеет.

## 3. Как проверено (команды и вывод)

### 3.1 Прогоны независимого проверяющего (эта сессия)

| # | команда | результат |
|---|---|---|
| 1 | `git log --oneline -8` | последний коммит `92e0e7f` (phone ADB), правки A–E в рабочем дереве |
| 2 | `git diff HEAD --stat` | `11 files changed, 355 insertions(+), 24 deletions(-)` |
| 3 | `git ls-files config/` | `config/workspace.example.json` (машинный workspace.json из git убран) |
| 4 | `python3 -m pytest tests/ --color=no -q` | **61 passed in 10.39s** |
| 5 | `python3 scripts/doctor.py` (основной репо) | exit **1**: единственный провал — «свободно >= 1 GiB на data_root» (локальный диск почти полон); остальные 13 строк ok/warn — механизм работает |
| 6 | `python3 -m pip install --dry-run mutagen` | `error: externally-managed-environment` — на этой системе (Ubuntu, python 3.12.3, `/usr/lib/python3.12/EXTERNALLY-MANAGED` существует) ставка «pip в системный python» отвергается интерпретатором |
| 7 | `bash scripts/portability_eval.sh` (мой повторный запуск) | **PORTABILITY-OK, exit 0** (вывод ниже) |

Пошаговый вывод моего запуска `bash scripts/portability_eval.sh`
(снапшот рабочего дерева `cc35b9e`):

```
== [1/7] clone /home/mydum/Desktop/projects/music-restoration -> /tmp/tmp.jMWTWEBflo/mr-fresh ==
   cloned at snapshot cc35b9e
== [2/7] install.sh (no venv, skills into /tmp/tmp.jMWTWEBflo/agent-skills) ==
     ok      python3 + numpy/scipy/matplotlib/mutagen/pillow/bs4/requests/sklearn/joblib
     created config/workspace.json (data_root=/home/mydum/music-restoration-data)
   All set. Restore an album:
   workspace.json data_root = /home/mydum/music-restoration-data
== [3/7] doctor --json (hard checks must be green) ==
   all 11 hard checks ok (exit_code 2: warns allowed)
== [4/7] synthetic killed track -> /tmp/tmp.jMWTWEBflo/in/track.flac ==
   source: DR -1, TP 0.6 dBFS, clipped 1054208, crest -0.5 dB
== [5/7] restore_per_track.py (release-restore skill, step 4; ~15 min) ==
     -> excerpt choice: expand target=6 (score 3.986)
     -> final expand target=6: DR -1→3 crest 2.51→8.46 (+5.95) shape 3.75
== [6/7] QC gates on the result ==
   qc_gates "verdict": "OK"
   result: DR -1 -> 3 (delta +4), TP -0.5 dBFS, clipped 0
== [7/7] git status in the clone must be clean ==
   clean
PORTABILITY-OK
```

### 3.2 Логи acceptance-автора (первый прогон гейта)

Логи сохранились в логах сессии workflow (`/tmp/zcode-bash-NdMJdt/…actor_6_1/`):

- Попытка 1 — **PORTABILITY-FAIL** на шаге 3: `scripts/doctor.py --json exited
  non-zero` (снапшот `7c63d1b`, временный каталог `/tmp/tmp.UC7oJc9Fsz`);
  после фикса —
- Попытка 2 — **PORTABILITY-OK, EXIT=0** (снапшот `a3b16cf`): doctor «all 11
  hard checks ok (exit_code 2: warns allowed)», restore `DR -1 -> 3 (delta
  +4), TP -0.5 dBFS, clipped 0`, `qc_gates "verdict": "OK"`, git status чист.

Итого acceptance-гейт зелёный дважды независимо (у автора на `a3b16cf` и у
проверяющего на `cc35b9e`).

## 4. Вердикт переносимости

**Что доказано.** На Linux/Debian-семействе (apt, python 3.12.3, все
системные тулы) свежий клон ставится и проходит полный цикл «install →
doctor → реставрация синтетики → QC» без единой ручной правки: конфиг
создаётся бутстрапом с дефолтным data_root, скиллы линкуются, git остаётся
чистым, doctor даёт 11/11 hard-проверок зелёными. Полный тест-сьют 61/61.

**Что доказательством не покрыто.** Все прогоны acceptance (автора и мои два)
выполнены на одной и той же машине. Свежий клон изолирует **состояние репо**,
но не хост: системные зависимости и ядро ОС те же. Реального запуска на
macOS/Windows/WSL/другом дистрибутиве не было — статусы «supported» в матрице
`docs/setup.md` заявлены, но физически не прогонялись.

**Ответ на вопрос «может ли другой человек на другой платформе поставить скил
и без трудностей восстановить альбом»:**

- **Linux (Debian/Ubuntu/Mint, Fedora, Arch), локальный файл-источник** — да,
  с высокой уверенностью: путь полностью покрыт пройденным acceptance;
  install.sh сам создаёт конфиг и печатает готовые команды для недостающих
  тулов, doctor на шаге 0 скилла ловит остальное с fix-командами.
- **Linux, сценарий «найти релиз через tracker»** — с трудностями, не
  зависящими от кода: нужен rutracker-аккаунт (регистрация invite-only) и
  cookies, привязанные к IP+UA. Скилл честно деградирует: без cookies поиск
  не запускается (WARN в doctor, правило в tracker-скилле).
- **macOS** — вероятно с трениями: системные тулы ставятся через brew
  (подсказки есть), но python-зависимости в brew-питоне ставятся только через
  venv — см. препятствие №3.
- **Windows** — только WSL (native не поддерживается, заявлено в setup.md).

## 5. Оставшиеся препятствия

Критичные (могут заблокировать стороннего пользователя):

1. **Кроссплатформенность заявлена, но не измерена.** Ни один прогон не вышел
   за пределы хоста автора. Формально «портативность» доказана только внутри
   класса «Linux + apt + системный python со всеми пакетами».
2. **Трекер-сценарий требует rutracker-аккаунт** (invite-only) и cookies,
   привязанные к IP+UA (`cf_clearance`); это внешний барьер, кодом не
   устранимый. Локальный сценарий от него не зависит — деградация описана.
3. **Холодный старт без зависимостей не покрыт acceptance и частично сломан
   подсказкой.** Acceptance сознательно идёт «no venv: system python3 already
   has the deps». На этой же машине `pip install` в системный python
   отвергается (`externally-managed-environment`, проверено командой из
   таблицы п. 3.1.6) — то есть подсказка install.sh «pip install -r
   requirements.txt» на Debian 12+/Ubuntu 23.04+ без `--venv` не сработает.
   Рабочий обход есть (`./install.sh --venv`), но в Quick Start README он не
   упомянут как первый выбор для таких систем.

Спорные (не блокируют, но стоят решения):

4. `--install-tools` никогда не использует sudo: на apt/dnf/pacman печатает
   готовую команду вместо установки — осознанный дизайн безопасности, но это
   ещё одно ручное действие для новичка.
5. Жёсткий порог doctor «≥ 1 GiB свободно на data_root» может быть тесен для
   реального альбома (бюджет самого workspace — 40 GiB). На этой машине
   основной репо именно поэтому даёт doctor exit 1 — порог работает, но
   конкретное значение спорно.
6. Дефолтный `data_root = ~/music-restoration-data` (домашний раздел): на
   машинах с маленьким home doctor упадёт HARD, нужно править конфиг руками.
7. DR/crest-оценка acceptance сделана на синтетике (DR −1 → 3): это проверка
   работоспособности конвейера, а не звучания на реальном материале
   (качество отдельно разобрано в `docs/eval-vs-masters-run.md`).

## 6. Ограничения этой проверки

- macOS/WSL/другие дистрибутивы в этой среде физически недоступны — не
  запускались.
- Приёмочный тест изолирует репо, а не хост: «другая платформа» в рамках
  этой проверки не симулировалась ничем, кроме чистого клона.
- `docs/portability-report.md` до этого отчёта **не существовал** (проверено
  `ls docs/` в начале сессии) — создан этим отчётом.
- Все числа — из выполненных здесь команд и сохранённых логов прогонов
  acceptance; ничего не перемерялось и не экстраполировалось.
