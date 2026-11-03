# Код-ревью music-restoration (2026-11-02)

Ревью всего проекта от «пустого дерева» до HEAD 181aae4 (+dirty), две оси: Standards / Spec (по скиллу matt-code-review), архитектурная линза — matt-codebase-design. Ревьюер: агентская сессия ZCode; золотой путь продукта проверен живьём в тот же день (12 релизов прошли полный цикл restore).

## Резюме проекта

Агент-native продукт реставрации музыки: глубокое DSP/ML-ядро `rnext/` → тонкие CLI-скрипты `scripts/` → скиллы для агента; `research/` — выделенный исследовательский харнесс; `app/` — офлайн Electron-плеер слепых сравнений. Золотой путь («restore this» → трекер → реставрация → гейты → выдача) работает.

## Standards

1. **[HIGH] Два «мозга» выбора лучшего релиза.** `rnext/planner.py` `score_candidate` и `scripts/tracker_hunt.py:225` `score()` — две независимые реализации скоринга релизов с разными критериями; скилл-флоу и датасет-флоу могут ранжировать одинаковые темы по-разному. Фикс: один скорер в `rnext/planner`, второй — тонкий вызов. (Smell: Duplicated Code)
2. **[HIGH] Машинные пути зашиты в скиллы.** `skills/release-restore/SKILL.md:10` хардкодит `R=/media/mydum/Data/Music/restore`, `S=~/Desktop/projects/...` — у другого пользователя продукт молча сломается. Фикс: пути из `rnext.workspace.data_root()`; в скилле — «путь из config/workspace.json». (Smell: Shotgun Surgery / Data Clumps)
3. **[MEDIUM] Док-дрейф после сплита харнесса.** `skills/release-restore/SKILL.md:62` ссылается на `scripts/status_report.py`, файл в `research/status_report.py`. Фикс: сверить все пути во всех SKILL.md.
4. **[MEDIUM] Легаси-батч живёт рядом с продуктом.** `scripts/release_batch.py` (старый алгоритм) не помечен устаревшим; пользователь уже получил им «не то» качество. Фикс: deprecation в docstring/help или удалить.
5. **[MEDIUM] Гейт спайков не смотрит на источник.** `scripts/qc_gates.py:110-112` считает спайки абсолютно; у горячих мастеров FAIL'ит строго лучший результат (реальный кейс: Sometimes, 748 спайков в исходнике → 35 в реставре, гейт FAIL). Фикс: режим `--orig` — FAIL только если в выходе спайков больше, чем в источнике.
6. **[LOW] Дубли путей `$S`/`$R` в каждом SKILL.md.** Фикс: один экспорт из `rnext.workspace`.
7. **[LOW] Версии разъехались**: `pyproject.toml` 2.0.0, история — v2.1.

## Spec (README/docs vs реальность)

1. **[HIGH] «Open-source ready» ≠ портируемость.** Запуск требует `config/workspace.json` с путями этой машины, Transmission+RPC, живые куки rutracker; при протухании кук `rnext/rtclient.py:82` кидает «browser fallback needed», а самого фолбэка в репо нет (восстанавливается вручную через браузер). Фикс: `docs/setup.md` + скрипт browser-refresh кук + template `config/workspace.example.json`.
2. **[MEDIUM] Тесты покрывают только DSP** (`tests/test_dsp.py`, 6 тестов). Агент-критичные парсеры без тестов: `tracker_hunt.parse_music/parse_video`, `rttopic.parse_topic_page/parse_dr_table`, `finish_release`. Фикс: юнит-тесты на сохранённых HTML-фикстурах.
3. **[MEDIUM] Нет CI** (.github отсутствует) при заявленной open-source готовности. Фикс: GitHub Actions с pytest.
4. **[LOW] Нет CODING_STANDARDS.md / CONTRIBUTING.md.**
5. Заявления README про реставрационный цикл выполнены и проверены живьём (per-track, гейты, DR-отчёты/спектры). Незакрытый dirty-диф — только YAML-кавычки в 2 SKILL.md.

## Что хорошо
Дисциплина секретов (куки/UA/RPC в `~/.config`, в репо ничего); `rnext/` — глубокие модули, research переиспользует rnext без дублей парсеров; скиллы кодифицируют набитые шишки; плеер офлайн; .gitignore корректен; тесты зелёные.

## Quick wins
1. Сверить пути в SKILL.md (status_report и др.).
2. `qc_gates --orig` для спайков.
3. Один скорер релизов в `rnext/planner`.
4. Тесты на parse_music / parse_topic_page / parse_dr_table.
5. `docs/setup.md` + browser-refresh кук; deprecate `release_batch.py`; bump версии; закоммитить YAML-фикс; CI с pytest.

## Вердикт
Ядро делает то, что обещает README. Чужому пользователю продукт пока не готов: непортируемые пути, недокументированный вход (куки/Transmission), нет тестов на парсерах и CI. Оценка объёма: 1–2 дня.
