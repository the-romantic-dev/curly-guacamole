# AIIJC: EMCAD experiments

Код находится в `src/`, настройки экспериментов — в `configs/`, результаты — в `runs/<name>/`. Окружение Python: `challenges`.

Базовый pipeline этой ветки — **PVT-v2-B2 + EMCAD + forensic-ветка**, вход 640. Его не нужно собирать цепочкой флагов:

- mixed full-frame/crop обучение: по умолчанию 50% полных кадров, 50% кропов, foreground-guided выбор половины кропов; последние две эпохи full-frame;
- DCT с правильным natural-порядком таблицы Pillow, без legacy-переключателя;
- фиксированный train/development/holdout manifest и проверенные originals с нулевой маской в train;
- валидация по исходным маскам: восстановление вероятностей до оригинального размера перед порогом;
- выбор лучшего checkpoint и порогов только по development AIC, отдельные срезы provided/originals.

## Эксперименты

| Конфиг | Отличие от baseline |
|---|---|
| [baseline](configs/baseline.yaml) | Базовая модель, BCE + Dice на всех примерах |
| [positive_dice](configs/positive_dice.yaml) | Dice только на позитивных; BCE по всем |
| [local](configs/local.yaml) | Positive Dice + локальная RGB/residual-ветка 1024 |
| [stride4](configs/stride4.yaml) | Spatial refinement 144 каналов после EMCAD |
| [stride2_rgb](configs/stride2_rgb.yaml) | RGB refinement stride 2, batch 2 × accumulation 8 |
| [dct_aux](configs/dct_aux.yaml) | Дополнительная DCT-голова с весом 0.2 |

Общая база — 6 эпох × 24 000 показов, batch 4 × accumulation 4, seed 42, EMA, BF16. `local` наследует `positive_dice`; остальные варианты — `baseline`. Гиперпараметры можно менять, общие исправления отключить нельзя.

В `baseline.yaml` достаточно:

```yaml
paths:
  run_name: baseline
```

Все значения по умолчанию определены в [src/config.py](src/config.py) и [AugmentationConfig](src/data/augmentation/base.py). Snapshots сохраняют полностью разрешённые настройки и `pipeline_version: emcad_v1`.

## Запуск

Откройте [notebooks/train.ipynb](notebooks/train.ipynb), выберите `experiment` и выполните ячейки. До обучения ноутбук проверяет manifest и полный FLOPs-бюджет без загрузки pretrained-весов. Последняя ячейка запускает обучение.

Из корня проекта:

```powershell
& 'D:/Apps/anaconda3/envs/challenges/python.exe' -m src.training --config configs/local.yaml
```

Новый эксперимент содержит только отличия:

```yaml
extends: positive_dice.yaml
paths:
  run_name: longer
train:
  epochs: 18
```

`extends` разрешается относительно YAML; словари объединяются, списки заменяются. `.env`/переменные среды `AIIJC_DATA_PATH` и `AIIJC_RUNS_PATH` задают машинные пути. `dataset.protocol_path` — необязательная настройка расположения обязательного manifest; по умолчанию используется сохранённый `runs/validation_protocol_20260908/protocol`, даже если текущая рабочая папка другая. Это не переключатель старой/новой валидации.

По умолчанию `resume: false`; занятое имя получает суффикс. Для продолжения укажите фактическое имя папки и `train.resume: true`. Версия pipeline, модель, loss, augmentation, геометрия и provenance проверяются до перезаписи snapshot.

## Оценка и результаты

- [Протокол валидации](docs/validation_protocol.md): правила development и однократного holdout.
- [Карта экспериментов](docs/experiment_map.md): соответствие старых рецептов новым и архив.
- [Локальная ветка](docs/emcad_local_branch.md): архитектура, geometry и вычислительный бюджет.
- [notebooks/experiments.ipynb](notebooks/experiments.ipynb): отдельные таблицы текущих запусков и архива.

```powershell
& 'D:/Apps/anaconda3/envs/challenges/python.exe' -m src.inference runs/local submissions/local
& 'D:/Apps/anaconda3/envs/challenges/python.exe' -m src.eval --run runs/local
```

Вторая команда — окончательный holdout, запускается только после выбора кандидата. Протокол берётся из snapshot; holdout не участвует в подборе порогов.

Исторические веса, метрики и карточки сохранены под исходными именами в `runs/archive/`. Их конфиги описывают предыдущий pipeline: для воспроизведения используйте код `main`. Новая ветка явно отклоняет старые checkpoints без версии, чтобы не подменять их DCT или архитектуру.

## Проверки

```powershell
& 'D:/Apps/anaconda3/envs/challenges/python.exe' -m pytest
```

Пределы соревнования: 100 GFLOPs и 50 мс/изображение на H100. FLOPs считаются полным eval-forward через `FlopCounterMode`. Измерение H100 latency с preprocessing и улучшение качества требуют серверного эксперимента; рефакторинг сам обучение не запускает.
