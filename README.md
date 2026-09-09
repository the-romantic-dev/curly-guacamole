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

Все параметры явно перечислены в `configs/baseline.yaml`. Остальные рецепты наследуют его и задают только отличия. Загрузчик также поддерживает минимальный конфиг со встроенными defaults:

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

Параметры оборудования можно переопределить в `.env` или переменных процесса:

```dotenv
AIIJC_BATCH_SIZE=4
AIIJC_ACCUM_STEPS=4
AIIJC_AMP=bf16
AIIJC_DEVICE=cuda
AIIJC_WORKERS=10
```

Приоритет: переменные процесса → `.env` → YAML, включая наследуемые рецепты. Пустое значение оставляет настройку из YAML. Для `amp` допустимы `off`, `fp16`, `bf16`. Пример есть в [.env.example](.env.example). Переопределения применяются при `load_experiment_config`; snapshots сохраняют фактические значения, а `ExperimentConfig.from_dict(snapshot)` не подменяет параметры обучения настройками текущей видеокарты. Batch size, accumulation и AMP могут влиять на результат обучения; одинаковый effective batch не гарантирует одинаковое поведение BatchNorm.

По умолчанию `resume: false`; занятое имя получает суффикс. Для продолжения укажите фактическое имя папки и `train.resume: true`. Версия pipeline, модель, loss, augmentation, геометрия и provenance проверяются до перезаписи snapshot.

## Оценка и результаты

### Профиль скорости обучения

Изолированный CPU-микробенчмарк local-препроцессинга, без датасета и GPU:

```bash
python -m src.data.benchmark_local --output profiles/local_preprocess.json
```

Сравнивает прежний полный float32-буфер с записью групп по три канала сразу в итоговый AMP-буфер, проверяет побитовое совпадение и измеряет цветовое преобразование, нормализацию, разности, resize, cast и запись CHW. Использует синтетические исходники 640×640, 1024×1024 и 1536×2048, один CPU-поток, чередующийся порядок замеров. Можно задать `--dtype float16` или `--dtype float32`. В worker-профиле преобразование local-типа теперь включено в `local_features`; отдельный `local_cast` равен нулю. Сравнивайте сумму этих двух этапов с прежним отчётом.

Для замера подготовки изображений внутри DataLoader-воркеров:

```bash
AIIJC_WORKERS=4 python -m src.training.profiling --config configs/local.yaml --profile-data --batches 64 --output profiles/data.json
```

Блок `results.loader.worker_profile` содержит среднее, p50, p95 (мс на изображение) и долю каждого этапа: read_image, read_mask, qtable, setup, jpeg, dct, geometry, photometric, local_features, local_cast, resize, tensorize. Учитываются только потреблённые батчи после warmup. Это wall time внутри воркеров, включая возможные паузы планировщика; сборка батча, IPC, pinning и очереди сюда не входят. Время параллельных воркеров нельзя складывать с временем CUDA. Тайминги возвращаются в основной процесс вместе с батчем и удаляются до отправки на GPU. В обычном обучении эта диагностика выключена.

Local-входы для CUDA передаются в точности выбранного AMP: BF16 или FP16 (30 MiB вместо 60 MiB на изображение). Разности и resize вычисляются в float32; при `amp=off` и на CPU формат остаётся float32. Обучение использует отдельный copy stream с ожиданием готовности данных на compute stream, чтобы перенос следующего батча мог перекрываться с уже поставленными в очередь вычислениями. Состояние модели и формат checkpoint не меняются.

Сравнить прежнюю передачу float32, только компактную передачу и компактную передачу с отдельным stream:

```bash
python -m src.training.profiling --config configs/local.yaml --compare-transfer --batches 64 --output profiles/transfer.json
```

В итоговой строке `Transport comparison` сравнивайте wall time режимов `legacy`, `compact`, `optimized`. Для `optimized` поле `transfer` показывает неперекрытое ожидание compute stream, а не полную длительность копирования. Порядок этого короткого сравнения фиксирован; небольшие различия требуют повторного замера.

На сервере с CUDA, отдельно от работающего обучения:

```bash
python -m src.training.profiling --config configs/local.yaml --warmup 8 --batches 32 --output profiles/local.json
```

Команда использует параметры оборудования из `.env`. Две временные модели без загрузки pretrained и без сохранения запусков сравнивают обычный DataLoader (`loader`) с повторением одного готового батча на GPU (`gpu_replay`). Warmup исключён из статистики. Отчёт содержит wall time на батч и CUDA Events для transfer, forward+loss, backward, optimizer (включая clipping), EMA+scheduler и учёта метрик. Для optimizer/EMA также выводится время на обновление, учитывающее accumulation. CUDA Events измеряют интервалы в stream, включая паузы подачи команд CPU; это не сумма длительностей отдельных kernels. Replay исключает загрузку/перенос данных, но использует один фиксированный батч, поэтому является диагностическим сравнением. Архитектура и loss берутся из выбранного конфига; существующие веса и результаты не изменяются.

При сборке submission путь к данным берётся из `AIIJC_DATA_PATH` текущей среды (переменная процесса → `.env` → `global_config.DATA_PATH`), а не из snapshot обучения. Относительный путь разрешается от корня проекта. Явный аргумент `data_path` / `--data-path` имеет приоритет; без отдельного `template_path` шаблон submission также читается из выбранной папки данных.

- [Протокол валидации](docs/validation_protocol.md): правила development и однократного holdout.
- [Карта экспериментов](docs/experiment_map.md): соответствие старых рецептов новым и архив.
- [Локальная ветка](docs/emcad_local_branch.md): архитектура, geometry и вычислительный бюджет.
- [notebooks/experiments.ipynb](notebooks/experiments.ipynb): отдельные таблицы текущих запусков и архива.

```powershell
& 'D:/Apps/anaconda3/envs/challenges/python.exe' -m src.inference runs/local submissions/local
& 'D:/Apps/anaconda3/envs/challenges/python.exe' -m src.eval --run runs/local
```

Вторая команда — окончательный holdout, запускается только после выбора кандидата. Протокол берётся из snapshot; holdout не участвует в подборе порогов.

Выполненные EMCAD-запуски сохранены под короткими именами конфигов в `runs/` (ранние baseline имеют суффиксы `legacy` и `legacy_dct`), остальные исторические веса, метрики и карточки — в `runs/archive/`. Их конфиги описывают предыдущий pipeline: для воспроизведения используйте код `main`. Новая ветка явно отклоняет старые checkpoints без версии, чтобы не подменять их DCT или архитектуру.

## Проверки

```powershell
& 'D:/Apps/anaconda3/envs/challenges/python.exe' -m pytest
```

Пределы соревнования: 100 GFLOPs и 50 мс/изображение на H100. FLOPs считаются полным eval-forward через `FlopCounterMode`. Измерение H100 latency с preprocessing и улучшение качества требуют серверного эксперимента; рефакторинг сам обучение не запускает.
