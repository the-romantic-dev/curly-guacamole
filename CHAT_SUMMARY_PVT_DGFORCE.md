# Chat Summary: PVT-DGForce

## Исходная задача

Сопоставить статью **DG-Force: Disentangling and Gathering Forensic Cues is Needed for Image Manipulation Localization** (`D:/Downloads/5818.pdf`) с проектом и затем реализовать вариант DG-Force с текущим энкодером `pvt_v2_b2`.

## Что было в проекте до изменений

В проекте уже была DG-Force-inspired реализация `src/modules/forensic_disentangle.py`:

- post-stage PFD/EFD на итоговых картах четырёх стадий;
- soft patch occupancy и boundary supervision;
- token-wise patch/edge softmax balance;
- частичный cross-scale transfer;
- JPEG forensic branch, EMCAD и опциональный BiFPN.

Основное отличие от статьи: модуль выполнялся один раз после каждой стадии, а не внутри отдельных Transformer-блоков. Полный intra-stage MFT отсутствовал.

## Согласованный дизайн нового эксперимента

Создать полностью отдельную архитектуру без совместимости со старыми checkpoint:

- оставить `pvt_v2_b2`, JPEG branch и EMCAD;
- вставить DFDG после каждого Transformer-блока стадий 1-3;
- использовать глубины PVT `[3, 4, 6, 3]`, то есть 13 DFDG-модулей;
- intra-scale пары:
  - stage 1: block 0 -> block 2;
  - stage 2: block 0 -> block 3;
  - stage 3: block 0 -> block 4 и block 1 -> block 5;
- cross-scale transfer stage 1 -> 2 и stage 2 -> 3;
- objective статьи: `2 * mask BCE + patch BCE + edge BCE`;
- отдельный model type, конфиг и checkpoint family.

Спецификация и implementation plan:

- `docs/superpowers/specs/2026-09-17-pvt-dgforce-design.md`
- `docs/superpowers/plans/2026-09-17-pvt-dgforce.md`

Каталог `docs/` игнорируется `.gitignore`, поэтому документы присутствуют локально. Попытка коммита не удалась из-за запрета создания `.git/index.lock`.

## Реализованные файлы

### Новые

- `src/modules/dgforce.py`
  - `PatchForensicDisentangle`;
  - `EdgeForensicDisentangle`;
  - `DFDGLevel`;
  - `IntraScaleTransfer`;
  - `CrossScaleTransfer`.
- `src/modules/pvt_dgforce.py`
  - явное исполнение отдельных блоков `pvt_v2_b2`;
  - 13 layer-level DFDG;
  - четыре intra-scale пары;
  - два cross-stage transfer для patch и edge потоков.
- `src/modules/pvt_dgforce_segmenter.py`
  - независимый segmenter;
  - PVT-DGForce encoder + существующие JPEG fusion и EMCAD.
- `configs/experiments/pvt_dgforce.yaml`
- `tests/test_dgforce.py`

### Изменённые

- `src/config.py`
  - `ModelConfig.architecture`;
  - параметры DG-Force;
  - `LossConfig.mode` и `mask_weight`;
  - валидация отдельной архитектуры/objective.
- `src/losses.py`
  - `DGForceLoss`;
  - `build_loss`;
  - обратная совместимость `SegmentationLoss` с расширенным `LossConfig`.
- `src/training/builders.py`
  - выбор между старым `Segmenter` и `PVTDGForceSegmenter`.
- `src/training/engine.py`, `src/training/validation.py`
  - построение выбранного objective через `build_loss`.
- `tests/test_builders.py`, `tests/test_config.py`.

## Проверки

Полный набор:

- собрано 552 теста;
- 551 passed;
- 1 skipped;
- 0 failed.

Для запуска тестов в текущем conda-окружении NumPy необходимо импортировать раньше Torch, иначе конфликтуют две копии Intel OpenMP:

```powershell
$env:TRITON_CACHE_DIR='D:\Projects\aiijc2026_final\runs\.triton_cache'
D:\Apps\anaconda3\envs\challenges\python.exe -c "import numpy, pytest, sys; sys.exit(pytest.main(['tests','-q']))"
```

Triton cache перенесён в `runs/`, потому что sandbox запрещает запись в пользовательский `C:/Users/.../.triton/cache`.

Интеграционный smoke подтвердил:

- выходную маску правильного размера;
- 13 patch и 13 edge predictions во время обучения;
- конечные градиенты в DFDG, intra-MFT, cross-MFT, PVT backbone и EMCAD.

Параметры:

- baseline: `26,940,438`;
- PVT-DGForce: `41,509,353`;
- добавлено: `14,568,915`.

## Почему получилось около 203 GFLOPs

Измерения:

| Модель | 640x640 | 768x768 |
|---|---:|---:|
| Baseline | 71.57 GFLOPs | примерно 102 GFLOPs |
| PVT-DGForce | 140.41 GFLOPs | 201.37 GFLOPs |

203 GFLOPs соответствует входу около `768x768`. Стоимость растёт примерно пропорционально площади: `(768 / 640)^2 = 1.44`.

Главный источник перерасхода — текущий `IntraScaleTransfer`. Для каждого patch/edge потока он использует полноканальные операции:

```text
3x3 Conv: C -> C
3x3 Conv: 2C -> C
```

При 640x640 только intra-scale MFT стоит около 58 GFLOPs. Cross-scale MFT и остальные DFDG добавляют ещё около 11 GFLOPs. Это существенно тяжелее lightweight-варианта, подразумеваемого статьёй.

Рекомендуемое следующее улучшение: выполнять intra-scale transfer в bottleneck-пространстве `C/r`, а не в полном `C`. Это сохранит структуру DG-Force и должно значительно снизить FLOPs.

## Запуск эксперимента

```powershell
D:\Apps\anaconda3\envs\challenges\python.exe -m src.training --config configs/experiments/pvt_dgforce.yaml
```

## Состояние Git

Изменения оставлены незакоммиченными в текущей ветке `codex/jpeg640-baseline`, как запросил пользователь. Существовавшие до работы удаления файлов:

- `runs/emcad_mixed_original_fixed_val_natural_q_light/notes.md`;
- `runs/emcad_mixed_original_fixed_val_natural_q_light_pos_dice/notes.md`.

Они не относятся к PVT-DGForce и не изменялись в рамках этой задачи.
