> Исторический каталог до перехода на emcad_v1. Названия рецептов относятся к main; исходные YAML доступны в Git (`git show main:configs/<name>.yaml`). Результаты EMCAD находятся в runs, остальные — в runs/archive.

# Карта экспериментов

Срез локальных файлов на 2026-09-09. Это каталог наблюдаемых артефактов, а не очередь запусков и не рейтинг моделей. Конфиги и карточки в рабочем дереве включают незакоммиченные изменения.

Охват: 19 YAML-рецептов, 17 тренировочных snapshots и 2 служебные папки. У 4 рецептов нет локальной папки по run_name, у 2 выполненных запусков нет текущего рецепта с тем же именем.

## С чего продолжать

- Для новых сравнений использовать фиксированный train/development-протокол. Короткая опорная модель — сохранённый `pvt_v2_b2_protocol_originals_light` (UNet, 6 эпох); кандидат для сравнения — `pvt_v2_b2_emcad_mixed_original_fixed_val_light` (EMCAD, 6 эпох). Это опорные результаты, а не указание запускать текущие YAML без проверки.
- Длинный `pvt_v2_b2_protocol_originals` обучался 18 эпох. Текущий одноимённый рецепт задаёт 8: для повторения брать сохранённые настройки и новое имя запуска.
- Исторические mixed/original-эксперименты оставить отдельной серией. Старый full_frame с иной оценкой, диагностику BatchNorm и аудит протокола учитывать отдельно.
- Следующий natural_q-вариант уже описан конфигом, но папки с его точным run_name локально нет. Это не доказательство, что он нигде не запускался.

## Как читать источники

`configs/*.yaml` — текущие рецепты; `runs/<name>/config.yaml` — фактически сохранённые параметры; `summary.json` — сохранённые метрики; `notes.md` — авторские связи и LB. Связь рецепта с запуском ниже установлена по точному `paths.run_name`, а не доказанной истории происхождения. YAML, выбранный в ноутбуке, найден по исходному коду ячеек; это не доказывает, что конкретный запуск выполнен именно им.

## Текущие рецепты

В колонке изменений перечислены отличия от разрешённого непосредственного родителя. Служебное имя вынесено отдельно; resume показан как действие. Отсутствие папки означает только отсутствие локальных артефактов по этому имени.

| Рецепт | Родитель | Изменения | Эпохи | resume | Папка по run_name | Ноутбук |
|---|---|---|---|---|---|---|
| [baseline.yaml](../configs/baseline.yaml) | — | Полная исходная база | 8 | true | [pvt_v2_b2_full_frame](../../runs/archive/pvt_v2_b2_full_frame) | [baseline_pipeline.ipynb](../notebooks/baseline_pipeline.ipynb) |
| [baseline_mixed_original.yaml](../configs/baseline_mixed_original.yaml) | `baseline.yaml` | `augmentation.full_frame_probability=0.5`; `augmentation.foreground_crop_probability=0.5`; `augmentation.final_full_frame_epochs=2`; `eval.resolution="original"` | 8 | true | [pvt_v2_b2_mixed_original](../../runs/archive/pvt_v2_b2_mixed_original) | [baseline_mixed_original.ipynb](../notebooks/baseline_mixed_original.ipynb) |
| [baseline_mixed_original_long.yaml](../configs/baseline_mixed_original_long.yaml) | `baseline_mixed_original.yaml` | `augmentation.final_full_frame_epochs=4`; `train.epochs=18` | 18 | true | [pvt_v2_b2_mixed_original_long](../../runs/archive/pvt_v2_b2_mixed_original_long) | [baseline_mixed_original_long.ipynb](../notebooks/baseline_mixed_original_long.ipynb) |
| [baseline_mixed_original_natural_q.yaml](../configs/baseline_mixed_original_natural_q.yaml) | `baseline_mixed_original.yaml` | `dataset.jpeg_qtable_order="natural"` | 8 | true | [pvt_v2_b2_mixed_original_natural_q](../../runs/archive/pvt_v2_b2_mixed_original_natural_q) | [baseline_mixed_original_natural_q.ipynb](../notebooks/baseline_mixed_original_natural_q.ipynb) |
| [baseline_positive_dice_mixed_original.yaml](../configs/baseline_positive_dice_mixed_original.yaml) | `baseline_mixed_original.yaml` | `loss.dice_scope="positive"`; `loss.dice_weight=1` | 8 | true | [pvt_v2_b2_positive_dice_mixed_original](../../runs/archive/pvt_v2_b2_positive_dice_mixed_original) | [baseline_positive_dice_mixed_original.ipynb](../notebooks/baseline_positive_dice_mixed_original.ipynb) |
| [baseline_protocol.yaml](../configs/baseline_protocol.yaml) | `baseline_mixed_original.yaml` | `dataset.protocol_path="runs/validation_protocol_20260908/protocol"`; `dataset.train_originals=false` | 8 | true | `pvt_v2_b2_protocol_baseline` — папки нет | — |
| [baseline_protocol_originals.yaml](../configs/baseline_protocol_originals.yaml) | `baseline_protocol.yaml` | `dataset.train_originals=true` | 8 | true | [pvt_v2_b2_protocol_originals](../../runs/archive/pvt_v2_b2_protocol_originals) | — |
| [baseline_protocol_originals_light.yaml](../configs/baseline_protocol_originals_light.yaml) | `baseline_protocol_originals.yaml` | `train.epochs=6` | 6 | true | [pvt_v2_b2_protocol_originals_light](../../runs/archive/pvt_v2_b2_protocol_originals_light) | — |
| [efficientvit_b2_letterbox.yaml](../configs/efficientvit_b2_letterbox.yaml) | `efficientvit_b2_mixed_original.yaml` | `dataset.resize_mode="letterbox"` | 8 | true | `efficientvit_b2_1024_letterbox` — папки нет | [efficientvit_b2_letterbox.ipynb](../notebooks/efficientvit_b2_letterbox.ipynb) |
| [efficientvit_b2_mixed_original.yaml](../configs/efficientvit_b2_mixed_original.yaml) | `baseline_mixed_original.yaml` | `model.encoder_name="efficientvit_b2.r288_in1k"`; `dataset.image_size=1024`; `train.batch_size=2`; `train.accum_steps=8` | 8 | true | [efficientvit_b2_1024_mixed_original](../../runs/archive/efficientvit_b2_1024_mixed_original) | [efficientvit_b2_mixed_original.ipynb](../notebooks/efficientvit_b2_mixed_original.ipynb) |
| [emcad_mixed_original.yaml](../configs/emcad_mixed_original.yaml) | `baseline_mixed_original.yaml` | `model.decoder_name="emcad"`; `model.decoder_kwargs.kernel_sizes=[1,3,5]`; `model.decoder_kwargs.expansion_factor=2`; `model.decoder_kwargs.lgag_kernel_size=3`; `model.decoder_kwargs.activation="relu"` | 8 | true | [pvt_v2_b2_emcad_mixed_original](../../runs/baseline_legacy) | [emcad_mixed_original.ipynb](../notebooks/emcad_mixed_original.ipynb) |
| [emcad_mixed_original_fixed_val_light.yaml](../configs/emcad_mixed_original_fixed_val_light.yaml) | `baseline_protocol_originals_light.yaml` | `model.decoder_name="emcad"`; `model.decoder_kwargs.kernel_sizes=[1,3,5]`; `model.decoder_kwargs.expansion_factor=2`; `model.decoder_kwargs.lgag_kernel_size=3`; `model.decoder_kwargs.activation="relu"` | 6 | true | [pvt_v2_b2_emcad_mixed_original_fixed_val_light](../../runs/baseline_legacy_dct) | — |
| [emcad_mixed_original_fixed_val_natural_q_light.yaml](../configs/emcad_mixed_original_fixed_val_natural_q_light.yaml) | `emcad_mixed_original_fixed_val_light.yaml` | `dataset.jpeg_qtable_order="natural"` | 6 | true | `emcad_mixed_original_fixed_val_natural_q_light` — папки нет | — |
| [emcad_stride2_rgb_mixed_original.yaml](../configs/emcad_stride2_rgb_mixed_original.yaml) | `emcad_mixed_original.yaml` | `model.decoder_kwargs.rgb_refinement_channels=32`; `model.decoder_kwargs.rgb_detail_channels=24`; `train.batch_size=2`; `train.accum_steps=8` | 8 | true | [pvt_v2_b2_emcad_640_stride2_rgb_mixed_original](../../runs/stride2_rgb) | [emcad_stride2_rgb_mixed_original.ipynb](../notebooks/emcad_stride2_rgb_mixed_original.ipynb) |
| [emcad_stride4_mixed_original.yaml](../configs/emcad_stride4_mixed_original.yaml) | `emcad_mixed_original.yaml` | `model.decoder_kwargs.output_refinement_channels=144` | 8 | true | [pvt_v2_b2_emcad_640_stride4_mixed_original](../../runs/stride4) | [emcad_stride4_mixed_original.ipynb](../notebooks/emcad_stride4_mixed_original.ipynb) |
| [segformer_640_dct_aux_mixed_original.yaml](../configs/segformer_640_dct_aux_mixed_original.yaml) | `segformer_mixed_original.yaml` | `model.use_forensics=true`; `model.dct_aux_weight=0.2` | 8 | false | `pvt_v2_b2_segformer_640_dct_aux_mixed_original` — папки нет | [segformer_640_dct_aux_mixed_original.ipynb](../notebooks/segformer_640_dct_aux_mixed_original.ipynb) |
| [segformer_640_rgb_only_mixed_original.yaml](../configs/segformer_640_rgb_only_mixed_original.yaml) | `segformer_mixed_original.yaml` | `model.use_forensics=false`; `model.dct_aux_weight=0.0` | 8 | false | [pvt_v2_b2_segformer_640_rgb_only_mixed_original](../../runs/archive/pvt_v2_b2_segformer_640_rgb_only_mixed_original) | [segformer_640_rgb_only_mixed_original.ipynb](../notebooks/segformer_640_rgb_only_mixed_original.ipynb) |
| [segformer_672_mixed_original.yaml](../configs/segformer_672_mixed_original.yaml) | `segformer_mixed_original.yaml` | `dataset.image_size=672` | 8 | false | [pvt_v2_b2_segformer_672_mixed_original](../../runs/archive/pvt_v2_b2_segformer_672_mixed_original) | [segformer_672_mixed_original.ipynb](../notebooks/segformer_672_mixed_original.ipynb) |
| [segformer_mixed_original.yaml](../configs/segformer_mixed_original.yaml) | `baseline_mixed_original.yaml` | `model.decoder_name="segformer"`; `model.decoder_kwargs.embed_dim=128` | 8 | false | [pvt_v2_b2_segformer_mixed_original](../../runs/archive/pvt_v2_b2_segformer_mixed_original) | [segformer_mixed_original.ipynb](../notebooks/segformer_mixed_original.ipynb) |

## Сохранённые запуски

AIC округлён до 6 знаков, LB взят из ручной карточки и не проверялся на сайте. «Результат сохранён» не означает подтверждённое завершение: для старых запусков нет явного training_complete. Бюджет в колонке — epochs × epoch_size из snapshot, а не проверенное число обработанных примеров.

| Запуск | Группа оценки | Бюджет | AIC | LB (ручной) | Состояние |
|---|---|---|---|---|---|
| [efficientvit_b2_1024_mixed_original](../../runs/archive/efficientvit_b2_1024_mixed_original) | L: legacy original | 8 × 24000 | 0.848001 | — | Результат сохранён; завершение не подтверждено |
| [pvt_v2_b2_640_letterbox](../../runs/archive/pvt_v2_b2_640_letterbox) | L: legacy original | 8 × 24000 | 0.914389 | 0.907673 | Результат сохранён; завершение не подтверждено |
| [pvt_v2_b2_emcad_640_dct_aux_mixed_original](../../runs/dct_aux) | L: legacy original | 8 × 24000 | 0.925420 | 0.910268 | Результат сохранён; завершение не подтверждено |
| [pvt_v2_b2_emcad_640_stride2_rgb_mixed_original](../../runs/stride2_rgb) | L: legacy original | 8 × 24000 | 0.918998 | — | Результат сохранён; завершение не подтверждено |
| [pvt_v2_b2_emcad_640_stride4_mixed_original](../../runs/stride4) | L: legacy original | 8 × 24000 | 0.922854 | 0.911157 | Результат сохранён; завершение не подтверждено |
| [pvt_v2_b2_emcad_mixed_original](../../runs/baseline_legacy) | L: legacy original | 8 × 24000 | 0.927039 | 0.915578 | Результат сохранён; завершение не подтверждено |
| [pvt_v2_b2_emcad_mixed_original_fixed_val_light](../../runs/baseline_legacy_dct) | D: development + originals | 6 × 24000 | 0.910406 | 0.898549 | Завершён (явно) |
| [pvt_v2_b2_full_frame](../../runs/archive/pvt_v2_b2_full_frame) | R: legacy, resolution не записан | 8 × 24000 | 0.881482 | 0.865449 | Результат сохранён; завершение не подтверждено |
| [pvt_v2_b2_mixed_original](../../runs/archive/pvt_v2_b2_mixed_original) | L: legacy original | 8 × 24000 | 0.925063 | 0.919654 | Результат сохранён; завершение не подтверждено |
| [pvt_v2_b2_mixed_original_long](../../runs/archive/pvt_v2_b2_mixed_original_long) | L: legacy original | 18 × 24000 | 0.939410 | 0.924361 | Результат сохранён; завершение не подтверждено |
| [pvt_v2_b2_mixed_original_natural_q](../../runs/archive/pvt_v2_b2_mixed_original_natural_q) | L: legacy original | 8 × 24000 | 0.927416 | 0.912873 | Результат сохранён; завершение не подтверждено |
| [pvt_v2_b2_positive_dice_mixed_original](../../runs/archive/pvt_v2_b2_positive_dice_mixed_original) | L: legacy original | 8 × 24000 | 0.929025 | 0.911481 | Результат сохранён; завершение не подтверждено |
| [pvt_v2_b2_protocol_originals](../../runs/archive/pvt_v2_b2_protocol_originals) | D: development + originals | 18 × 24000 | 0.934892 | 0.928380 | Завершён (явно) |
| [pvt_v2_b2_protocol_originals_light](../../runs/archive/pvt_v2_b2_protocol_originals_light) | D: development + originals | 6 × 24000 | 0.912757 | 0.898597 | Завершён (явно) |
| [pvt_v2_b2_segformer_640_rgb_only_mixed_original](../../runs/archive/pvt_v2_b2_segformer_640_rgb_only_mixed_original) | L: legacy original | 8 × 24000 | 0.917749 | 0.910261 | Результат сохранён; завершение не подтверждено |
| [pvt_v2_b2_segformer_672_mixed_original](../../runs/archive/pvt_v2_b2_segformer_672_mixed_original) | L: legacy original | 8 × 24000 | 0.922998 | 0.897630 | Результат сохранён; завершение не подтверждено |
| [pvt_v2_b2_segformer_mixed_original](../../runs/archive/pvt_v2_b2_segformer_mixed_original) | L: legacy original | 8 × 24000 | 0.920019 | 0.908503 | Результат сохранён; завершение не подтверждено |

## Сопоставимость

- **R:** `pvt_v2_b2_full_frame`: snapshot и summary не содержат resolution. README описывает исторический baseline как resized. Его AIC не сравнивать напрямую с original-resolution оценками.
- **L:** original-resolution legacy scores, обычно 20 101 positive и 639 negative. Одинаковые количества не доказывают идентичность выборок; совпадение сохранённых строк проверено ниже. Различия бюджета, преобразований и версии кода всё равно ограничивают вывод об одной причине улучшения.
- **D:** 20 102 positive и 9 140 negative, из них 8 502 добавленных исходника. Combined AIC нельзя смешивать с L. Provided AIC также не делает старое и новое разбиение автоматически одинаковыми.

Для трёх D-запусков `training_rows.parquet`: 87,728 строк; полное совпадение pandas.DataFrame.equals (значения, порядок, колонки и типы): **да**.

Для трёх D-запусков `development_rows.parquet`: 29,242 строк; полное совпадение pandas.DataFrame.equals (значения, порядок, колонки и типы): **да**.

| Запуск D | protocol_digest | Provided AIC | Originals FPR | Holdout |
|---|---|---|---|---|
| `pvt_v2_b2_emcad_mixed_original_fixed_val_light` | `f3811932cac1ce83dc73d287084ba9a76857ed71291fb2fd72ea0b027ebbb877` | 0.905193 | 0.016820 | False |
| `pvt_v2_b2_protocol_originals` | `a72b59383a0acc5fa8ecce2a757f81916fb27162caf6b61a07ae9d9d2c9be580` | 0.934087 | 0.005881 | False |
| `pvt_v2_b2_protocol_originals_light` | `a72b59383a0acc5fa8ecce2a757f81916fb27162caf6b61a07ae9d9d2c9be580` | 0.906840 | 0.011880 | False |

Хеш протокола у EMCAD отличается от двух UNet, хотя сохранённые train/development-таблицы совпадают. Это позволяет сопоставлять наблюдаемые development-оценки на этих строках, но не доказывает неизменность полного протокола или кода. Причина смены хеша не установлена. Не подменять хеши в snapshots ради resume/holdout.

**UNet light → EMCAD light не является чистой заменой декодера.** По snapshots физический batch изменился 16 → 4, accumulation 1 → 4. Эффективный batch остался 16, но BatchNorm работает с физическим batch. Оба запуска имеют 6 эпох; observed development AIC — 0.912757 и 0.910406 соответственно. Эта пара не изолирует влияние архитектуры.

Проверка legacy `oof/val_rows.parquet` относительно `pvt_v2_b2_mixed_original`:

| Запуск | Строки | Полное совпадение таблицы |
|---|---|---|
| efficientvit_b2_1024_mixed_original | 20740 | True |
| pvt_v2_b2_640_letterbox | 20740 | True |
| pvt_v2_b2_emcad_640_dct_aux_mixed_original | 20740 | True |
| pvt_v2_b2_emcad_640_stride2_rgb_mixed_original | 20740 | True |
| pvt_v2_b2_emcad_640_stride4_mixed_original | 20740 | True |
| pvt_v2_b2_emcad_mixed_original | 20740 | True |
| pvt_v2_b2_full_frame | 20740 | True |
| pvt_v2_b2_mixed_original | 20740 | True |
| pvt_v2_b2_mixed_original_long | 20740 | True |
| pvt_v2_b2_mixed_original_natural_q | 20740 | True |
| pvt_v2_b2_positive_dice_mixed_original | 20740 | True |
| pvt_v2_b2_segformer_640_rgb_only_mixed_original | 20740 | True |
| pvt_v2_b2_segformer_672_mixed_original | 20740 | True |
| pvt_v2_b2_segformer_mixed_original | 20740 | True |

False означает различие таблиц; само по себе оно не устанавливает, отличаются изображения или только метаданные. Содержимое файлов изображений этим аудитом не проверялось.

## Расхождения текущих рецептов и snapshots

Сравниваются явно заданные поля разрешённого YAML с одноимёнными полями плоского snapshot; секция decoder_kwargs сохранена при сравнении. Отсутствующие в старом snapshot поля перечислены отдельно: современные defaults не выдаются за доказанный исторический факт. Пути машины исключены. Это сравнение настроек, не проверка воспроизводимости весов.

### baseline.yaml

Snapshot: [config.yaml](../../runs/archive/pvt_v2_b2_full_frame/config.yaml)

Различия: нет среди присутствующих полей.

### baseline_mixed_original.yaml

Snapshot: [config.yaml](../../runs/archive/pvt_v2_b2_mixed_original/config.yaml)

Различия: нет среди присутствующих полей.

### baseline_mixed_original_long.yaml

Snapshot: [config.yaml](../../runs/archive/pvt_v2_b2_mixed_original_long/config.yaml)

Различия: нет среди присутствующих полей.

### baseline_mixed_original_natural_q.yaml

Snapshot: [config.yaml](../../runs/archive/pvt_v2_b2_mixed_original_natural_q/config.yaml)

Различия: нет среди присутствующих полей.

### baseline_positive_dice_mixed_original.yaml

Snapshot: [config.yaml](../../runs/archive/pvt_v2_b2_positive_dice_mixed_original/config.yaml)

Различия: нет среди присутствующих полей.

### baseline_protocol_originals.yaml

Snapshot: [config.yaml](../../runs/archive/pvt_v2_b2_protocol_originals/config.yaml)

Различия: `augmentation.final_full_frame_epochs`: snapshot `4` → рецепт `2`; `train.epochs`: snapshot `18` → рецепт `8`; `train.batch_size`: snapshot `16` → рецепт `4`; `train.accum_steps`: snapshot `1` → рецепт `4`.

### baseline_protocol_originals_light.yaml

Snapshot: [config.yaml](../../runs/archive/pvt_v2_b2_protocol_originals_light/config.yaml)

Различия: `train.batch_size`: snapshot `16` → рецепт `4`; `train.accum_steps`: snapshot `1` → рецепт `4`.

### efficientvit_b2_mixed_original.yaml

Snapshot: [config.yaml](../../runs/archive/efficientvit_b2_1024_mixed_original/config.yaml)

Различия: нет среди присутствующих полей.

### emcad_mixed_original.yaml

Snapshot: [config.yaml](../../runs/pvt_v2_b2_emcad_mixed_original/config.yaml)

Различия: нет среди присутствующих полей.

### emcad_mixed_original_fixed_val_light.yaml

Snapshot: [config.yaml](../../runs/pvt_v2_b2_emcad_mixed_original_fixed_val_light/config.yaml)

Различия: нет среди присутствующих полей.

### emcad_stride2_rgb_mixed_original.yaml

Snapshot: [config.yaml](../../runs/pvt_v2_b2_emcad_640_stride2_rgb_mixed_original/config.yaml)

Различия: нет среди присутствующих полей.

### emcad_stride4_mixed_original.yaml

Snapshot: [config.yaml](../../runs/pvt_v2_b2_emcad_640_stride4_mixed_original/config.yaml)

Различия: нет среди присутствующих полей.

### segformer_640_rgb_only_mixed_original.yaml

Snapshot: [config.yaml](../../runs/archive/pvt_v2_b2_segformer_640_rgb_only_mixed_original/config.yaml)

Различия: нет среди присутствующих полей.

### segformer_672_mixed_original.yaml

Snapshot: [config.yaml](../../runs/archive/pvt_v2_b2_segformer_672_mixed_original/config.yaml)

Различия: нет среди присутствующих полей.

### segformer_mixed_original.yaml

Snapshot: [config.yaml](../../runs/archive/pvt_v2_b2_segformer_mixed_original/config.yaml)

Различия: нет среди присутствующих полей.

## Происхождение и долги карточек

Авторские parent/baseline/change показаны без исправлений и не трактуются как доказательство единственного изменения. Пустые гипотезы не восстановлены по имени папки.

| Карточка | parent | baseline (с fallback) | change автора | Долг |
|---|---|---|---|---|
| [efficientvit_b2_1024_mixed_original](../../runs/archive/efficientvit_b2_1024_mixed_original/notes.md) | pvt_v2_b2_mixed_original | pvt_v2_b2_mixed_original | Заменил энекодер на легкий EfficientViT B2 и подавал туда изображения с 1024 px рейсайзом | заготовка в тексте; verdict не принимается текущим парсером |
| [pvt_v2_b2_640_letterbox](../../runs/archive/pvt_v2_b2_640_letterbox/notes.md) | pvt_v2_b2_mixed_original | pvt_v2_b2_mixed_original | Ресайз делается пропорицонально сторонам изображения, пустоты заполняются paddings | заготовка в тексте; verdict не принимается текущим парсером |
| [pvt_v2_b2_emcad_640_dct_aux_mixed_original](../../runs/pvt_v2_b2_emcad_640_dct_aux_mixed_original/notes.md) | pvt_v2_b2_emcad_mixed_original | pvt_v2_b2_emcad_mixed_original | Добавил aux голову из forensic ветки в лосс, чтобы она лучше обучалась | заготовка в тексте |
| [pvt_v2_b2_emcad_640_stride2_rgb_mixed_original](../../runs/pvt_v2_b2_emcad_640_stride2_rgb_mixed_original/notes.md) | — | — | — | заготовка в тексте; нет change |
| [pvt_v2_b2_emcad_640_stride4_mixed_original](../../runs/pvt_v2_b2_emcad_640_stride4_mixed_original/notes.md) | pvt_v2_b2_emcad_mixed_original | pvt_v2_b2_emcad_mixed_original | Две dense 3x3 свёртки 64→144→64 с residual после последней стадии EMCAD. | — |
| [pvt_v2_b2_emcad_mixed_original](../../runs/pvt_v2_b2_emcad_mixed_original/notes.md) | pvt_v2_b2_mixed_original | pvt_v2_b2_mixed_original | Замена UNet декодера на более легкий EMCAD декодер | заготовка в тексте; verdict не принимается текущим парсером |
| [pvt_v2_b2_emcad_mixed_original_fixed_val_light](../../runs/pvt_v2_b2_emcad_mixed_original_fixed_val_light/notes.md) | — | — | — | заготовка в тексте; нет change |
| [pvt_v2_b2_full_frame](../../runs/archive/pvt_v2_b2_full_frame/notes.md) | — | — | Бейзлайн | verdict не принимается текущим парсером |
| [pvt_v2_b2_mixed_original](../../runs/archive/pvt_v2_b2_mixed_original/notes.md) | pvt_v2_b2_full_frame | pvt_v2_b2_full_frame | Добавил обучение 50/50 с full frame изображениями без кропа + 2 последние эпохи дотюна на 100% full_frame | заготовка в тексте; verdict не принимается текущим парсером |
| [pvt_v2_b2_mixed_original_long](../../runs/archive/pvt_v2_b2_mixed_original_long/notes.md) | pvt_v2_b2_mixed_original | pvt_v2_b2_mixed_original | Долгий запусук pvt_v2_b2_mixed_original на 18 эпох (4 последние fullframe) | заготовка в тексте |
| [pvt_v2_b2_mixed_original_natural_q](../../runs/archive/pvt_v2_b2_mixed_original_natural_q/notes.md) | pvt_v2_b2_mixed_original | pvt_v2_b2_mixed_original | — | заготовка в тексте; нет change |
| [pvt_v2_b2_positive_dice_mixed_original](../../runs/archive/pvt_v2_b2_positive_dice_mixed_original/notes.md) | — | — | — | заготовка в тексте; нет change |
| [pvt_v2_b2_protocol_originals](../../runs/archive/pvt_v2_b2_protocol_originals/notes.md) | — | — | — | заготовка в тексте; нет change |
| [pvt_v2_b2_protocol_originals_light](../../runs/archive/pvt_v2_b2_protocol_originals_light/notes.md) | — | — | — | заготовка в тексте; нет change |
| [pvt_v2_b2_segformer_640_rgb_only_mixed_original](../../runs/archive/pvt_v2_b2_segformer_640_rgb_only_mixed_original/notes.md) | pvt_v2_b2_segformer_mixed_original | pvt_v2_b2_segformer_mixed_original | Абляция c отключением ветки форензики | заготовка в тексте; verdict не принимается текущим парсером |
| [pvt_v2_b2_segformer_672_mixed_original](../../runs/archive/pvt_v2_b2_segformer_672_mixed_original/notes.md) | pvt_v2_b2_segformer_mixed_original | pvt_v2_b2_segformer_mixed_original | Увеличение входного размера рейсайза с 640 до 672 | заготовка в тексте; verdict не принимается текущим парсером |
| [pvt_v2_b2_segformer_mixed_original](../../runs/archive/pvt_v2_b2_segformer_mixed_original/notes.md) | pvt_v2_b2_mixed_original | pvt_v2_b2_mixed_original | Заменил UNet декодер на легкий MLP декодер из SegFormer | заготовка в тексте |

## Запуски без текущего рецепта и служебные папки

Сравнение фактических snapshots уточняет две потерянные связи:

- `pvt_v2_b2_640_letterbox` относительно `pvt_v2_b2_mixed_original`: изменены только run_name и resize_mode (`stretch` → `letterbox`). Это совпадает с авторским parent.
- `pvt_v2_b2_emcad_640_dct_aux_mixed_original` относительно `pvt_v2_b2_emcad_mixed_original`: другое run_name, добавлены `dct_aux_weight: 0.2` и `use_forensics: true`; у родителя эти поля отсутствуют. Нельзя подменять его текущим SegFormer DCT-aux-рецептом: сохранённый decoder_name — `emcad`.

Полный список:

- [pvt_v2_b2_640_letterbox](../../runs/archive/pvt_v2_b2_640_letterbox): нет YAML с таким run_name; параметры доступны в snapshot. Не сопоставлять с похожим названием автоматически.
- [pvt_v2_b2_emcad_640_dct_aux_mixed_original](../../runs/dct_aux): нет YAML с таким run_name; параметры доступны в snapshot. Не сопоставлять с похожим названием автоматически.
- [efficientvit_b2_1024_mixed_original_bn_diagnostic_20260906_140332_037397](../../runs/archive/efficientvit_b2_1024_mixed_original_bn_diagnostic_20260906_140332_037397): диагностика BatchNorm; не обычный тренировочный запуск.
- [validation_protocol_20260908](../../runs/validation_protocol_20260908): аудит данных и сохранённый протокол; не обычный тренировочный запуск.

## Порядок дальнейшего упрощения

В `docs/validation_protocol.md` осталось описание наследования long baseline (18 эпох), а текущий `baseline_protocol.yaml` наследует mixed baseline (8 эпох). Для состояния на дату аудита использовать таблицу выше; историческое описание не является точной инструкцией повторения выполненного запуска.

1. Зафиксировать новую базу из выбранного snapshot с новым именем; сохранить разбиение, бюджет и оценку явно. Для коротких абляций ориентир — 6 × 24 000; длинные 18 × 24 000 учитывать отдельно.
2. Сделать новые варианты с одним уровнем наследования и короткими идентификаторами. Существующие snapshots и папки оставить на месте.
3. Вывести выбор нового запуска / продолжения конкретной папки из рецепта и объединить обучение в один ноутбук. Перед запуском показывать полностью разрешённые параметры и отличие от выбранной базы.
4. Расширить существующий ExperimentRegistry данными snapshot, группой оценки и статусом; отделить служебные папки. Не требовать от ручной карточки дублирования метрик.

Аудит не запускал обучение, inference или holdout и не изменял исторические карточки. Для обновления карты повторно сверить рецепты, snapshots, summary и таблицы строк; этот документ — датированный срез.
