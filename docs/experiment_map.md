# Актуальная EMCAD-линейка

Ветка `codex/emcad-baseline`, pipeline `emcad_v1`. Во всех рецептах встроены mixed-frame обучение, natural DCT, фиксированный протокол, originals в train и original-resolution validation. Общая база: PVT-v2-B2 + EMCAD, 640, шесть эпох. Короткие имена используются и для конфигов, и для сохранённых EMCAD-запусков. Настройки выполненного обучения определяет snapshot: старые результаты не обязательно получены на текущей версии рецепта.

| Новый конфиг | Предшествующий рецепт / идея | Изменение |
|---|---|---|
| [baseline](../configs/baseline.yaml) | emcad_mixed_original_fixed_val_natural_q_light | Общие условия стали поведением pipeline |
| [dct576](../configs/dct576.yaml) | baseline | RGB 576, текущая forensic-ветка |
| [rgb576](../configs/rgb576.yaml) | dct576 | Forensic-ветка отключена |
| [rgb576_no_aux](../configs/rgb576_no_aux.yaml) | rgb576 | Aux-голова отключена: вес 0 вместо 0.4; остальные настройки наследуются. Сравнивать validation AIC, Dice_pos и FPR_neg |
| [rgb576_strided](../configs/rgb576_strided.yaml) | rgb576 | RGB1152 → обучаемая Conv3×3 stride 2 → RGB576 → ImageNet-нормализация; [описание](strided_resize.md) |
| [rgb576_strided_gn](../configs/rgb576_strided_gn.yaml) | rgb576_strided | GroupNorm вместо BatchNorm в EMCAD; отдельное обучение |
| [rgb576_strided_mlp](../configs/rgb576_strided_mlp.yaml) | rgb576_strided | Ресайзер Conv3×3 3→16 → GELU → Conv3×3 16→3 stride 2; BatchNorm |
| [jpeg576](../configs/jpeg576.yaml) | dct576 | Native JPEG Artifact Module + компактная пирамида; [условия сравнения](jpeg_ablation.md) |
| [jpeg576_pretrained](../configs/jpeg576_pretrained.yaml) | jpeg576 | Инициализация JPEG Artifact Module из `DCT_djpeg.pth`, с последующим дообучением |
| [positive_dice](../configs/positive_dice.yaml) | baseline_positive_dice_mixed_original; последующий EMCAD positive-Dice | Dice только по позитивам, вес 1 |
| [local](../configs/local.yaml) | emcad_local_1024_positive_dice_light | Локальная ветка 1024 поверх positive_dice |
| [luma](../configs/luma.yaml) | Диагностика потери яркостных деталей при resize | GPU-экстрактор яркости 1024, 8/16/24 каналов, одно слияние после EMCAD; качество пока не измерено |
| [stride4](../configs/stride4.yaml) | emcad_stride4_mixed_original | Refinement 144 каналов |
| [stride2_rgb](../configs/stride2_rgb.yaml) | emcad_stride2_rgb_mixed_original | RGB refinement 32/24, batch 2 × accum 8 |
| [dct_aux](../configs/dct_aux.yaml) | pvt_v2_b2_emcad_640_dct_aux_mixed_original | DCT auxiliary weight 0.2 |

`stride4`, `stride2_rgb` и `dct_aux` перенесены на новую общую базу; старые scores не являются результатами этих новых рецептов. Для нового варианта копируйте только отличия, задавайте новое имя; длительность обучения — обычный параметр, отдельные light/long-конфиги не нужны.

## Архив

7 выполненных EMCAD-запусков находятся в [runs](../runs); остальные 13 исторических папок — в [runs/archive](../runs/archive). EMCAD-папки переименованы по таблице ниже. Исходные IDs и настройки сохранены в snapshots; checkpoints, метрики и логи не изменены. Включён новый загруженный `emcad_mixed_original_fixed_val_natural_q_light_pos_dice`. Ссылки run/parent/baseline в notes обновлены под новые имена.

Удалённые активные рецепты и ноутбуки UNet/SegFormer/EfficientViT остаются в Git на `main`; для получения конкретного рецепта: `git show main:configs/<name>.yaml`. Подробный прежний каталог сохранён как [исторический аудит](archive/experiment_map_20260909.md).

Immutable manifest и аудит его создания остаются в `runs/validation_protocol_20260908`; manifest не пересоздавался. Реестр активных экспериментов пропускает архив и служебную папку протокола. Для остальных исторических результатов используйте `ExperimentRegistry('runs/archive')`.

## Имена сохранённых EMCAD-запусков

| Исходное имя | Папка |
|---|---|
| `emcad_mixed_original_fixed_val_natural_q_light` | [baseline](../runs/baseline) |
| `emcad_mixed_original_fixed_val_natural_q_light_pos_dice` | [positive_dice](../runs/positive_dice) |
| `pvt_v2_b2_emcad_640_dct_aux_mixed_original` | [dct_aux](../runs/dct_aux) |
| `pvt_v2_b2_emcad_640_stride2_rgb_mixed_original` | [stride2_rgb](../runs/stride2_rgb) |
| `pvt_v2_b2_emcad_640_stride4_mixed_original` | [stride4](../runs/stride4) |
| `pvt_v2_b2_emcad_mixed_original` | [baseline_legacy](../runs/baseline_legacy) |
| `pvt_v2_b2_emcad_mixed_original_fixed_val_light` | [baseline_legacy_dct](../runs/baseline_legacy_dct) |

`baseline_legacy` — ранний baseline; `baseline_legacy_dct` — версия с исправленной валидацией до исправления DCT. Для `local` выполненного запуска пока нет.
