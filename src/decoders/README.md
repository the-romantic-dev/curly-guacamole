# Декодеры

Локальный пакет с API в духе timm. Встроены `unet`, `segformer` и `emcad`;
предобученных весов декодеров нет.

```python
from src.decoders import create_decoder, list_decoders, is_decoder

list_decoders()       # ['emcad', 'segformer', 'unet']
list_decoders('seg*') # ['segformer']

decoder = create_decoder(
    'segformer',
    encoder_channels=[64, 128, 320, 512],
    encoder_strides=[4, 8, 16, 32],
    embed_dim=128,
    norm='batch',
    use_aux=True,
)
features, aux_logits = decoder(encoder_features)
```

## Конфигурация эксперимента

```yaml
model:
  encoder_name: pvt_v2_b2
  decoder_name: segformer
  decoder_kwargs:
    embed_dim: 128
  forensic_channels: [64, 96, 128]
  aux_weight: 0.4
  norm: batch
```

Для U-Net:

```yaml
  decoder_name: unet
  decoder_kwargs:
    decoder_channels: [128, 64, 32, 16, 16]
    aux_stage: 2
```

`decoder_kwargs` передаются конструктору декодера. Неизвестный параметр
вызывает ошибку, а не игнорируется. Метаданные энкодера, `norm` и `use_aux`
передаёт Segmenter; переопределять их через `decoder_kwargs` нельзя.
Все параметры сохраняются в snapshot и используются при инференсе.

Старые `decoder_channels` и `decoder_embed_dim` в model и аргументах Segmenter
поддерживаются; значения из `decoder_kwargs` имеют приоритет. Без
`decoder_name` выбирается U-Net. Ключи state_dict существующих моделей
сохранены; старые импорты из `src.modules` также работают.

## Добавление декодера

1. Создать модуль, например `src/decoders/my_decoder.py`.
2. Унаследовать класс от `Decoder` и добавить `@register_decoder('my_decoder')`.
3. Импортировать модуль в `src/decoders/__init__.py` для регистрации во всех
   процессах обучения и инференса. Для внешнего плагина достаточно импортировать
   его до вызова `create_decoder` в каждом процессе.

Конструктор принимает `encoder_channels`, `encoder_strides`, `norm`, `use_aux`
и собственные именованные параметры. Не добавляйте поглощающий `**kwargs`,
чтобы опечатки в конфигурации обнаруживались.

Контракт `Decoder`:

- Вход: список NCHW-признаков от мелкого stride к крупному.
- `forward`: пара `(features, aux_logits)`, где aux может быть `None`.
- `out_channels`: число каналов выходных признаков.
- `output_stride`: шаг выходных признаков относительно изображения.
- `head_kernel_size`: нечётный размер ядра внешней mask head; по умолчанию 1.

Segmenter самостоятельно строит mask head, увеличивает logits до размера
изображения и обрабатывает classification head. Новому декодеру не требуется
менять Segmenter, builders, конфигурационную схему или код инференса.

Встроенный SegFormer — адаптация: линейные 1×1-проекции всех масштабов,
bilinear resize к самому подробному масштабу, concat и 1×1 fusion с norm/ReLU.
Дополнительная голова обучается на самой подробной проекции до fusion.
U-Net сохраняет последовательный upsampling и skip connections baseline.

## Эксперимент EMCAD

Конфигурация: `configs/emcad_mixed_original.yaml`, запуск:
`notebooks/emcad_mixed_original.ipynb`.

```yaml
  decoder_name: emcad
  decoder_kwargs:
    kernel_sizes: [1, 3, 5]
    expansion_factor: 2
    lgag_kernel_size: 3
    activation: relu
```

Архитектура из [EMCAD, CVPR 2024](https://arxiv.org/abs/2405.06880):
channel/spatial attention, параллельные depthwise-свёртки с суммированием,
channel shuffle, residual, efficient upsampling и grouped attention gates
на skip connections. Spatial attention разделяет веса между четырьмя стадиями.
Требуются четыре уровня с шагами `[4, 8, 16, 32]` и чётным числом каналов.

Адаптация сохраняет каналы энкодера и выдаёт признаки на шаге 4.
Auxiliary head с весом 0.4 получает объединённые признаки перед последним
refinement. Используется одна основная mask head, существующая classification
head и loss проекта. Multi-head supervision авторов не воспроизводится.
Инициализация — стандартная PyTorch, upsampling явно задаёт размер skip,
что поддерживает нечётные размеры признаков. Это эксперимент с архитектурой
EMCAD в baseline, а не воспроизведение результатов статьи.

Для сверки архитектуры использован
[репозиторий авторов](https://github.com/SLDGroup/EMCAD).
Его исходный код распространяется под UT Austin Research License,
а не MIT/Apache; здесь нет зависимости от установки этого репозитория
или загрузки его обученных моделей.
