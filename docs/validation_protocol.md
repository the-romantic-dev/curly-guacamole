# Фиксированная валидация

В pipeline `emcad_v1` training всегда использует `runs/validation_protocol_20260908/protocol`: 62 219 примеров соревнования + 25 509 проверенных originals. Development содержит 20 740 + 8 502, holdout — 20 740 + 8 476. Originals имеют нулевые маски и включаются без флага. Путь manifest разрешается от корня проекта; расположение можно изменить через dataset.protocol_path, отключить протокол нельзя.

Manifest не генерируется заново при запуске модели. Его checksums, связи групп и реальные train/development строки сохраняются и проверяются. Новые данные требуют нового manifest и нового запуска.

## Обучение

Выберите любой текущий конфиг, например:

```powershell
& 'D:/Apps/anaconda3/envs/challenges/python.exe' -m src.training --config configs/positive_dice.yaml
```

Development выбирает checkpoint и пороги по combined AIC. Вероятности восстанавливаются до исходного размера до порога; GT оценивается в исходном разрешении. Provided и originals записываются отдельно в `development/metrics.json`, `per_image.parquet`, `slices.csv`. Для групп без positive Dice/AIC равны null; FPR остаётся определённым.

Combined AIC с originals нельзя напрямую сравнивать с историческим AIC без этих негативов. Проверка original-пар автоматическая и консервативная; она не гарантирует обнаружение всех перцептивных дубликатов.

## Окончательный holdout

После выбора единственного кандидата:

```powershell
& 'D:/Apps/anaconda3/envs/challenges/python.exe' -m src.eval --run runs/positive_dice
```

Протокол читается из snapshot. Evaluator проверяет завершение обучения, provenance, контрольные суммы checkpoint, фактические строки train/development и сохранённую рабочую точку. Holdout не подбирает пороги.

Перед оценкой создаётся `holdout_claim.json`; повторная оценка и дальнейшее resume этого запуска запрещены. При аварии marker остаётся: сначала выясните причину. Это защита одного run; сравнение многочисленных кандидатов по holdout по-прежнему нарушает независимость выбора модели.

Исторический аудит создания протокола остаётся в `runs/validation_protocol_20260908`, старые модели — в `runs/archive`. Старые snapshots воспроизводятся прежним кодом на main, без подмены preprocessing.
