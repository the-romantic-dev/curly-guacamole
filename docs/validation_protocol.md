# Запуски с независимым holdout

Все команды выполняются из `D:\Projects\aiijc2026_final`, окружение `challenges`.

Сохранённый протокол: `runs/validation_protocol_20260908/protocol`. Не пересоздавайте его для каждой модели. `protocol.json` содержит контрольные суммы таблиц; изменение данных требует нового протокола и новых запусков.

| Часть | Примеры соревнования | Проверенные уникальные исходники |
|---|---:|---:|
| Train | 62 219 | 25 509, включаются через train_originals |
| Development | 20 740 | 8 502, включены |
| Holdout | 20 740 | 8 476, включены |

Исходники считаются отрицательными с нулевой маской. Их включение меняет состав AIC: новый combined score нельзя напрямую сравнивать со старыми validation scores или считать прогнозом LB. Метрики provided и originals сохраняются отдельно. Проверка пар консервативная и автоматическая; она не доказывает происхождение данных и не исключает все перцептивные дубликаты.

## Обучение

Первый контролируемый запуск — прежний состав train, но новая схема оценки:

```powershell
& 'D:/Apps/anaconda3/envs/challenges/python.exe' -m src.training --config configs/baseline_protocol.yaml
```

Второй — те же настройки и бюджет, с исходниками в train:

```powershell
& 'D:/Apps/anaconda3/envs/challenges/python.exe' -m src.training --config configs/baseline_protocol_originals.yaml
```

Запускайте последовательно. Новые конфиги наследуют long baseline (18 эпох). В этой задаче обучение не запускалось. Каждый эксперимент получает уникальный run_name. Для продолжения прерванного запуска установите resume: true, сохранив протокол, состав train и остальные настройки. Не продолжайте старые checkpoints на новом разбиении: они могли видеть holdout.

На каждой эпохе только development выбирает пороги и лучший checkpoint. `summary.json` явно помечает `evaluation_role: development`. Исторические ключи val/aic_tuned и best_aic сохранены для совместимости, но относятся к development, не к holdout.

После улучшения сохраняются `development/metrics.json`, `development/per_image.parquet`, `development/slices.csv`: Dice, FP/FPR, показатели без gate, срезы по источнику, JPEG и площади. `combined` — критерий выбора; `provided` — только поставленные соревнованием примеры; `originals` — отдельный FPR исходников. Для наборов без положительных примеров AIC и Dice равны null, а не искусственному нулю.

## Окончательная оценка

Сначала выберите кандидата по development. Затем один раз выполните:

```powershell
& 'D:/Apps/anaconda3/envs/challenges/python.exe' -m src.eval --run runs/pvt_v2_b2_protocol_originals --protocol runs/validation_protocol_20260908/protocol --batch-size 4 --workers 4
```

Команда читает best.pt и сохранённые при выборе checkpoint пороги; ничего не подбирает. Она проверяет завершение обучения, совпадение протокола и фактических train/development строк. Результаты — `holdout/metrics.json`, `holdout/slices.csv`, `holdout/per_image.parquet` и `summary.json`.

Перед чтением holdout создаётся `holdout_claim.json`: повторный запуск и дальнейший resume этого обучения запрещены. Если процесс аварийно завершился, сохраняйте marker и выясняйте причину перед ручным восстановлением: автоматический повтор с другим checkpoint недопустим. Ограничение действует на один run; дисциплина выбора кандидатов между разными run остаётся ответственностью исследователя. Не выбирайте архитектуры по множеству holdout-отчётов.

Для новой модели создайте YAML, наследующий baseline_protocol.yaml, задайте уникальный run_name и нужные изменения model. Для модели с исходниками наследуйте baseline_protocol_originals.yaml. Остальные параметры сравнения сохраняйте одинаковыми.

## Исторический аудит

`runs/validation_protocol_20260908` содержит исходники, решения по каждой паре, исключённые связи с историческим train и результаты четырёх старых моделей с их прежними порогами. Это диагностические результаты, а не независимый holdout нового протокола. Competition test не использовался.
