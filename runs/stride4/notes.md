---
run: stride4
series: pvt
parent: baseline_legacy
baseline: baseline_legacy
verdict: inconclusive
source: configs/emcad_stride4_mixed_original.yaml
change: "Две dense 3x3 свёртки 64→144→64 с residual после последней стадии EMCAD."
leaderboard_score: 0.9111572066728966
---

**Что проверяем.** Две dense 3x3 свёртки 64→144→64 с residual после последней стадии EMCAD. Вложить свободные FLOPS в точность локализации.

**Протокол.** EMCAD 640 mixed/original, DCT включён, DCT aux выключен,
seed 42, fold 0, 8 × 24 000 показов, последние 2 эпохи full-frame.
Обучение с нуля; parent задаёт происхождение эксперимента, не загрузку весов.

**Вычисления.** Полная модель: 98.058314 GFLOPS, batch 1, eval, 640×640,
FlopCounterMode; совпадение meta и реального CPU forward с SDPA math.
H100 latency пока не измерена. CPU fused SDPA даёт меньший счётчик;
для сравнения используем полный math/meta подсчёт, как у исторического EMCAD (89.564848).

**Статус.** Подготовлен, обучение ещё не запускалось. Метрики отсутствуют.
Сравнивать AIC, Dice по площади GT (особенно <1% и 1–3%) и FPR negatives.
