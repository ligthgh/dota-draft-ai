# Dota Draft AI v7

Добавлены признаки:

- hero_id;
- позиция 1–5 каждого героя;
- фактическая роль через определённую позицию;
- порядок пиков 1–10;
- средний MMR матча;
- MMR/rank bracket;
- 25 взаимодействий герой против героя.

## Порядок запуска

### 1. Собрать матчи последних 30 дней

```bash
python -m src.collect --pages 300 --sleep 2
```

### 2. Собрать подробности матчей

Для проверки:

```bash
python -m src.collect_detailed --matches 500 --sleep 2
```

Для нормального обучения желательно:

```bash
python -m src.collect_detailed --matches 10000 --sleep 2
```

Повторные запросы к уже скачанным матчам берутся из `data/match_details`.

### 3. Обучить новую нейросеть

```bash
python -m src.train_nn_v7
```

Результаты:

```text
models/draft_nn_v7.pt
models/draft_nn_v7_meta.pt
models/nn_v7_metrics.json
```

## Что означает роль

В v7 `position 1–5` является фактической игровой ролью, определённой из
lane_role, roaming и farm priority.

## Порядок пиков

Берётся из `picks_bans`.

Если OpenDota не вернул draft sequence для конкретного матча,
pick_order = 0, что означает unknown.

### MMR bracket

```text
0 unknown
1 < 1000
2 1000–1999
3 2000–2999
4 3000–3999
5 4000–4999
6 5000–5999
7 6000–6999
8 7000+
```

Сравни новую модель со старой по `test_logloss` и `roc_auc`.
