# Dota Draft AI v2 — нейросеть и рекомендация пятого героя

## Что добавлено

Эта версия решает задачу:

```text
4 своих героя
+
5 героев противника
↓
нейросеть проверяет всех доступных героев
↓
Top-N лучших вариантов для пятого пика
```

## 1. Установить новые зависимости

В корне проекта:

```bash
.venv\Scripts\activate
python -m pip install -r requirements.txt
```

## 2. Собрать больше матчей

Для нейросети 2-20 тысяч матчей слишком мало.

Минимум для эксперимента:

```bash
python -m src.collect --pages 500 --sleep 1
```

Желательно стремиться к 50 000-100 000+ качественных матчей.

Важно: текущий `collect.py` перезаписывает `data/matches.csv`.
Если хочешь собирать миллионы матчей частями, следующим шагом нужно добавить append/resume и БД.

## 3. Обучить нейросеть

```bash
python -m src.train_nn
```

Создаст:

```text
models/draft_nn.pt
models/draft_nn_meta.pt
models/nn_metrics.json
```

Во время обучения увидишь примерно:

```text
epoch=01 train_loss=...
test_logloss=...
auc=...
acc=...
```

Смотри прежде всего на `test_logloss` и `roc_auc`, а не только accuracy.

## 4. Запустить интерфейс подбора пятого героя

```bash
python -m streamlit run src/recommend_app.py
```

Обычно откроется:

```text
http://localhost:8501
```

## Как работает нейросеть

На вход получает 10 hero ID:

```text
Radiant: 5
Dire:    5
```

Каждый герой превращается в embedding-вектор.

Дополнительно сеть получает 25 взаимодействий:

```text
Radiant hero 1 x Dire hero 1
Radiant hero 1 x Dire hero 2
...
Radiant hero 5 x Dire hero 5
```

Это позволяет сети учить не только индивидуальную силу героя,
но и matchup-зависимые эффекты.

## Как выбирается пятый герой

Например:

```text
Radiant:
A
B
C
D
?

Dire:
E
F
G
H
I
```

Recommender делает:

```text
A B C D Anti-Mage vs E F G H I -> 53.1%
A B C D Axe       vs E F G H I -> 49.4%
A B C D Puck      vs E F G H I -> 57.8%
...
```

и сортирует кандидатов по прогнозу.

## Очень важное ограничение

Текущая сеть пока НЕ знает:

- позиции 1-5
- текущий patch
- MMR
- порядок пиков
- bans
- lane matchups
- facet
- aspect
- item builds
- player skill

Поэтому это именно техническая v2, а не готовый esports-grade draft assistant.

Следующее улучшение должно быть не "сделать сеть глубже", а добавить качественные признаки:
patch, role, MMR и pick order.
