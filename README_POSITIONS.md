# Dota Draft AI v3 — реальные позиции героев

Теперь роли не нужно заполнять вручную.

## Запуск в PyCharm

Открой вкладку Terminal.

### 1. Обновить список свежих матчей

```bash
python -m src.collect --pages 50 --sleep 1
```

### 2. Собрать статистику реальных позиций

Быстрый тест:

```bash
python -m src.collect_positions --matches 100 --sleep 1.1
```

Нормальнее:

```bash
python -m src.collect_positions --matches 500 --sleep 1.1
```

Создаст:

```text
data/hero_position_stats.json
```

Подробные матчи кэшируются в `data/match_details/`, поэтому повторный запуск не скачивает их заново.

### 3. Запустить Draft AI

```bash
python -m streamlit run src/recommend_app.py
```

## Как определяются позиции

Используются реальные parsed matches и поля `lane_role`, `is_roaming`,
`net_worth`, `gold_per_min`, `last_hits`.

Эвристика:

- mid -> pos 2;
- фармящий safelane -> pos 1;
- второй safelane -> pos 5;
- фармящий offlane -> pos 3;
- второй offlane / roamer -> pos 4.

По умолчанию герой допускается на позицию, если есть хотя бы 3 наблюдения и
на эту позицию приходится не менее 15% его наблюдаемых игр.

Пример изменения порогов:

```bash
python -m src.collect_positions --matches 500 --min-games 5 --min-share 0.10
```

## Важно

Это автоматическое определение метовых позиций, но всё ещё эвристика.
Следующий этап — записать позиции непосредственно в train dataset и обучать
нейросеть на `hero embedding + position embedding`.
