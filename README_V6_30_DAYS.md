# Dota Draft AI v6 — фиксированное окно 30 дней

Проект всегда использует только матчи за последние 30 дней.

## Сбор матчей

```bash
python -m src.collect --pages 300 --sleep 2
```

## Позиции героев

```bash
python -m src.collect_positions --matches 500 --sleep 2
```

## Обучение

```bash
python -m src.train_nn
```

## Запуск интерфейса

```bash
python -m streamlit run src/recommend_app.py
```

Параметр `--hours` больше не нужен.
