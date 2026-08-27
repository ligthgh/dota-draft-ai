# Dota Draft AI v5 — только самые свежие матчи

В этой версии полностью убрана попытка определять текущий patch.

Вместо этого модель использует только матчи из заданного временного окна.

## Почему так лучше

- нет запроса к несуществующему `/constants/patches`;
- не нужно делать отдельный `/matches/{id}` запрос для каждого матча;
- намного меньше риск получить HTTP 429;
- после нового патча старые матчи автоматически исчезают из выборки по мере истечения окна.

## Рекомендуемый запуск

Для максимально свежей модели:

```bash
python -m src.collect --hours 24 --pages 100 --sleep 1.5
```

Это означает:

```text
брать только матчи за последние 24 часа
```

Если данных мало:

```bash
python -m src.collect --hours 48 --pages 200 --sleep 1.5
```

или:

```bash
python -m src.collect --hours 72 --pages 300 --sleep 1.5
```

## Важно про pages

`--hours` задаёт допустимый возраст матча.

`--pages` задаёт максимум страниц OpenDota, которые программа может просмотреть.

Сборщик автоматически остановится, когда дойдёт до матчей старше выбранного окна.

## После сбора

Для ролей:

```bash
python -m src.collect_positions --matches 300 --sleep 1.5
```

Затем обучение:

```bash
python -m src.train_nn
```

Затем приложение:

```bash
python -m streamlit run src/recommend_app.py
```

## Если OpenDota отвечает 429

Сборщик `publicMatches` теперь автоматически ждёт и повторяет запрос.

Можно также снизить нагрузку:

```bash
python -m src.collect --hours 24 --pages 50 --sleep 3
```

Для `collect_positions` всё ещё требуются подробные запросы `/matches/{id}`,
поэтому этот этап делай на ограниченном числе матчей, например 100-300.

## Практический режим

Во время разработки:

```bash
python -m src.collect --hours 24 --pages 20 --sleep 2
python -m src.collect_positions --matches 50 --sleep 2
python -m src.train_nn
python -m streamlit run src/recommend_app.py
```

Для более серьёзной модели:

```bash
python -m src.collect --hours 72 --pages 300 --sleep 1.5
python -m src.collect_positions --matches 300 --sleep 1.5
python -m src.train_nn
```
