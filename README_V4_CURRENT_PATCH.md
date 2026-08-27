# Dota Draft AI v4 — без Radiant/Dire + только текущий патч

## Что изменилось

- В интерфейсе больше нет выбора **Radiant / Dire**.
- Есть только **Моя команда (4 героя)** и **Противник (5 героев)**.
- Для каждого кандидата модель считает прогноз в обеих ориентациях карты и усредняет результат. Это снижает искусственный Radiant/Dire bias.
- `collect.py` автоматически определяет начало самого свежего патча и сохраняет только матчи после его выхода.
- `train_nn.py` повторно проверяет дату патча перед обучением, поэтому старые матчи случайно не попадут в модель.
- `collect_positions.py` тоже использует только матчи текущего патча.

## Правильный порядок запуска в PyCharm Terminal

```bash
python -m src.collect --pages 100 --sleep 1
```

Проверь созданный файл:

```text
data/current_patch.json
```

Затем собери роли/позиции:

```bash
python -m src.collect_positions --matches 500 --sleep 1.1
```

Обучи нейросеть заново:

```bash
python -m src.train_nn
```

Запусти интерфейс:

```bash
python -m streamlit run src/recommend_app.py
```

## Важно после выхода нового патча

Старую модель использовать не надо. Просто снова выполни последовательность:

```bash
python -m src.collect --pages 100 --sleep 1
python -m src.collect_positions --matches 500 --sleep 1.1
python -m src.train_nn
python -m streamlit run src/recommend_app.py
```

`matches.csv` перезапишется матчами нового патча, а нейросеть будет обучена заново.

## Если OpenDota не сможет автоматически определить дату патча

Можно явно указать начало патча Unix timestamp через переменную окружения `DOTA_PATCH_START`. В обычной ситуации этого делать не нужно.
