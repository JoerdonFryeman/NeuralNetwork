# Нейронная сеть

Проект представляет собой комплексную систему для создания и обучения нейронной сети. Он включает в себя модули для конфигурации, обработки данных, определения структуры сети, выполнения машинного обучения, визуализации и тестирования.

## Структура проекта

- `main.py`: Основной модуль для запуска процесса создания, обучения и настройки сети.
- `config_files/configuration.py`: Модуль для загрузки конфигурационных данных и настройки логирования.
- `config_files/logging.json`: Файл конфигурации логирования.
- `config_files/numbers.json`: Файл графического представления цифр.
- `data/classification.py`: Модуль для классификации выходных данных сети.
- `data/data.py`: Модуль для работы с набором данных для моделей машинного обучения.
- `encoders/image_encoder.py`: Кодирующий изображения в массивы данных скрипт.
- `learning_data/dataset.zip`: Небольшой массив данных для тестового обучения.
- `machine_learning/calculations.py`: Модуль для связанных с вычислениями методов.
- `machine_learning/regularization.py`: Модуль для методов регуляризации.
- `machine_learning/train.py`: Модуль для выполнения основного процесса машинного обучения.
- `machine_learning/weights.py`: Модуль для обновления и сохранения весов.
- `network/layers.py`: Модуль для создания слоёв нейронной сети.
- `network/neural_network.py`: Реализующий основной класс для работы с нейронной сетью модуль.
- `tools/support_functions.py`: Содержащий вспомогательные функции модуль.
- `tools/visualisation.py`: Модуль для визуализации результатов работы нейронной сети.
- `tests/tests.py`: Модуль для тестирования различных частей проекта.

## Требования

- python: >= 3.12
- numpy: >= 2.1
- pillow: >= 11.2
- Ключ класса (название каталога) данных не должен равняться нулю!

## Установка

## Для Linux

Для запуска процесса установки выполните следующую команду:
```console
cd encoders && python3 -m venv venv && source venv/bin/activate && pip install --upgrade pip && pip install -r requirements.txt && cd ../ && python3 main.py
```

## Запуск

## Для Linux

Для запуска основного процесса выполните:
```console
cd encoders && source venv/bin/activate && cd ../ && python3 main.py
```
Без скрипта кодирования выполните:
```console
python3 main.py
```

## Настройка

Параметры обучения

- `init_func` (default: xavier): Выбор метода инициализации весов: `uniform`, `xavier`, `he`
- `epochs` (default: 1000): Количество эпох для обучения.
- `learning_rate` (default: 0.001): Скорость обучения.
- `error_tolerance` (default: 0.001): Допустимый уровень ошибки.
- `regularization` (default: 0.001): Параметр регуляризации.
- `lasso_regularization` (default: False): Использовать Lasso регуляризацию.
- `ridge_regularization` (default: True): Использовать Ridge регуляризацию.

## Лицензия

Этот проект разрабатывается под лицензией MIT.

## Поддержать с помощью Биткоина:

bc1qewfgtrrg2gqgtvzl5d2pr9pte685pp5n3g6scy
