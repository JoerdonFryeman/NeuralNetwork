# Neural Network
Проект представляет собой комплексную систему для создания и обучения нейронных сетей. Он включает в себя модули для конфигурации, обработки данных, определения структуры сети, выполнения машинного обучения, визуализации и тестирования.

---

## Структура проекта

- `main.py`: Основной модуль для запуска процесса создания, обучения и настройки сети.
- `configuration.py`: Модуль для загрузки конфигурационных данных и настройки логирования.
- `data.py`: Модуль для работы с набором данных для моделей машинного обучения.
- `layers.py`: Модуль для работы с различными слоями нейронной сети.
- `neural_network.py`: Модуль, реализующий основной класс для работы с нейронной сетью.
- `machine_learning.py`: Модуль для выполнения основного процесса машинного обучения.
- `support_functions.py`: Модуль, содержащий вспомогательные функции для нейронной сети.
- `visualisation.py`: Модуль для визуализации результатов работы нейронной сети.
- `tests.py`: Модуль для тестирования различных частей проекта.
- `image_encoder.py`: Кодирующий изображения в массивы данных скрипт.
- `logging.json`: Файл конфигурации логирования.
- `numbers.json`: Файл графического представления цифр.

---

## Установка

### Требования

- Python 3.12
- numpy 2.1
- pillow 11
- Имя класса (каталога) данных не должно равняться нулю!

Для запуска процесса установки выполните следующую команду:
```console
cd encoders && python3 -m venv venv && source venv/bin/activate && pip install -r requirements.txt && cd ../ && python main.py
```

## Запуск

Для запуска основного процесса выполните:
```console
cd encoders && source venv/bin/activate && cd ../ && python main.py
```
Если используете свой скрипт кодирования:
```console
python main.py
```
## Настройка

### Параметры обучения

- `init_func` (default: `xavier`): Выбор метода инициализации весов: `uniform`, `xavier`, `he`
- `epochs` (default: 1000): Количество эпох для обучения.
- `learning_rate` (default: 0.001): Скорость обучения.
- `error_tolerance` (default: 0.001): Допустимый уровень ошибки.
- `regularization` (default: 0.001): Параметр регуляризации.
- `lasso_regularization` (default: False): Использовать Lasso регуляризацию.
- `ridge_regularization` (default: True): Использовать Ridge регуляризацию.

---

## Описание модулей

### main.py
Основной модуль для запуска процесса создания, обучения и настройки сети.

### configuration.py
Модуль для загрузки конфигурационных данных из JSON-файлов и настройки логирования.

### data.py
Модуль для работы с набором данных, таких как изображения игрального кубика или другие рукописные цифры.

### layers.py
Модуль для работы с различными слоями нейронной сети, включая внутренние и внешние слои.

### neural_network.py
Основной класс для создания и обучения нейронной сети.

### machine_learning.py
Модуль для выполнения основного процесса машинного обучения, включая тренировку модели и обновление весов.

### support_functions.py
Модуль, содержащий функции активации и инициализации для нейронной сети.

### visualisation.py
Модуль для визуализации результатов работы нейронной сети.

### image_encoder.py
Вспомогательный модуль кодирования изображений в числовые массивы.
Для его отключения в модуле configuration.py флаг use_image_encoder необходимо сменить на False.

### tests.py
Модуль, содержащий тесты для проверки корректности работы различных частей проекта.

---

## Запуск тестов

Для запуска всех тестов выполните:
```bash
python -m unittest discover
```

---

## Лицензия

Этот проект разрабатывается под лицензией MIT.

---

## Поддержать с помощью Биткоина:

bc1qewfgtrrg2gqgtvzl5d2pr9pte685pp5n3g6scy
