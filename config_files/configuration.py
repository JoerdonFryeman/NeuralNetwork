import os
import json
from platform import system
from logging import config, getLogger

directories: tuple[str, str, str, str, str, str, str, str, str] = (
    'weights_biases_and_data', 'temporary_files', 'tools', 'network', 'machine_learning',
    'learning_data', 'learning_data', 'data', 'config_files'
)
for d in directories:
    try:
        os.mkdir(d)
    except FileExistsError:
        pass

use_image_encoder: bool = True
directory_path: str = 'numbers'
invert_colors: bool = False
image_size: tuple[int, int] = (28, 28)

if use_image_encoder:
    # Запускает встроенный скрипт для преобразования изображений в числовые массивы.
    from encoders.image_encoder import encode_images_from_directory

    encode_images_from_directory(
        f'learning_data/{directory_path}', 'weights_biases_and_data/input_dataset.json', invert_colors, image_size
    )


def get_json_data(directory: str, name: str) -> dict:
    """
    Возвращает данные в формате json из указанного файла.

    :param directory: Название каталога.
    :param name: Имя файла без расширения.

    :return: Словарь с данными из json файла.
    :raises FileNotFoundError: Если файл не найден.
    """
    try:
        with open(f'{directory}/{name}.json', encoding='UTF-8') as json_file:
            data = json.load(json_file)
        return data
    except FileNotFoundError:
        raise FileNotFoundError('Файл не найден!')
    except json.JSONDecodeError:
        raise ValueError(f'Ошибка декодирования JSON в файле: {name}')


def save_json_data(directory: str, name: str, data: dict[str, list[float]]) -> None:
    """
    Сохраняет файл json.

    :param directory: Директория сохраняемого файла.
    :param name: Имя сохраняемого файла.
    :param data: Данные сохраняемого файла.
    """
    with open(f'{directory}/{name}.json', 'w') as json_file:
        json.dump(data, json_file, indent=4)


def select_os_command(command: str):
    """
    Возвращает необходимую команду в зависимости от операционной системы.

    :param command: Необходимая команда в строковом виде.
    :return: Возвращает системную команду.
    """
    if command == 'clear_screen':
        return os.system({'Linux': lambda: 'clear', 'Windows': lambda: 'cls'}[system()]())
    return None


config.dictConfig(get_json_data('config_files', 'logging'))
logger = getLogger()
