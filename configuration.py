import os
import json
from platform import system
from logging import config, getLogger


use_image_encoder: bool = True
directory_path: str = 'learning_data/numbers/'
output_file: str = 'weights_biases_and_data/encoded_data.json'
invert_colors: bool = False
image_size: tuple[int, int] = (28, 28)


try:
    directories: tuple[str, str, str, str, str, str, str] = (
        'learning_data', 'learning_data/numbers/', 'weights_biases_and_data',
        'temporary_files', 'config_files', 'encoders', 'tests'
    )
    # Создаёт необходимый каталог в случае его отсутствия.
    for i in range(len(directories)):
        os.mkdir(directories[i])
        i += 1
except FileExistsError:
    pass


if use_image_encoder:
    # Запускает встроенный скрипт для преобразования изображений в числовые массивы.
    from encoders.image_encoder import encode_images_from_directory

    encode_images_from_directory(directory_path, output_file, invert_colors, image_size)


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


def save_json_data(directory: str, name: str, data):
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


config.dictConfig(get_json_data('config_files', 'logging'))
logger = getLogger()
