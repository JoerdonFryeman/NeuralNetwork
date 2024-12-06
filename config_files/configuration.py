import os
from json import load
from logging import config, getLogger


def get_json_data(name: str) -> dict:
    """
    Возвращает данные в формате json из указанного файла.
    :param name: Имя файла без расширения.
    :return: Словарь с данными из json файла.
    :raises FileNotFoundError: Если файл не найден.
    """
    try:
        with open(f'config_files/{name}.json', encoding='UTF-8') as file:
            data = load(file)
        return data
    except FileNotFoundError:
        raise FileNotFoundError('Файл не найден!')


def make_directory(directory: str) -> None:
    try:
        os.mkdir(directory)
        print(f'Создан каталог "{directory}".\n')
    except FileExistsError:
        print(f'Найден каталог "{directory}".\n')


make_directory('temporary_files')
config.dictConfig(get_json_data('logging'))
logger = getLogger()
