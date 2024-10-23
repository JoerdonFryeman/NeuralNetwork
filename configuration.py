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
        with open(f'{name}.json', encoding='UTF-8') as file:
            data = load(file)
        return data
    except FileNotFoundError:
        raise FileNotFoundError('File not found!')


config.dictConfig(get_json_data('logging'))
logger = getLogger()
