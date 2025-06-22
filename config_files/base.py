import os
import json
from platform import system
from logging import config, getLogger

# Создание необходимых директорий в случае их отсутствия.

directories: tuple[str, str, str, str, str, str, str, str, str, str, str, str] = (
    'weights_biases_and_data', 'tools', 'tests', 'temporary_files', 'network', 'machine_learning',
    'learning_data', 'games', 'encoders', 'data', 'config_files', 'config_files/ascii_arts'
)
for d in directories:
    try:
        os.mkdir(d)
    except FileExistsError:
        pass


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


def get_method_info(cls, key: str):
    """
    Логирует информацию о методе класса, включая его имя и документацию.

    :param cls: Класс, вызываемого метода.
    :param key: Ключ, соответствующий методу в словаре класса.
    :return: None. Метод не возвращает значения, а только выполняет логирование.
    """
    logger.info(
        f'Метод {cls.__dict__[key].__name__} класса {cls.__name__}\n{cls.__dict__[key].__doc__}'
    )


config.dictConfig(get_json_data('config_files', 'logging'))
logger = getLogger()
