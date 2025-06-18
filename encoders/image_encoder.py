import os
import numpy as np
from PIL import Image
import json


def encode_image_to_array(file_path: str, invert_colors: bool = False, size: tuple = (7, 7)) -> list:
    """
    Преобразует изображение в массив данных.

    :param file_path: Путь к файлу изображения.
    :param invert_colors: Булевая переменная для инверсии цветов (черный <-> белый).
    :param size: Размер для изменения изображения.

    :return: Нормализованный одномерный список пикселей изображения.
    """
    with Image.open(file_path) as img:
        # Конвертация в градации серого.
        img = img.convert("L")
        # Изменение размера изображения на заданное количество пикселей.
        img = img.resize(size)
        img_array = np.array(img)
        # Нормализация и преобразование в одномерный список.
        normalized_array = (img_array / 255.0).flatten().tolist()
        if invert_colors:
            # Инвертирование значения.
            normalized_array = [1 - pixel for pixel in normalized_array]
        return normalized_array


def encode_images_from_directory(dir_path: str, output_file: str, invert_colors: bool = False, size: tuple = (7, 7)):
    """
    Преобразует изображения из директории и поддиректорий в массив данных и сохраняет в json файл.

    :param dir_path: Путь к главной директории с поддиректориями.
    :param output_file: Путь к выходному файлу json.
    :param invert_colors: Булевая переменная для инверсии цветов (черный <-> белый).
    :param size: Размер для изменения изображения.
    """
    if not os.path.isdir(dir_path):
        raise ValueError(f'Каталог {dir_path} не существует!')
    classes_data = {}
    for sub_dir_name in sorted(os.listdir(dir_path)):
        sub_dir_path = os.path.join(dir_path, sub_dir_name)
        if not os.path.isdir(sub_dir_path):
            continue
        images_data = []
        for file_name in sorted(os.listdir(sub_dir_path)):
            file_path = os.path.join(sub_dir_path, file_name)
            if os.path.isfile(file_path):
                try:
                    layer = encode_image_to_array(file_path, invert_colors, size)
                    images_data.append(layer)
                except Exception as e:
                    print(f'Не удалось обработать {file_name}: {e}')
            else:
                raise FileNotFoundError(f'Ожидаемый файл {file_name} не существует в каталоге {sub_dir_path}!')
        # Добавление дублирующего ключа в конце списка данных.
        classes_data[sub_dir_name] = [images_data, float(sub_dir_name)]
    # Приведение структуры к формату с ключом 'numbers'.
    data = {'classes': classes_data}
    with open(output_file, 'w') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)
