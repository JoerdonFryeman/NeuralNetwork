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
        img = img.convert("L")  # Конвертация в градации серого
        img = img.resize(size)  # Изменение размера изображения на заданное количество пикселей
        img_array = np.array(img)
        normalized_array = (img_array / 255.0).flatten().tolist()  # Нормализация и преобразование в одномерный список
        if invert_colors:
            # Инвертирование значения
            normalized_array = [1 - pixel for pixel in normalized_array]
        return normalized_array


def encode_images_from_directory(dir_path: str, output_file: str, invert_colors: bool = False, size: tuple = (7, 7)):
    """
    Преобразует изображения из директории и поддиректорий в массив данных и сохраняет в JSON файл.

    :param dir_path: Путь к главной директории с поддиректориями.
    :param output_file: Путь к выходному файлу JSON.
    :param invert_colors: Булевая переменная для инверсии цветов (черный <-> белый).
    :param size: Размер для изменения изображения.
    """
    if not os.path.isdir(dir_path):
        raise ValueError(f"Directory {dir_path} does not exist")
    numbers_data = {}
    for sub_dir_name in sorted(os.listdir(dir_path)):
        sub_dir_path = os.path.join(dir_path, sub_dir_name)
        if not os.path.isdir(sub_dir_path):
            continue
        images_data = []
        for idx, file_name in enumerate(sorted(os.listdir(sub_dir_path))):
            file_path = os.path.join(sub_dir_path, file_name)
            if os.path.isfile(file_path):
                try:
                    layer = encode_image_to_array(file_path, invert_colors, size)
                    images_data.append(layer)
                    print(f"Processed file {idx + 1}: {file_name} => Sample size: {len(layer)}")
                except Exception as e:
                    print(f"Failed to process {file_name}: {e}")
            else:
                raise FileNotFoundError(f"Expected file {file_name} does not exist in the directory {sub_dir_path}")
        numbers_data[sub_dir_name] = images_data
    # Приведение структуры к формату с ключом "numbers"
    data = {"numbers": numbers_data}
    with open(output_file, 'w') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    directory_path = 'numbers/'  # Путь к главной директории с поддиректориями
    output_file = '/home/kepler/Programming/Python/Projects/NeuralNetwork/image_encoder/encoded_images.json'
    # Настройки
    invert_colors = False
    image_size = (28, 28)
    encode_images_from_directory(directory_path, output_file, invert_colors, image_size)
