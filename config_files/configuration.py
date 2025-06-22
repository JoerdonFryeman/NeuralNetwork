from .base import *
from games.rps import RPS

# Запускает скрипт для преобразования изображений в числовые массивы.

use_image_encoder: bool = True
directory_path: str = 'numbers'
invert_colors: bool = False
image_size: tuple[int, int] = (28, 28)

if use_image_encoder:
    from encoders.image_encoder import ImageEncoder

    image_encoder = ImageEncoder()
    image_encoder.encode_images_from_directory(
        f'learning_data/{directory_path}', 'weights_biases_and_data/input_dataset.json', invert_colors, image_size
    )

# Запускает скрипт игры "Камень, ножницы, бумага".

use_rps: bool = False
if use_rps:
    rps = RPS()
    select_os_command('clear_screen')
    rps.get_computer_choice()
