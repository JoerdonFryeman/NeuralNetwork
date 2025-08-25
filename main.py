from base.base import select_os_command, logger
from data.data import Data
from network.neural_network import NeuralNetwork
from encoders.text_encoder import TextEncoder


class Configuration:
    @staticmethod
    def use_text_encoder(text_encoder: bool, target_mode: bool) -> None:
        """Запускает скрипт преобразования текста в числовые массивы."""
        if text_encoder:
            text_encoder = TextEncoder(target_mode)
            text_encoder.encode_sentence()

    @staticmethod
    def use_image_encoder(image_encoder: bool, invert_colors: bool = False) -> None:
        """Запускает скрипт преобразования изображений в числовые массивы."""
        directory_path: str = 'numbers'
        image_size: tuple[int, int] = (28, 28)

        if image_encoder:
            from encoders.image_encoder import ImageEncoder

            image_encoder = ImageEncoder()
            image_encoder.encode_images_from_directory(
                f'learning_data/{directory_path}',
                'weights_biases_and_data/input_dataset.json', invert_colors, image_size
            )


class Control(Configuration):
    """Класс управляет параметрами и характеристиками нейронной сети."""

    __slots__ = (
        'training', 'init_func', 'epochs', 'learning_rate', 'learning_decay', 'error_tolerance',
        'regularization', 'lasso_regularization', 'ridge_regularization', 'visual'
    )

    def __init__(self):
        """
        Инициализирует объект класса с параметрами по умолчанию.

        :param training (bool): Флаг режима обучения. По умолчанию False.

        :param init_func (str): Метод инициализации весов сети. По умолчанию 'xavier'.
        :param epochs (int): Количество эпох обучения. По умолчанию 100.
        :param learning_rate (float): Начальная скорость обучения. По умолчанию 0.001.
        :param learning_decay (float): Коэффициент уменьшения скорости обучения. По умолчанию 0.009.
        :param error_tolerance (float): Допустимый предел ошибок. По умолчанию 0.001.
        :param regularization (float): Коэффициент регуляризации. По умолчанию 0.001.
        :param lasso_regularization (bool): Флаг использования L1 регуляризации. По умолчанию False.
        :param ridge_regularization (bool): Флаг использования L2 регуляризации. По умолчанию True.

        :param visual (bool): Флаг режима визуализации. По умолчанию True.
        """
        self.training: bool = True

        self.use_text_encoder(True, True)
        self.use_image_encoder(False)

        self.init_func: str = 'xavier'
        self.epochs: int = 100
        self.learning_rate: float = 0.001
        self.learning_decay: float = 0.009
        self.error_tolerance: float = 0.001
        self.regularization: float = 0.001
        self.lasso_regularization: bool = False
        self.ridge_regularization: bool = True

        self.visual: bool = True


control = Control()
data = Data()


def init_network() -> None | float:
    """
    Инициализирует параметры управления и создает объекты нейронной сети.

    :return: Возвращает результат вычислений выходного слоя, если сеть не находится в режиме обучения.
    """
    network = NeuralNetwork(
        control.training, control.init_func, data.get_data_sample(data.serial_class_number, data.serial_data_number)
    )
    output_layer = network.build_neural_network(
        control.epochs, control.learning_rate, control.learning_decay, control.error_tolerance,
        control.regularization, control.lasso_regularization, control.ridge_regularization
    )
    if not control.training:
        if control.visual:
            network.get_info_visualisation(network.input_dataset, network.layers, output_layer)
        return output_layer
    return None


def change_training_mode(training_mode: str) -> None:
    """
    Управляет консольными командами.

    :param training_mode: Команда начала обучения нейросети.
    """
    commands_yes: tuple[str, str, str, str] = ('да', 'д', 'yes', 'y')
    commands_no: tuple[str, str, str, str] = ('нет', 'н', 'no', 'n')

    if training_mode.lower() in commands_yes:
        control.training = True
        select_os_command('clear_screen')
        print('Выполняю обучение нейронной сети!\n')
    elif training_mode.lower() in commands_no:
        select_os_command('clear_screen')
        print('Выполняю построение нейронной сети!\n')
        control.training = False
    else:
        select_os_command('clear_screen')
        if training_mode.lower() != '':
            message = f'Команда "{training_mode.lower()}" не распознана!\n'
        else:
            message = 'Вы ничего не ответили!\n'
        print(f'{message}Выполняю построение нейронной сети.\n')
        control.training = False


def main() -> None:
    """Запускающая все процессы главная функция."""
    if control.training:
        change_training_mode(input('Внимание! Активирован режим обучения! Выполнить?\nДа/нет: '))
    try:
        if control.training:
            init_network()
            control.visual, control.training = False, False
            data.load_output_layer_data(init_network, True)
            print('Обучение нейронной сети завершено!\n')
        elif not control.training:
            data.load_output_layer_data(init_network, False)
            print('\nПостроение нейронной сети завершено!\n')
    except ValueError as v_error:
        logger.error(f'Проверка выдала ошибку: {v_error}')
    except ZeroDivisionError as z_error:
        logger.error(f'Целевое значение (target) не должно равняться нулю! {z_error}')
    except Exception as e_error:
        logger.error(f'Произошла непредвиденная ошибка: {e_error}')


if __name__ == '__main__':
    main()
