from config_files.configuration import select_os_command, logger
from data.data import Data
from network.neural_network import NeuralNetwork


class Control:
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
        :param epochs (int): Количество эпох обучения. По умолчанию 1000.
        :param learning_rate (float): Начальная скорость обучения. По умолчанию 0.001.
        :param learning_decay (float): Коэффициент уменьшения скорости обучения. По умолчанию 0.009.
        :param error_tolerance (float): Допустимый предел ошибок. По умолчанию 0.001.
        :param regularization (float): Коэффициент регуляризации. По умолчанию 0.001.
        :param lasso_regularization (bool): Флаг использования L1 регуляризации. По умолчанию False.
        :param ridge_regularization (bool): Флаг использования L2 регуляризации. По умолчанию True.

        :param visual (bool): Флаг режима визуализации. По умолчанию True.
        """
        self.training: bool = False
        self.init_func: str = 'xavier'
        self.epochs: int = 1000
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
    select_os_command('clear_screen')
    change_training_mode(input('Активировать режим обучения?\n'))
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
        logger.error(f'Возможно неверное именование! Имя класса (каталога) данных не должно равняться нулю! {z_error}')
    except Exception as e_error:
        logger.error(f'Произошла непредвиденная ошибка: {e_error}')


if __name__ == '__main__':
    main()
