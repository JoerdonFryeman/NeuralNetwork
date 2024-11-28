from data import Data
from configuration import logger
from neural_network import NeuralNetwork


class Control:
    """Класс управляет параметрами и характеристиками нейронной сети."""

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
        :param test_mode (bool): Флаг тестового режима. По умолчанию False.
        """
        self.training: bool = False
        self.init_func: str = 'xavier'
        self.epochs: int = 1000
        self.learning_rate: float = 0.001
        self.learning_decay = 0.009
        self.error_tolerance: float = 0.001
        self.regularization: float = 0.001
        self.lasso_regularization: bool = False
        self.ridge_regularization: bool = True
        self.test_mode: bool = False


def init_objects(control: Control) -> tuple[Data, NeuralNetwork]:
    """
    Инициализирует и возвращает объекты данных и нейронной сети на основе заданного контроля параметров.

    :param control: Объект Control, содержащий параметры обучения и конфигурации для нейронной сети.
    :return: Кортеж, содержащий объекты Data и NeuralNetwork, готовые для использования.
    """
    data = Data()
    network = NeuralNetwork(control.training, control.init_func, data.get_data_sample())
    return data, network


def main() -> None:
    """
    Главная функция программы, запускающая процесс создания и обучения нейронной сети.

    Инициализирует параметры управления, создает объекты данных и нейронной сети,
    а затем начинает процесс обучения сети, обрабатывая любые возникшие ошибки в процессе.
    """
    control = Control()
    data, network = init_objects(control)
    try:
        logger.info("Начало построения нейронной сети.")
        network.build_neural_network(
            control.epochs, control.learning_rate, control.learning_decay, control.error_tolerance,
            control.regularization, control.lasso_regularization, control.ridge_regularization, control.test_mode
        )
        logger.info("Построение нейронной сети завершено.")
    except ValueError as error:
        logger.error(f'Проверка выдала ошибку: {error}')
    except Exception as e:
        logger.error(f'Произошла непредвиденная ошибка: {e}')


if __name__ == '__main__':
    main()
