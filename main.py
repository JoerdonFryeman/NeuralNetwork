from data import Data
from configuration import logger
from neural_network import NeuralNetwork


class Control:
    """
    Класс для управления параметрами конфигурации нейронной сети.

    Атрибуты:
    ----------
    training : bool
        Указывает, находится ли сеть в режиме тренировки. По умолчанию False.
    initialization : str
        Метод инициализации весов сети. Например, 'uniform'. По умолчанию 'uniform'.
    epochs : int
        Количество эпох для обучения. По умолчанию 1000.
    learning_rate : float
        Скорость обучения нейронной сети. По умолчанию 0.001.
    error_tolerance : float
        Допустимый уровень ошибки для обучения. По умолчанию 0.001.
    regularization : float
        Параметр регуляризации для предотвращения переобучения. По умолчанию 0.001.
    lasso_regularization : bool
        Включение или отключение Lasso регуляризации. По умолчанию False.
    ridge_regularization : bool
        Включение или отключение Ridge регуляризации. По умолчанию True.
    """

    def __init__(self):
        """Инициализирует объект Control с параметрами по умолчанию."""
        self.training: bool = False
        self.initialization: str = 'uniform'
        self.epochs: int = 1000
        self.learning_rate: float = 0.001
        self.error_tolerance: float = 0.001
        self.regularization: float = 0.001
        self.lasso_regularization: bool = False
        self.ridge_regularization: bool = True


def initialize_objects(control: Control):
    data = Data()
    network = NeuralNetwork(
        control.training, control.initialization,
        data.get_data_sample()
    )
    return data, network


def main():
    """Основная функция, которая запускает процесс создания и визуализации нейронной сети."""
    control = Control()
    data, network = initialize_objects(control)

    try:
        logger.info("Начало построения нейронной сети.")
        network.build_neural_network(
            control.epochs, control.learning_rate, control.error_tolerance,
            control.regularization, control.lasso_regularization, control.ridge_regularization
        )
        logger.info("Построение нейронной сети завершено.")
    except ValueError as error:
        logger.error(f'Проверка выдала ошибку: {error}')
    except Exception as e:
        logger.error(f'Произошла непредвиденная ошибка: {e}')


if __name__ == '__main__':
    main()
