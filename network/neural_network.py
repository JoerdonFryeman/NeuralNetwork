from pickle import load

from base.base import logger
from machine_learning.train import Train
from tools.support_functions import ActivationFunctions
from .layers import LayerBuilder, HiddenLayer


class NeuralNetwork(Train, ActivationFunctions, LayerBuilder):
    """Класс построения многослойной нейронной сети."""

    __slots__ = ('training', 'init_func', 'input_dataset', 'layers')

    def __init__(self, training: bool, init_func: str, input_dataset: list[float]):
        """
        Инициализирует экземпляр класса с заданными параметрами обучения, методом инициализации.

        :param training: Флаг, обозначающий режим тренировки сети.
        :param init_func: Метод инициализации весов нейронной сети.
        :param input_dataset: Набор входных данных, представленный списком чисел.
        :param layers (dict): Задействованные в текущей модели слои.
        """
        super().__init__()
        self.training: bool = training
        self.init_func: str = init_func
        self.input_dataset: list[int | float] = self._validate_input_dataset(input_dataset)
        self.layers: dict[str, object] = {}

    @staticmethod
    def _load_weights_and_biases(filename: str) -> dict:
        """
        Загружает веса и смещения из указанного файла.

        :param filename: Имя файла, из которого будут загружены веса и смещения.
        :return: Словарь с весами и смещениями.
        """
        with open(filename, 'rb') as file:
            data: dict[str, dict] = load(file)
        return data

    @staticmethod
    def _validate_input_dataset(input_dataset: list) -> list[int | float]:
        """
        Проверяет корректность входных данных.
        Входные данные считаются корректными, если они представлены в виде списка.

        :param input_dataset: Список входных данных, который нужно проверить на корректность.

        :return: Проверенный список входных данных.
        :raises ValueError: Если входные данные некорректны.
        """
        # Входные данные считаются корректными, если они представлены в виде списка.
        if not isinstance(input_dataset, list):
            raise ValueError(f'Значение "{input_dataset}" должно быть списком!')
        # Если все элементы списка являются целыми числами (int) или вещественными числами (float).
        if not all(isinstance(x, (int, float)) for x in input_dataset):
            raise ValueError(f'Все элементы списка "{input_dataset}" должны быть целыми или вещественными числами!')
        return input_dataset

    @staticmethod
    def _propagate(layer) -> list[int | float]:
        """
        Пропускает данные через слой и возвращает результаты.
        Метод вызывает get_layer_dataset() у переданного объекта слоя и возвращает результаты.

        :param layer: Объект слоя, содержащий метод get_layer_dataset().

        :return: Данные слоя в виде списка float.
        """
        return layer.get_layer_dataset()

    def _add_layer(self, name: str, layer) -> None:
        """
        Добавляет слой в нейронную сеть.
        В данном методе осуществляется добавление нового слоя в словарь слоев сети.
        В качестве ключа используется имя слоя, а в качестве значения — сам объект слоя.

        :param name: Название слоя.
        :param layer: Объект слоя.

        :return: None
        """
        logger.info(f'Добавление слоя "{name}" в сеть.')
        self.layers[name] = layer

    def _create_layer(
            self, layer_class, layer_name: str, input_dataset: list[int | float], neuron_number: int, act_func: callable
    ):
        """
        Вспомогательный метод для создания и добавления слоя.
        Метод пытается загрузить веса и смещения из файла 'weights_and_biases.pkl'.
        Если файл не найден, используются пустые значения. Затем создается слой,
        и добавляется в нейронную сеть с использованием метода add_layer().

        :param layer_class: Класс слоя.
        :param layer_name: Название слоя.
        :param input_dataset: Входные данные для слоя.
        :param neuron_number: Количество нейронов, для которых необходимо произвести вычисления.
        :param act_func: Функция активации, применяемая к взвешенной сумме для каждого нейрона.

        :return: Созданный объект слоя
        """
        try:
            # Пытается загрузить веса и смещения из файла.
            data: dict = self._load_weights_and_biases('weights_biases_and_data/weights_and_biases.pkl')
            logger.info(f'Веса и смещения для слоя "{layer_name}" успешно загружены и установлены.')
        except FileNotFoundError:
            logger.error('Файл weights_and_biases.pkl не найден!')
            data: dict = {'weights': {}, 'biases': {}}

        weights: list[list[float]] = data['weights'].get(layer_name)
        bias: float = data['biases'].get(layer_name)
        # Создаётся объект слоя, инициализируя его текущими весами и смещениями.
        layer = layer_class(self.training, self.init_func, input_dataset, weights, bias, neuron_number, act_func)
        # Созданный слой добавляется в нейронную сеть.
        self._add_layer(layer_name, layer)
        logger.debug(f'Слой "{layer_name}" создан с параметрами: {layer}')
        # Возвращается объект созданного слоя, который теперь является частью архитектуры нейронной сети.
        return layer

    def build_neural_network(
            self, epochs: int, learning_rate: float, learning_decay: float, error_tolerance: float,
            regularization: float, lasso_regularization: bool, ridge_regularization: bool
    ) -> float:
        """
        Строит нейронную сеть, добавляя внешние и скрытые слои.

        Метод создает первый и второй скрытые слои и внешний выходной слой
        с помощью метода `_create_layer`. Затем происходит построение модели,
        и, если стоит флаг `self.training`, проводится обучение сети.
        В конце вызывается метод визуализации сети.

        :param epochs: Количество эпох для обучения.
        :param learning_rate: Скорость обучения.
        :param learning_decay: Уменьшение скорости обучения.
        :param error_tolerance: Допустимый уровень ошибки.
        :param regularization: Параметр регуляризации.
        :param lasso_regularization: Использовать Lasso регуляризацию.
        :param ridge_regularization: Использовать Ridge регуляризацию.

        :return output_layer: Выходные данные.
        """
        hidden_layer_first = self._create_layer(
            HiddenLayer, 'hidden_layer_first', self.input_dataset, 5, self.get_tanh
        )
        hidden_layer_second = self._create_layer(
            HiddenLayer, 'hidden_layer_second', self._propagate(hidden_layer_first), 20, self.get_tanh
        )
        output_layer = self.get_sigmoid(sum(self._propagate(hidden_layer_second)))
        # В режиме обучения запускается метод обучения слоёв на массиве данных.
        if self.training:
            self.train_layers_on_dataset(
                hidden_layer_first, hidden_layer_second, epochs, learning_rate, learning_decay,
                error_tolerance, regularization, lasso_regularization, ridge_regularization
            )
            logger.info('Обучение нейронной сети завершено.')
        # Возвращается результат в виде вещественного числа в десятичной системе счисления.
        return float(f'{output_layer:.10f}')
