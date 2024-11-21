from pickle import load
from configuration import logger
from layers import LayerBuilder, OuterLayer, HiddenLayer
from machine_learning import MachineLearning
from support_functions import ActivationFunctions
from visualisation import Visualisation


class NeuralNetwork(Visualisation, MachineLearning, ActivationFunctions, LayerBuilder):
    """
        Класс NeuralNetwork представляет собой реализацию нейронной сети,
        сочетающей возможности визуализации (Visualization), машинного обучения (MachineLearning),
        активационных функций (ActivationFunctions) и построения слоев (LayerBuilder).

        Основные методы класса включают:
        - Инициализацию с проверкой входных данных.
        - Загрузку весов и смещений из файла.
        - Добавление, удаление и получение слоев.
        - Пропуск данных через слои.
        - Построение нейронной сети с внешними и скрытыми слоями.

        Аргументы при инициализации:
        :param input_dataset: Список входных данных.
        """

    def __init__(self, training, initialization, input_dataset: list[float]):
        """
        Инициализирует нейронную сеть.
        :param input_dataset: Список входных данных.
        """
        super().__init__()
        self.training = training
        self.initialization = initialization
        self.input_dataset = self.validate_input_dataset(input_dataset)
        self.layers: dict[str, object] = {}

    @staticmethod
    def _load_weights_and_biases(filename: str) -> dict:
        """
        Загружает веса и смещения из указанного файла.

        :param filename: Имя файла, из которого будут загружены веса и смещения.
        :return: Словарь с весами и смещениями.
        """
        with open(filename, 'rb') as file:
            data: dict[str, dict[str, any]] = load(file)
        return data

    @staticmethod
    def validate_input_dataset(input_dataset: list) -> list[float]:
        """
        Проверяет корректность входных данных.

        :param input_dataset: Список входных данных, который нужно проверить на корректность.
        :return: Проверенный список входных данных.
        :raises ValueError: Если входные данные некорректны.

        Входные данные считаются корректными, если:
        - Они представлены в виде списка.
        - Все элементы списка являются целыми числами (int) или вещественными числами (float).
        """
        if not isinstance(input_dataset, list):
            raise ValueError(f'Значение "{input_dataset}" должно быть списком!')
        if not all(isinstance(x, (int, float)) for x in input_dataset):
            raise ValueError(f'Все элементы списка "{input_dataset}" должны быть целыми или вещественными числами!')
        return input_dataset

    @staticmethod
    def propagate(layer) -> list[float]:
        """
        Пропускает данные через слой и возвращает результаты.

        :param layer: Объект слоя, содержащий метод get_layer_dataset().
        :return: Данные слоя в виде списка float.

        Метод вызывает get_layer_dataset() у переданного объекта слоя и возвращает результаты.
        """
        return layer.get_layer_dataset()

    def add_layer(self, name: str, layer: object) -> None:
        """
        Добавляет слой в нейронную сеть.

        :param name: Название слоя.
        :param layer: Объект слоя.
        :return: None

        В данном методе осуществляется добавление нового слоя в словарь слоев сети.
        В качестве ключа используется имя слоя, а в качестве значения — сам объект слоя.
        """
        logger.info(f'Добавление слоя "{name}" в сеть.')
        self.layers[name] = layer

    def remove_layer(self, name: str) -> None:
        """
        Удаляет слой из нейронной сети.

        :param name: Название слоя.
        :return: None

        Метод проверяет наличие слоя в словаре слоев по его названию.
        Если слой найден, он удаляется, и в лог пишется соответствующее сообщение.
        В противном случае выводится предупреждение о том, что слой не найден.
        """
        if name in self.layers:
            logger.info(f'Удаление слоя "{name}" из сети.')
            del self.layers[name]
        else:
            logger.warning(f'Слой "{name}" не найден при попытке удаления.')

    def get_layer(self, name: str) -> object:
        """
        Возвращает слой по его имени.

        :param name: Название слоя.
        :return: Объект слоя или None, если слой не найден

        Метод ищет слой по его названию в словаре слоев и возвращает его.
        Если слой с указанным именем не найден, возвращается None.
        """
        return self.layers.get(name)

    def _create_layer(self, layer_class, layer_name: str, input_dataset, *args):
        """
        Вспомогательный метод для создания и добавления слоя.

        :param layer_class: Класс слоя.
        :param layer_name: Название слоя.
        :param input_dataset: Входные данные для слоя.
        :param args: Дополнительные аргументы для создания слоя.
        :return: Созданный объект слоя

        Метод пытается загрузить веса и смещения из файла 'weights_and_biases.pkl'.
        Если файл не найден, используются пустые значения. Затем создается слой,
        и добавляется в нейронную сеть с использованием метода add_layer().
        """
        try:
            data: dict = self._load_weights_and_biases('weights_and_biases.pkl')
            logger.info(f'Веса и смещения для слоя "{layer_name}" успешно загружены и установлены.')
        except FileNotFoundError:
            logger.error('Файл weights_and_biases.pkl не найден.')
            data = {'weights': {}, 'biases': {}}

        weights = data['weights'].get(layer_name)
        bias = data['biases'].get(layer_name)

        layer = layer_class(self.training, self.initialization, input_dataset, weights, bias, *args)
        self.add_layer(layer_name, layer)
        logger.debug(f'Слой "{layer_name}" создан с параметрами: {layer}')
        return layer

    def build_neural_network(self) -> None:
        """
        Строит нейронную сеть, добавляя внешние и скрытые слои.

        Метод создает первый и второй скрытые слои и внешний выходной слой
        с помощью метода `_create_layer`. Затем происходит построение модели,
        и, если стоит флаг `self.training`, проводится обучение сети.
        В конце вызывается метод визуализации сети.

        :return: None
        """
        hidden_layer_first = self._create_layer(
            HiddenLayer, 'hidden_layer_first', self.input_dataset,
            7, self.get_leaky_relu, True
        )
        hidden_layer_second = self._create_layer(
            HiddenLayer, 'hidden_layer_second', self.propagate(hidden_layer_first),
            7, self.get_leaky_relu, True
        )
        output_outer_layer = self._create_layer(
            OuterLayer, 'output_outer_layer', self.propagate(hidden_layer_second),
            self.get_elu, self.get_elu, [True, True]
        )
        logger.info('Построение нейронной сети завершено.')

        if self.training:
            self.train_layers_on_dataset(
                self.data_number,
                hidden_layer_first, hidden_layer_second, output_outer_layer
            )
            logger.info('Обучение нейронной сети завершено.')

        self.get_visualisation(self.input_dataset, self.layers)
