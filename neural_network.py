from configuration import logger
from layers import LayerBuilder, OuterLayer, HiddenLayer
from machine_learning import MachineLearning
from activation_functions import ActivationFunctions


class NeuralNetwork(MachineLearning, ActivationFunctions, LayerBuilder):
    """
    Класс для построения и управления нейронной сетью, наследуется от LayerBuilder.
    Управляет различными слоями и их взаимодействием.
    """

    def __init__(self, input_dataset):
        """
        Инициализирует нейронную сеть.
        :param input_dataset: Список входных данных.
        """
        super().__init__()
        self.training = True
        self.input_dataset = self.validate_input_dataset(input_dataset)
        self.layers = {}

    @staticmethod
    def validate_input_dataset(input_dataset):
        """
        Проверяет корректность входных данных.
        :param input_dataset: Список входных данных.
        :return: Проверенный список входных данных.
        :raises ValueError: Если входные данные некорректны.
        """
        if not isinstance(input_dataset, list):
            raise ValueError(f'Значение "{input_dataset}" должно быть списком!')
        if not all(isinstance(x, (int, float)) for x in input_dataset):
            raise ValueError(f'Все элементы списка "{input_dataset}" должны быть целыми или вещественными числами!')
        return input_dataset

    @staticmethod
    def propagate(layer):
        """
        Пропускает данные через слой.
        :param layer: Объект слоя.
        :return: Данные слоя.
        """
        return layer.get_layer_dataset()

    def add_layer(self, name: str, layer):
        """
        Добавляет слой в нейронную сеть.
        :param name: Название слоя.
        :param layer: Объект слоя.
        """
        logger.info(f'Добавление слоя "{name}" в сеть.')
        self.layers[name] = layer

    def remove_layer(self, name: str):
        """
        Удаляет слой из нейронной сети.
        :param name: Название слоя.
        """
        if name in self.layers:
            logger.info(f'Удаление слоя "{name}" из сети.')
            del self.layers[name]
        else:
            logger.warning(f'Слой "{name}" не найден при попытке удаления.')

    def get_layer(self, name: str):
        """
        Возвращает слой по его имени.
        :param name: Название слоя.
        :return: Объект слоя или None, если слой не найден.
        """
        return self.layers.get(name)

    def build_neural_network(self):
        """
        Строит нейронную сеть, добавляя внешние и скрытые слои.
        """
        logger.info('Начало построения нейронной сети.')
        input_outer_layer = OuterLayer(
            self.input_dataset, self.get_linear, self.get_linear, [True, True]
        )
        self.add_layer('input_outer_layer', input_outer_layer)
        logger.debug(f'Входной внешний слой создан с параметрами: {input_outer_layer}')

        hidden_layer_first = HiddenLayer(
            self.propagate(input_outer_layer), 12, self.get_leaky_relu, True
        )
        self.add_layer('hidden_layer_first', hidden_layer_first)
        logger.debug(f'Первый скрытый слой создан с параметрами: {hidden_layer_first}')

        hidden_layer_second = HiddenLayer(
            self.propagate(hidden_layer_first), 8, self.get_leaky_relu, True
        )
        self.add_layer('hidden_layer_second', hidden_layer_second)
        logger.debug(f'Второй скрытый слой создан с параметрами: {hidden_layer_second}')

        output_outer_layer = OuterLayer(
            self.propagate(hidden_layer_second), self.get_elu, self.get_elu, [True, True]
        )
        self.add_layer('output_outer_layer', output_outer_layer)
        logger.debug(f'Выходной внешний слой создан с параметрами: {output_outer_layer}')
        logger.info('Построение нейронной сети завершено.')

        if self.training:
            self.train_layers_on_dataset(
                self.data_number,
                input_outer_layer,
                hidden_layer_first,
                hidden_layer_second,
                output_outer_layer
            )
            logger.info('Обучение нейронной сети завершено.')

    def get_visualisation(self):
        """
        Выводит визуальное представление нейронной сети.
        """
        print(f'Класс: {self.__class__.__name__}')
        print(f'Всего слоёв: {len(self.layers)}')
        print(f'Входные данные: {self.input_dataset}\n')
        for name, layer in self.layers.items():
            print(f'Слой: {name}')
            print(f'Данные слоя: {[float(f'{i:.2f}') for i in layer.get_layer_dataset()]}\n')
            if name == 'output_outer_layer':
                print(f'Выходные данные: {sum(layer.get_layer_dataset()) * 10:.0f}\n')
