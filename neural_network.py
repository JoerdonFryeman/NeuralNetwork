from random import uniform, seed
from configuration import logger


class LayerBuilder:
    """
    Класс для построения слоёв нейронной сети.
    Содержит методики для генерации весов, вычисления данных нейронов и различных активационных функций.
    """
    _test_mode = False

    def __repr__(self):
        """
        Возвращает строковое представление объекта LayerBuilder.
        :return: Строковое представление объекта.
        """
        return f'Модуль: {__name__}; Класс: {self.__class__.__name__}; Адрес в памяти: {hex(id(self))}\n'

    @classmethod
    def _generate_weights_size(cls, input_dataset: list[int | float], neuron_number: int) -> list[list[float]]:
        """
        Генерирует размеры весов для нейронов.
        :param input_dataset: Список входных данных.
        :param neuron_number: Количество нейронов.
        :return: Массив из списков весов.
        """
        if cls._test_mode:
            seed(0)  # Фиксация предсказуемых значений для тестирования
        logger.info(f'Генерация весов для входных данных размером {len(input_dataset)} и {neuron_number} нейронов.')
        return [[uniform(-1, 1) for _ in range(len(input_dataset))] for _ in range(neuron_number)]

    @staticmethod
    def _verify_switch_type(switch: bool | list[bool], neuron_number: int) -> list[bool]:
        """
        Проверяет тип переключателя для нейронов.
        :param switch: Булевое значение или список булевых значений.
        :param neuron_number: Количество нейронов.
        :return: Список булевых значений.
        """
        if isinstance(switch, bool):
            logger.info(f'Преобразование булевого значения {switch} в список из {neuron_number} элементов.')
            return [switch] * neuron_number
        elif isinstance(switch, list):
            logger.info(f'Использование переданного списка булевых значений {switch}.')
            return switch
        raise ValueError(f'Значение "{switch}" должно быть {bool} или {list}!')

    @classmethod
    def _calculate_neuron_dataset(
            cls, input_dataset: list[int | float], neuron_number: int, bias: float, switch: bool | list[bool]
    ) -> list[float]:
        """
        Вычисляет значения массива данных нейронов с заданными параметрами.
        :param input_dataset: Список входных данных.
        :param neuron_number: Количество нейронов.
        :param bias: Смещение.
        :param switch: Булевое значение или список булевых значений.
        :return: Список результатов обработки данных нейронов.
        """
        if cls._test_mode:
            seed(0)  # Фиксация предсказуемых значений для тестирования
        logger.info(f'Вычисление данных для {neuron_number} нейронов с bias={bias}.')
        neuron_dataset = []
        weights = cls._generate_weights_size(input_dataset, neuron_number)
        switch_list = cls._verify_switch_type(switch, neuron_number)
        for n in range(neuron_number):
            neuron_dataset.append(
                [i * w + bias if switch_list[n] else i * w - bias for i, w in zip(input_dataset, weights[n])]
            )
        result = [sum(i) for i in neuron_dataset]
        logger.info(f'Результаты вычислений: {result}')
        return result

    @staticmethod
    def _get_linear(x):
        """
        Линейная активационная функция.
        :param x: Входное значение.
        :return: То же входное значение.
        """
        return x

    @staticmethod
    def _get_relu(x):
        """
        ReLU (Rectified Linear Unit) активационная функция.
        :param x: Входное значение.
        :return: Максимум между нулем и входным значением.
        """
        return max(0, x)

    @staticmethod
    def _get_sigmoid(x):
        """
        Сигмоидная активационная функция.
        :param x: Входное значение.
        :return: Значение сигмоидной функции для входного значения.
        """
        n = 10
        exp = 1.0
        for i in range(n, 0, -1):
            exp = 1 + x * exp / i
        return 1 / (1 + exp)

    @staticmethod
    def _get_tanh(x):
        """
        Активационная функция гиперболический тангенс (tanh).
        :param x: Входное значение.
        :return: Значение функции tanh для входного значения.
        """
        e_pos_2x = 1.0
        e_neg_2x = 1.0
        n = 10
        for i in range(n, 0, -1):
            e_pos_2x = 1 + 2 * x * e_pos_2x / i
            e_neg_2x = 1 - 2 * x * e_neg_2x / i
        return (e_pos_2x - e_neg_2x) / (e_pos_2x + e_neg_2x)


class InputLayer(LayerBuilder):
    """
    Класс для входного слоя нейронной сети, наследуется от LayerBuilder.
    Выполняет начальную обработку входных данных, применяя указанную активационную функцию.
    """

    def __init__(self, input_dataset: list[int | float], activation_function):
        """
        Инициализирует входной слой.
        :param input_dataset: Список входных данных.
        :param activation_function: Функция активации для слоя.
        """
        if self._test_mode:
            seed(0)
        self.input_dataset = input_dataset
        self.__neuron_number = 2
        self.bias = uniform(-1, 1)
        self.switch_list = [False, True]
        self.activation_function = activation_function

    def get_layer_dataset(self) -> list[float]:
        """
        Получает массив данных слоя с примененной активационной функцией.
        :return: Список значений после применения активационной функции.
        """
        result_negative, result_positive = self._calculate_neuron_dataset(
            self.input_dataset, self.__neuron_number, self.bias, self.switch_list
        )
        logger.debug(self)
        return [self.activation_function(result_negative), self.activation_function(result_positive)]


class DeepLayer(LayerBuilder):
    """
    Класс для глубокого слоя нейронной сети, наследуется от LayerBuilder.
    Обрабатывает данные слоями, создавая сложные представления входных данных.
    """

    def __init__(self, input_dataset: list[int | float], neuron_number: int, activation_function):
        """
        Инициализирует глубокий слой.
        :param input_dataset: Список входных данных.
        :param neuron_number: Количество нейронов.
        :param activation_function: Функция активации для слоя.
        """
        if self._test_mode:
            seed(0)
        self.input_dataset = input_dataset
        self.neuron_number = neuron_number
        self.bias = uniform(-1, 1)
        self.activation_function = activation_function

    def get_layer_dataset(self) -> list[float]:
        """
        Получает массив данных слоя с примененной активационной функцией.
        :return: Список значений после применения активационной функции.
        """
        result = self._calculate_neuron_dataset(self.input_dataset, self.neuron_number, self.bias, True)
        logger.debug(self)
        return [self.activation_function(i) for i in result]


class OutputLayer(LayerBuilder):
    """
    Класс для выходного слоя нейронной сети, наследуется от LayerBuilder.
    Завершает обработку данных и возвращает конечный результат.
    """

    def __init__(self, input_dataset, activation_function):
        """
        Инициализирует выходной слой.
        :param input_dataset: Список входных данных.
        :param activation_function: Функция активации для слоя.
        """
        if self._test_mode:
            seed(0)  # Фиксация предсказуемых значений для тестирования
        self.input_dataset = input_dataset
        self.bias = uniform(-1, 1)
        self.activation_function = activation_function

    def get_layer_dataset(self):
        """
        Получает массив данных слоя с примененной активационной функцией.
        :return: Значение после применения активационной функции.
        """
        input_dataset = self.input_dataset
        sum_input = sum(input_dataset)
        if sum_input > 0:
            result = sum_input + self.bias
        else:
            result = sum_input - self.bias
        logger.debug(self)
        return self.activation_function(result)


class NeuralNetwork(LayerBuilder):
    """
    Класс для построения и управления нейронной сетью, наследуется от LayerBuilder.
    Управляет различными слоями и их взаимодействием.
    """

    def __init__(self, input_dataset):
        """
        Инициализирует нейронную сеть.
        :param input_dataset: Список входных данных.
        """
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
        Строит нейронную сеть, добавляя входной, глубокие и выходной слои.
        """
        logger.info('Начало построения нейронной сети.')
        input_layer = InputLayer(self.input_dataset, self._get_sigmoid)
        self.add_layer('input_layer', input_layer)
        logger.debug(f'Входной слой создан с параметрами: {input_layer}')

        deep_layer_first = DeepLayer(self.propagate(input_layer), 3, self._get_tanh)
        self.add_layer('deep_layer_first', deep_layer_first)
        logger.debug(f'Первый глубокий слой создан с параметрами: {deep_layer_first}')

        deep_layer_second = DeepLayer(self.propagate(deep_layer_first), 2, self._get_tanh)
        self.add_layer('deep_layer_second', deep_layer_second)
        logger.debug(f'Второй глубокий слой создан с параметрами: {deep_layer_second}')

        output_layer = OutputLayer(self.propagate(deep_layer_second), self._get_tanh)
        self.add_layer('output_layer', output_layer)
        logger.debug(f'Выходной слой создан с параметрами: {output_layer}')

        logger.info('Построение нейронной сети завершено.')

    def get_visualisation(self):
        """
        Выводит визуальное представление нейронной сети.
        """
        print(f'Класс: {self.__class__.__name__}')
        print(f'Всего слоёв: {len(self.layers)}')
        print(f'Входные данные: {self.input_dataset}\n')
        for name, layer in self.layers.items():
            print(f'Слой: {name}')
            print(f'Данные слоя: {layer.get_layer_dataset()}\n')
