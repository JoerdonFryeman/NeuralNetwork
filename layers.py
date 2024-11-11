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
    def _initialize_weights(cls, input_size: int, neuron_number: int) -> list[list[float]]:
        """
        Генерирует размеры весов для нейронов.
        :param input_size: Размер списка входных данных.
        :param neuron_number: Количество нейронов.
        :return: Массив из списков весов.
        """
        if cls._test_mode:
            seed(0)  # Фиксация предсказуемых значений для тестирования
        logger.info(f'Генерация весов для входных данных размером {input_size} и {neuron_number} нейронов.')
        return [[uniform(-0.01, 0.01) for _ in range(input_size)] for _ in range(neuron_number)]

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
            cls, input_dataset: list[int | float], neuron_number: int,
            weights: list[list[int | float]], bias: float, switch: bool | list[bool]
    ) -> list[float]:
        """
        Вычисляет значения массива данных нейронов с заданными параметрами.
        :param input_dataset: Список входных данных.
        :param neuron_number: Количество нейронов.
        :param weights: Список весов нейронов.
        :param bias: Смещение.
        :param switch: Булевое значение или список булевых значений.
        :return: Список результатов обработки данных нейронов.
        """
        if cls._test_mode:
            seed(0)  # Фиксация предсказуемых значений для тестирования
        logger.info(f'Вычисление данных для {neuron_number} нейронов с bias={bias}.')
        neuron_dataset = []
        switch_list = cls._verify_switch_type(switch, neuron_number)
        for n in range(neuron_number):
            neuron_dataset.append(
                [i * w + bias if switch_list[n] else i * w - bias for i, w in zip(input_dataset, weights[n])]
            )
        result = [sum(i) for i in neuron_dataset]
        logger.info(f'Результаты вычислений: {result}')
        return result


class OuterLayer(LayerBuilder):
    """
    Класс для входного слоя нейронной сети, наследуется от LayerBuilder.
    Выполняет начальную обработку входных данных, применяя указанную активационную функцию.
    """

    def __init__(
            self, input_dataset: list[int | float],
            activation_function_first, activation_function_second, switch_list: list[bool]
    ):
        """
        Инициализирует входной слой.
        :param input_dataset: Список входных данных.
        :param activation_function_first: Функция активации первого нейрона.
        :param activation_function_second: Функция активации второго нейрона.
        """
        if self._test_mode:
            seed(0)
        self.input_dataset = input_dataset
        self.input_size = len(input_dataset)
        self.__neuron_number = 2
        self.weights = self._initialize_weights(self.input_size, self.__neuron_number)
        self.bias = uniform(-0.01, 0.01)
        self.activation_function_first = activation_function_first
        self.activation_function_second = activation_function_second
        self.switch_list = switch_list

    def get_layer_dataset(self) -> list[float]:
        """
        Получает массив данных слоя с примененной активационной функцией.
        :return: Список значений после применения активационной функции.
        """
        neuron_data_first, neuron_data_second = self._calculate_neuron_dataset(
            self.input_dataset, self.__neuron_number, self.weights, self.bias, self.switch_list
        )
        logger.debug(self)
        return [self.activation_function_first(neuron_data_first), self.activation_function_second(neuron_data_second)]


class HiddenLayer(LayerBuilder):
    """
    Класс для глубокого слоя нейронной сети, наследуется от LayerBuilder.
    Обрабатывает данные слоями, создавая сложные представления входных данных.
    """

    def __init__(
            self, input_dataset: list[int | float],
            neuron_number: int, activation_function, switch: bool
    ):
        """
        Инициализирует глубокий слой.
        :param input_dataset: Список входных данных.
        :param neuron_number: Количество нейронов.
        :param activation_function: Функция активации для слоя.
        """
        if self._test_mode:
            seed(0)
        self.input_dataset = input_dataset
        self.input_size = len(input_dataset)
        self.neuron_number = neuron_number
        self.weights = self._initialize_weights(self.input_size, neuron_number)
        self.bias = uniform(-0.01, 0.01)
        self.activation_function = activation_function
        self.switch = switch

    def get_layer_dataset(self) -> list[float]:
        """
        Получает массив данных слоя с примененной активационной функцией.
        :return: Список значений после применения активационной функции.
        """
        result = self._calculate_neuron_dataset(
            self.input_dataset, self.neuron_number, self.weights, self.bias, self.switch
        )
        logger.debug(self)
        return [self.activation_function(i) for i in result]
