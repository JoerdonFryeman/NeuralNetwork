from random import uniform, seed
from configuration import logger
from support_functions import InitializationFunctions


class LayerBuilder(InitializationFunctions):
    """
    Класс для построения слоёв нейронной сети.
    Содержит методики для генерации весов, вычисления данных нейронов и различных активационных функций.
    """
    _test_mode: bool = False

    def __repr__(self) -> str:
        """
        Возвращает строковое представление объекта LayerBuilder.
        :return: Строковое представление объекта.
        """
        return f'Модуль: {__name__}; Класс: {self.__class__.__name__}; Адрес в памяти: {hex(id(self))}\n'

    def _select_initialization_function(
            self, mode: str, input_size: int = None, neuron_number: int = None, for_bias: bool = False
    ) -> float | tuple[float, float]:
        """
        Выбирает метод инициализации и возвращает сгенерированные значения.

        :param mode: Режим инициализации ('uniform', 'xavier', 'he').
        :param input_size: Размер входных данных (для 'xavier' и 'he').
        :param neuron_number: Размер выходных данных (для 'xavier').
        :param for_bias: Флаг для инициализации смещений.
        :return: Инициализированные значения в зависимости от режима (либо кортеж с границами диапазона, либо 0.0 для смещения).
        :raises ValueError: Если режим инициализации неизвестен.
        """
        if for_bias:
            return 0.0
        if mode == 'uniform':
            return self.get_uniform_initialization(0.5)
        elif mode == 'xavier':
            return self.get_xavier_initialization(input_size, neuron_number)
        elif mode == 'he':
            return self.get_he_initialization(input_size)
        else:
            raise ValueError(f'Неизвестный режим инициализации: {mode}')

    def _get_weights_mode(
            self, training, input_size: int, neuron_number: int, weights: list[list[float]] | None, mode: str
    ) -> list[list[float]]:
        """
        Инициализирует веса для слоёв нейронной сети.

        :param input_size: Размер входных данных.
        :param neuron_number: Количество нейронов.
        :param weights: Существующие веса, если они есть.
        :param mode: Режим инициализации ('uniform', 'xavier', 'he').
        :return: Инициализированные веса.
        """
        if training or not weights:
            if self._test_mode:
                seed(0)
            logger.info(f'Инициализация весов для входных данных размером {input_size} и {neuron_number} нейронов.')
            limits: float | tuple[float, float] = self._select_initialization_function(mode, input_size, neuron_number)
            return [[uniform(*limits) for _ in range(input_size)] for _ in range(neuron_number)]
        return weights

    def _get_bias_mode(self, training, bias: float, mode: str) -> float | tuple[float, float]:
        """
        Инициализирует смещения для слоёв нейронной сети.

        :param bias: Существующие смещения, если они есть.
        :param mode: Режим инициализации ('uniform', 'xavier', 'he').
        :return: Инициализированные смещения.
        """
        if training or not bias:
            if self._test_mode:
                seed(0)
            logger.info(f'Инициализация смещений.')
            limits: tuple[float, float] = self._select_initialization_function(mode, for_bias=(mode != 'uniform'))
            if mode == 'uniform':
                return uniform(*limits)
            return limits
        return bias

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
        raise ValueError(f'Ожидался тип bool или list[bool], но получен {type(switch).__name__}')

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
        neuron_dataset: list[list[float]] = []
        switch_list: list[bool] = cls._verify_switch_type(switch, neuron_number)
        for n in range(neuron_number):
            neuron_dataset.append(
                [i * w + bias if switch_list[n] else i * w - bias for i, w in zip(input_dataset, weights[n])]
            )
        result: list[float] = [sum(i) for i in neuron_dataset]
        logger.info(f'Результаты вычислений: {result}')
        return result


class OuterLayer(LayerBuilder):
    """
    Класс внешнего слоя нейронной сети, наследуется от LayerBuilder.
    Выполняет начальную или итоговую обработку данных, применяя указанную активационную функцию.
    """

    def __init__(
            self, training, initialization, input_dataset: list[int | float],
            weights: list[list[float]], bias: float | tuple[float, float],
            activation_function_first: callable, activation_function_second: callable, switch_list: list[bool]
    ):
        """
        Инициализирует входной слой.
        :param input_dataset: Список входных данных.
        :param activation_function_first: Функция активации первого нейрона.
        :param activation_function_second: Функция активации второго нейрона.
        """
        if self._test_mode:
            seed(0)
        self.input_dataset: list[int | float] = input_dataset
        self.__neuron_number: int = 2
        self.weights: list[list[float]] = self._get_weights_mode(
            training, len(input_dataset), self.__neuron_number, weights, initialization
        )
        self.bias: float | tuple[float, float] = self._get_bias_mode(training, bias, initialization)
        self.activation_function_first: callable = activation_function_first
        self.activation_function_second: callable = activation_function_second
        self.switch_list: list[bool] = switch_list

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
    Класс для скрытого слоя нейронной сети, наследуется от LayerBuilder.
    Обрабатывает данные слоями, создавая сложные представления входных данных.
    """

    def __init__(
            self, training, initialization, input_dataset: list[int | float],
            weights: list[list[float]], bias: float | tuple[float, float],
            neuron_number: int, activation_function: callable, switch: bool
    ):
        """
        Инициализирует скрытый слой.
        :param input_dataset: Список входных данных.
        :param neuron_number: Количество нейронов.
        :param activation_function: Функция активации для слоя.
        """
        if self._test_mode:
            seed(0)
        self.input_dataset: list[int | float] = input_dataset
        self.neuron_number: int = neuron_number
        self.weights: list[list[float]] = self._get_weights_mode(
            training, len(input_dataset), neuron_number, weights, initialization
        )
        self.bias: float | tuple[float, float] = self._get_bias_mode(training, bias, initialization)
        self.activation_function: callable = activation_function
        self.switch: bool = switch

    def get_layer_dataset(self) -> list[float]:
        """
        Получает массив данных слоя с примененной активационной функцией.
        :return: Список значений после применения активационной функции.
        """
        result: list[float] = self._calculate_neuron_dataset(
            self.input_dataset, self.neuron_number, self.weights, self.bias, self.switch
        )
        logger.debug(self)
        return [self.activation_function(i) for i in result]
