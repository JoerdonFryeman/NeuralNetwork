from random import uniform

from base.base import logger
from tools.support_functions import InitializationFunctions


class LayerBuilder(InitializationFunctions):
    """Класс предназначен для создания слоёв нейронной сети."""

    def __repr__(self) -> str:
        return f'Модуль: {__name__}; Класс: {self.__class__.__name__}; Адрес в памяти: {hex(id(self))}\n'

    @staticmethod
    def calculate_neuron_dataset(
            input_dataset: list[int | float], weights: list[list[int | float]], bias: float,
            neuron_number: int, act_func: callable
    ) -> list[float]:
        """
        Вычисляет выходные данные для заданного количества нейронов на основе входных данных и весов.

        :param input_dataset: Список входных данных для нейронов.
        :param weights: Двумерный список весов для каждого нейрона.
        :param bias: Смещение, которое будет добавлено или вычтено из взвешенной суммы, в зависимости от параметра switch.
        :param neuron_number: Количество нейронов, для которых необходимо произвести вычисления.
        :param act_func: Функция активации, применяемая к взвешенной сумме для каждого нейрона.

        :return: Список значений, вычисленных для каждого нейрона после применения функции активации.
        """
        activated_output = []
        for neuron in range(neuron_number):
            weighted_sum_and_bias = sum(i * n for i, n in zip(input_dataset, weights[neuron])) + bias
            activated_output.append(act_func(weighted_sum_and_bias))
        return activated_output

    def _select_init_func(
            self, init_func: str, input_size: int = None, neuron_number: int = None, for_bias: bool = False
    ) -> float | tuple[float, float]:
        """
        Выбирает и применяет метод инициализации параметров для слоя нейронной сети.

        :param init_func: Метод инициализации ('uniform', 'xavier', 'he').
        :param input_size: Количество входных нейронов (необходимо для 'xavier' и 'he').
        :param neuron_number: Количество выходных нейронов (необходимо для 'xavier').
        :param for_bias: Флаг, указывающий, требуется ли инициализация для смещения. По умолчанию False.

        :return: Инициализированное значение или кортеж инициализированных значений в зависимости от выбранного метода.
        :raises ValueError: Выбрасывается, если init_func не соответствует ни одному из поддерживаемых методов инициализации.
        """
        # Если параметр for_bias установлен в True, возвращается нулевое смещение, так как смещение не требует особой инициализации.
        if for_bias:
            return 0.0

        init_funcs_dict = {
            'uniform': lambda: self.get_uniform(),
            'xavier': lambda: self.get_xavier(input_size, neuron_number),
            'he': lambda: self.get_he(input_size)
        }
        if init_func not in init_funcs_dict:
            logger.error(f'Неизвестный режим инициализации: {init_func}')
            raise ValueError(f'Неизвестный режим инициализации: {init_func}')
        return init_funcs_dict[init_func]()

    def select_weights_mode(
            self, training: bool, init_func: str, input_size: int, weights: list[list[float]] | None, neuron_number: int
    ) -> list[list[float]]:
        """
        Определяет и инициализирует режим весов для слоя нейронной сети.

        :param training: Флаг, указывающий на режим обучения. Если True, инициируются новые веса.
        :param init_func: Метод инициализации весов ('uniform', 'xavier', 'he').
        :param input_size: Количество входных нейронов.
        :param weights: Существующий список весов или None, если веса нужно инициализировать.
        :param neuron_number: Количество выходных нейронов.

        :return: Список инициализированных весов, если текущий режим обучения или веса отсутствуют. В противном случае возвращает переданные веса.
        """
        # Проверяет, находится ли модель в режиме обучения или установлены ли веса.
        if training or not weights:
            logger.info(f'Инициализация весов для входных данных размером {input_size} и {neuron_number} нейронов.')
            # Получение пределов инициализации, вызывается select_init_func с соответствующими параметрами.
            limits: float | tuple[float, float] = self._select_init_func(init_func, input_size, neuron_number)
            # Создаёт и возвращает двумерный список весов, используя переданные пределы для каждого нейрона.
            return [[uniform(*limits) for _ in range(input_size)] for _ in range(neuron_number)]
        # Если веса уже установлены и модель не в режиме обучения, возвращает существующие веса.
        return weights

    def select_bias_mode(self, training: bool, init_func: str, bias: float) -> float | tuple[float, float]:
        """
        Определяет и инициализирует режим смещения (bias) для слоя нейронной сети.

        :param training: Флаг, указывающий на режим обучения. Если True, инициируются новое смещение.
        :param init_func: Метод инициализации смещения ('uniform', 'xavier', 'he').
        :param bias: Текущее значение смещения или 0, если оно должно быть инициализировано.

        :return: Инициализированное значение смещения или диапазон инициализированных значений в зависимости от выбранного метода. Если режим обучения выключен и смещение уже задано, возвращает текущее значение смещения.
        """
        if training or not bias:
            logger.info(f'Инициализация смещений.')
            # Получает пределы инициализации, вызывая select_init_func с параметром for_bias, который равен True, если метод инициализации не 'uniform'.
            limits: tuple[float, float] = self._select_init_func(init_func, for_bias=(init_func != 'uniform'))
            # Если метод инициализации 'uniform', возвращает случайное число внутри полученных пределов.
            if init_func == 'uniform':
                return uniform(*limits)
            # Для других методов возвращает сами пределы.
            return limits
        # Если смещение уже задано и модель не в режиме обучения, возвращает текущее смещение.
        return bias


class HiddenLayer(LayerBuilder):
    """Класс представляет собой скрытый слой нейронной сети."""

    __slots__ = ('input_dataset', 'weights', 'bias', 'neuron_number', 'act_func')

    def __init__(
            self, training, init_func, input_dataset: list[int | float], weights: list[list[float]], bias: float,
            neuron_number: int, act_func: callable
    ):
        self.input_dataset: list[int | float] = input_dataset
        self.weights: list[list[float]] = self.select_weights_mode(
            training, init_func, len(input_dataset), weights, neuron_number
        )
        self.bias: float | tuple[float, float] = self.select_bias_mode(training, init_func, bias)
        self.neuron_number: int = neuron_number
        self.act_func: callable = act_func

    def get_layer_dataset(self) -> list[float]:
        """
        Вычисляет и возвращает выходы слоя с учетом весов, смещений и функции активации.

        :return: Список значений, представляющих выходы каждого нейрона после применения функции активации.
        """
        result: list[float] = self.calculate_neuron_dataset(
            self.input_dataset, self.weights, self.bias, self.neuron_number, self.act_func
        )
        return result
