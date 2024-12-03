from random import uniform, seed

from config_files.configuration import logger
from support_functions import InitializationFunctions


class LayerBuilder(InitializationFunctions):
    """Класс предназначен для создания слоёв нейронной сети."""

    def __repr__(self) -> str:
        """
        Возвращает строковое представление объекта LayerBuilder.
        :return: Строковое представление объекта.
        """
        return f'Модуль: {__name__}; Класс: {self.__class__.__name__}; Адрес в памяти: {hex(id(self))}\n'

    def _select_init_func(
            self, init_func: str, input_size: int = None,
            neuron_number: int = None, for_bias: bool = False
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
        # Словарь сопоставляет названия методов инициализации с соответствующими лямбда-функциями.
        # Каждая лямбда-функция вызывает метод для определенного режима инициализации.
        init_funcs_dict = {
            'uniform': lambda: self.get_uniform(),
            'xavier': lambda: self.get_xavier(input_size, neuron_number),
            'he': lambda: self.get_he(input_size)
        }
        if init_func not in init_funcs_dict:
            raise ValueError(f'Неизвестный режим инициализации: {init_func}')
        # Возвращаем результат выполнения соответствующей функции инициализации из словаря.
        return init_funcs_dict[init_func]()

    def select_weights_mode(
            self, training, input_size: int, neuron_number: int,
            weights: list[list[float]] | None, init_func: str, test_mode: bool
    ) -> list[list[float]]:
        """
        Определяет и инициализирует режим весов для слоя нейронной сети.

        :param training: Флаг, указывающий на режим обучения. Если True, инициируются новые веса.
        :param input_size: Количество входных нейронов.
        :param neuron_number: Количество выходных нейронов.
        :param weights: Существующий список весов или None, если веса нужно инициализировать.
        :param init_func: Метод инициализации весов ('uniform', 'xavier', 'he').
        :param test_mode: Флаг, указывающий на тестовый режим. При тестовом режиме перед инициализацией задается фиксированное начальное значение для генератора случайных чисел для воспроизводимости.

        :return: Список инициализированных весов, если текущий режим обучения или веса отсутствуют. В противном случае возвращает переданные веса.
        """
        # Проверяет, находится ли модель в режиме обучения или установлены ли веса.
        if training or not weights:
            # Если активирован тестовый режим, устанавливается фиксированное значение для генератора случайных чисел.
            if test_mode:
                seed(0)
            logger.info(f'Инициализация весов для входных данных размером {input_size} и {neuron_number} нейронов.')
            # Получение пределов инициализации, вызывается select_init_func с соответствующими параметрами.
            limits: float | tuple[float, float] = self._select_init_func(init_func, input_size, neuron_number)
            # Создаёт и возвращает двумерный список весов, используя переданные пределы для каждого нейрона.
            return [[uniform(*limits) for _ in range(input_size)] for _ in range(neuron_number)]
        # Если веса уже установлены и модель не в режиме обучения, возвращает существующие веса.
        return weights

    def select_bias_mode(
            self, training, bias: float, init_func: str, test_mode: bool
    ) -> float | tuple[float, float]:
        """
        Определяет и инициализирует режим смещения (bias) для слоя нейронной сети.

        :param training: Флаг, указывающий на режим обучения. Если True, инициируются новое смещение.
        :param bias: Текущее значение смещения или 0, если оно должно быть инициализировано.
        :param init_func: Метод инициализации смещения ('uniform', 'xavier', 'he').
        :param test_mode: Флаг, указывающий на тестовый режим. При тестовом режиме перед инициализацией задается фиксированное начальное значение для генератора случайных чисел для воспроизводимости.

        :return: Инициализированное значение смещения или диапазон инициализированных значений в зависимости от выбранного метода. Если режим обучения выключен и смещение уже задано, возвращает текущее значение смещения.
        """
        if training or not bias:
            if test_mode:
                seed(0)
            logger.info(f'Инициализация смещений.')
            # Получает пределы инициализации, вызывая select_init_func с параметром for_bias,
            # который равен True, если метод инициализации не 'uniform'.
            limits: tuple[float, float] = self._select_init_func(init_func, for_bias=(init_func != 'uniform'))
            # Если метод инициализации 'uniform', возвращает случайное число внутри полученных пределов.
            if init_func == 'uniform':
                return uniform(*limits)
            # Для других методов возвращает сами пределы.
            return limits
        # Если смещение уже задано и модель не в режиме обучения, возвращает текущее смещение.
        return bias

    @staticmethod
    def calculate_neuron_dataset(
            input_dataset: list[int | float], neuron_number: int, weights: list[list[int | float]],
            bias: float, activate_func: callable, switch: bool, test_mode: bool
    ) -> list[float]:
        """
        Вычисляет выходные данные для заданного количества нейронов на основе входных данных и весов.

        :param input_dataset: Список входных данных для нейронов.
        :param neuron_number: Количество нейронов, для которых необходимо произвести вычисления.
        :param weights: Двумерный список весов для каждого нейрона.
        :param bias: Смещение, которое будет добавлено или вычтено из взвешенной суммы, в зависимости от параметра switch.
        :param activate_func: Функция активации, применяемая к взвешенной сумме для каждого нейрона.
        :param switch: Логический флаг, определяющий, будет ли смещение добавлено (если True) или вычтено (если False) из взвешенной суммы.
        :param test_mode: Флаг, указывающий на тестовый режим. При включенном тестовом режиме задается фиксированное начальное значение для генератора случайных чисел для воспроизводимости.

        :return: Список значений, вычисленных для каждого нейрона после применения функции активации.
        """
        if test_mode:
            seed(0)
        logger.info(f'Вычисление данных для {neuron_number} нейронов с bias={bias}. Используя switch={switch}')
        # Инициализация списка для хранения результатов вычислений нейронов.
        neuron_dataset: list[float] = []
        # Проходим по каждому нейрону
        for n in range(neuron_number):
            # Для каждого нейрона вычисляется взвешенная сумма входных данных и соответствующих весов.
            # Если `switch` равен `True`, то к каждой взвешенной сумме добавляется смещение `bias`.
            # Если `switch` равен `False`, то смещение вычитается.
            neuron_output = sum(i * w + bias if switch else i * w - bias for i, w in zip(input_dataset, weights[n]))
            # Результат взвешенной суммы передается функции активации,
            # и результат этой функции добавляется в список результатов.
            neuron_dataset.append(activate_func(neuron_output))
        logger.info(f'Результаты вычислений: {neuron_dataset}')
        return neuron_dataset


class HiddenLayer(LayerBuilder):
    """Класс представляет собой скрытый слой нейронной сети."""

    def __init__(
            self, training, init_func, input_dataset: list[int | float],
            weights: list[list[float]], bias: float | tuple[float, float],
            neuron_number: int, activate_func: callable, switch: bool, test_mode: bool
    ):
        if test_mode:
            seed(0)
        self.input_dataset: list[int | float] = input_dataset
        self.neuron_number: int = neuron_number
        self.weights: list[list[float]] = self.select_weights_mode(
            training, len(input_dataset), neuron_number, weights, init_func, test_mode
        )
        self.bias: float | tuple[float, float] = self.select_bias_mode(
            training, bias, init_func, test_mode
        )
        self.activate_func: callable = activate_func
        self.switch: bool = switch
        self.test_mode = test_mode

    def get_layer_dataset(self) -> list[float]:
        """
        Вычисляет и возвращает выходы слоя с учетом весов, смещений и функции активации.

        :return: Список значений, представляющих выходы каждого нейрона после применения функции активации.
        """
        # Вызывается метод calculate_neuron_dataset для расчета выходных данных каждого нейрона.
        result: list[float] = self.calculate_neuron_dataset(
            self.input_dataset, self.neuron_number, self.weights, self.bias,
            self.activate_func, self.switch, self.test_mode
        )
        logger.debug(self)
        return result
