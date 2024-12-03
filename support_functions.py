class ActivationFunctions:
    """Класс предоставляет реализации различных используемых в нейронных сетях активационных функций."""

    @staticmethod
    def get_linear(x: float) -> float:
        """
        Линейная активационная функция.
        :param x: Входное значение.
        :return: То же входное значение.
        """
        return x

    @staticmethod
    def get_relu(x: float) -> float:
        """
        ReLU (Rectified Linear Unit) активационная функция.
        :param x: Входное значение.
        :return: Максимум между нулем и входным значением.
        """
        return max(0.0, x)

    @staticmethod
    def get_sigmoid(x: float) -> float:
        """
        Сигмоидная активационная функция.
        :param x: Входное значение.
        :return: Значение сигмоидной функции для входного значения.
        """
        n: int = 10
        exp: float = 1.0
        for i in range(n, 0, -1):
            exp = 1 + x * exp / i
        return 1 / (1 + exp)

    @staticmethod
    def get_tanh(x: float) -> float:
        """
        Активационная функция гиперболический тангенс (tanh).
        :param x: Входное значение.
        :return: Значение функции tanh для входного значения.
        """
        e_pos_2x: float = 1.0
        e_neg_2x: float = 1.0
        n: int = 10
        for i in range(n, 0, -1):
            e_pos_2x = 1 + 2 * x * e_pos_2x / i
            e_neg_2x = 1 - 2 * x * e_neg_2x / i
        return (e_pos_2x - e_neg_2x) / (e_pos_2x + e_neg_2x)

    @staticmethod
    def get_leaky_relu(x: float, alpha: float = 0.01) -> float:
        """
        Leaky ReLU активационная функция.
        :param x: Входное значение.
        :param alpha: Сила утечки (по умолчанию 0.01).
        :return: Максимум между alpha*x и входным значением.
        """
        return x if x > 0 else alpha * x

    @staticmethod
    def __exp(x: float, terms: int = 20) -> float:
        """
        Вычисляет экспоненту числа x с помощью ряда Тейлора.
        :param x: Входное значение.
        :param terms: Количество членов ряда (по умолчанию 20).
        :return: Значение exp(x).
        """
        result: float = 1.0
        factorial: float = 1.0
        power: float = 1.0
        for i in range(1, terms):
            factorial *= i
            power *= x
            result += power / factorial
        return result

    def get_elu(self, x: float, alpha: float = 1.0) -> float:
        """
        ELU (Exponential Linear Unit) активационная функция.
        :param x: Входное значение.
        :param alpha: Параметр альфа (по умолчанию 1.0).
        :return: ELU от входного значения.
        """
        return x if x >= 0 else alpha * (self.__exp(x) - 1)

    def get_softmax(self, x: float | list[float]) -> list[float]:
        """
        Softmax активационная функция.
        :param x: Входное значение (может быть списком или отдельным числом).
        :return: Softmax распределение значений.
        """
        if isinstance(x, (float, int)):
            x = [x]
        max_val = max(x)
        exp_values: list[float] = [self.__exp(i - max_val) for i in x]
        sum_exp: float = sum(exp_values)
        return [i / sum_exp for i in exp_values]


class InitializationFunctions:
    """Класс содержит методы для инициализации весов нейронных сетей."""

    @staticmethod
    def get_uniform(value: float = 0.5) -> tuple[float, float]:
        """
        Возвращает границы диапазона для равномерной инициализации.
        :param value: Граница диапазона.
        :return: Кортеж с границами диапазона.
        """
        return -value, value

    @staticmethod
    def get_xavier(input_size: int, output_size: int) -> tuple[float, float]:
        """
        Функция инициализации Ксавьер.
        Вычисляется граница диапазона как квадратный корень из 6,
        деленного на сумму количества входных и выходных нейронов.
        :param input_size: Количество входных нейронов.
        :param output_size: Количество выходных нейронов.
        :return: Кортеж с границами диапазона для инициализации весов.
        """
        limit: float = (6 / (input_size + output_size)) ** 0.5
        return -limit, limit

    @staticmethod
    def get_he(input_size: int) -> tuple[float, float]:
        """
        Функция инициализации He.
        Вычисляется граница диапазона как квадратный корень из 2, деленного на количество входных нейронов.
        :param input_size: Количество входных нейронов.
        :return: Кортеж с границами диапазона для инициализации весов.
        """
        limit: float = (2 / input_size) ** 0.5
        return -limit, limit


class OtherFunctions:
    @staticmethod
    def calculate_average(value: list[int | float] | tuple[int | float]) -> int | float:
        """
        Вычисляет среднеарифметическое значение.

        :param value: Список входящих значений.
        :return: Среднеарифметическое значение.
        """
        return sum(value) / len(value)
