class ActivationFunctions:
    @staticmethod
    def get_linear(x):
        """
        Линейная активационная функция.
        :param x: Входное значение.
        :return: То же входное значение.
        """
        return x

    @staticmethod
    def get_relu(x):
        """
        ReLU (Rectified Linear Unit) активационная функция.
        :param x: Входное значение.
        :return: Максимум между нулем и входным значением.
        """
        return max(0, x)

    @staticmethod
    def get_sigmoid(x):
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
    def get_tanh(x):
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

    @staticmethod
    def get_leaky_relu(x, alpha=0.01):
        """
        Leaky ReLU активационная функция.
        :param x: Входное значение.
        :param alpha: Сила утечки (по умолчанию 0.01).
        :return: Максимум между alpha*x и входным значением.
        """
        return x if x > 0 else alpha * x

    @staticmethod
    def __exp(x, terms=20):
        """
        Вычисляет экспоненту числа x с помощью ряда Тейлора.
        :param x: Входное значение.
        :param terms: Количество членов ряда (по умолчанию 20).
        :return: Значение exp(x).
        """
        result = 1.0
        factorial = 1.0
        power = 1.0
        for i in range(1, terms):
            factorial *= i
            power *= x
            result += power / factorial
        return result

    def get_elu(self, x, alpha=1.0):
        """
        ELU (Exponential Linear Unit) активационная функция.
        :param x: Входное значение.
        :param alpha: Параметр альфа (по умолчанию 1.0).
        :return: ELU от входного значения.
        """
        return x if x >= 0 else alpha * (self.__exp(x) - 1)

    def get_softmax(self, x):
        """
        Softmax активационная функция.
        :param x: Входное значение (может быть списком или отдельным числом).
        :return: Softmax распределение значений.
        """
        if isinstance(x, float) or isinstance(x, int):
            x = [x]

        exp_values = [self.__exp(i) for i in x]
        sum_exp = sum(exp_values)
        return [i / sum_exp for i in exp_values]
