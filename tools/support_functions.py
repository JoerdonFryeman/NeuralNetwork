class SupportFunctions:
    @staticmethod
    def calculate_average(value: list[float]) -> float:
        return sum(value) / len(value)

    @staticmethod
    def exp(x: float, terms: int = 20) -> float:
        result: float = 1.0
        factorial: float = 1.0
        power: float = 1.0

        for i in range(1, terms):
            factorial *= i
            power *= x
            result += power / factorial
        return result

    @staticmethod
    def get_relu_derivative(x: float) -> float:
        return 1.0 if x > 0 else 0.0

    @staticmethod
    def get_leaky_relu_derivative(x: float, alpha: float = 0.01) -> float:
        return 1.0 if x > 0 else alpha

    def get_elu_derivative(self, x: list[float], i: int, alpha=1.0) -> float:
        return 1 if x[i] > 0 else alpha * self.exp(x[i])

    @staticmethod
    def get_sigmoid_derivative(sigmoid, x: float) -> float:
        sigmoid_value = sigmoid(x)
        return sigmoid_value * (1 - sigmoid_value)

    @staticmethod
    def get_tanh_derivative(tanh, x: float) -> float:
        return 1 - tanh(x) ** 2


class ActivationFunctions(SupportFunctions):

    @staticmethod
    def get_relu(x: float) -> float:
        return max(0.0, x)

    @staticmethod
    def get_leaky_relu(x: float, alpha: float = 0.01) -> float:
        return x if x > 0 else alpha * x

    def get_elu(self, x: float, alpha: float = 1.0) -> float:
        return x if x >= 0 else alpha * (self.exp(x) - 1)

    @staticmethod
    def get_sigmoid(x: float) -> float:
        exp: float = 1.0

        for i in range(10, 0, -1):
            exp = 1 + x * exp / i
        return 1 / (1 + exp)

    @staticmethod
    def get_tanh(x: float) -> float:
        exp_pos_2x: float = 1.0
        exp_neg_2x: float = 1.0

        for i in range(1, 11):
            exp_pos_2x *= (2 * x) / i + 1
            exp_neg_2x *= (2 * -x) / i + 1

        return (exp_pos_2x - exp_neg_2x) / (exp_pos_2x + exp_neg_2x)


class NormalizationFunctions(SupportFunctions):

    @staticmethod
    def normalize_min_max(value: int, min_val: int | float, max_val: int | float) -> float:
        if min_val == max_val:
            return value / (value * value)
        return (value - min_val) / (max_val - min_val)

    def get_softmax(self, x: list[float]) -> list[float]:
        exp_values: list[float] = [self.exp(i - max(x)) for i in x]
        sum_exp: float = sum(exp_values)
        return [i / sum_exp for i in exp_values]


class InitializationFunctions:

    @staticmethod
    def get_uniform(value: float = 0.5) -> tuple[float, float]:
        return -value, value

    @staticmethod
    def get_xavier(input_size: int, output_size: int) -> tuple[float, float]:
        limit: float = (6 / (input_size + output_size)) ** 0.5
        return -limit, limit

    @staticmethod
    def get_he(input_size: int) -> tuple[float, float]:
        limit: float = (2 / input_size) ** 0.5
        return -limit, limit
