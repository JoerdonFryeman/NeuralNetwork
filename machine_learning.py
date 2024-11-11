from data import Data


class MachineLearning(Data):
    def __init__(self):
        """
        Инициализирует объект MachineLearning с начальными параметрами.
        """
        self.epochs = 100  # Количество эпох для обучения.
        self.learning_rate = 0.01  # Скорость обучения.
        self.error_tolerance = 0.01  # Допустимый уровень ошибки.
        self.regularization = 0.01  # Параметр регуляризации.
        self.lasso_regularization = False  # Использовать Lasso регуляризацию.
        self.ridge_regularization = True  # Использовать Ridge регуляризацию.

    @staticmethod
    def _calculate_error(predicted: float, target: float) -> float:
        """
        Вычисляет ошибку между предсказанным и целевым значениями.

        :param predicted: Предсказанное значение.
        :param target: Целевое значение.
        :return: Ошибка.
        """
        return 0.5 * (predicted - target) ** 2

    @staticmethod
    def _get_lasso_regularization(regularization: float, weights: list[list[float]], i: int, j: int) -> float:
        """
        Вычисляет Lasso регуляризацию для данного веса.

        :param regularization: Параметр регуляризации.
        :param weights: Список весов.
        :param i: Индекс первой координаты веса.
        :param j: Индекс второй координаты веса.
        :return: Значение Lasso регуляризации.
        """
        return regularization * (1 if weights[i][j] > 0 else -1)

    @staticmethod
    def _get_ridge_regularization(regularization: float, weights: list[list[float]], i: int, j: int) -> float:
        """
        Вычисляет Ridge регуляризацию для данного веса.

        :param regularization: Параметр регуляризации.
        :param weights: Список весов.
        :param i: Индекс первой координаты веса.
        :param j: Индекс второй координаты веса.
        :return: Значение Ridge регуляризации.
        """
        return regularization * weights[i][j]

    @staticmethod
    def _calculate_gradient_descent(
            weights: list[list[float]], i: int, j: int,
            learning_rate: float, gradient: float, input_dataset: list[float]
    ):
        """
        Обновляет вес с использованием градиентного спуска.

        :param weights: Список весов.
        :param i: Индекс первой координаты веса.
        :param j: Индекс второй координаты веса.
        :param learning_rate: Скорость обучения.
        :param gradient: Градиент.
        :param input_dataset: Входной набор данных.
        """
        weights[i][j] -= learning_rate * gradient * input_dataset[j]

    def update_weights(self, layer, gradient: float, lasso: bool, ridge: bool):
        """
        Обновляет веса слоя с использованием заданных параметров регуляризации.

        :param layer: Объект слоя.
        :param gradient: Градиент.
        :param lasso: Использовать Lasso регуляризацию.
        :param ridge: Использовать Ridge регуляризацию.
        """
        for i in range(len(layer.weights)):
            for j in range(len(layer.weights[i])):
                regularization_term = 0
                if lasso:
                    regularization_term += self._get_lasso_regularization(self.regularization, layer.weights, i, j)
                if ridge:
                    regularization_term += self._get_ridge_regularization(self.regularization, layer.weights, i, j)
                self._calculate_gradient_descent(
                    layer.weights, i, j, self.learning_rate, gradient + regularization_term, layer.input_dataset
                )
                if not lasso and not ridge:
                    self._calculate_gradient_descent(
                        layer.weights, i, j, self.learning_rate, gradient, layer.input_dataset
                    )
        layer.bias -= self.learning_rate * gradient

    def train(self, layer, data_number: int):
        """
        Обучает слой на основании данных.

        :param layer: Объект слоя.
        :param data_number: Номер данных.
        """
        for epoch in range(self.epochs):
            prediction, target = sum(layer.get_layer_dataset()), self.get_normalized_target_value(data_number)
            gradient = prediction - target
            self.update_weights(layer, gradient, self.lasso_regularization, self.ridge_regularization)
            if epoch % 5 == 0:
                print(
                    f'Epoch: {epoch}, error: {self._calculate_error(prediction, target):.32f}, '
                    f'prediction: {prediction * 10:.4f}, result: {sum(layer.get_layer_dataset()):.4f}'
                )
            if abs(prediction - target) < self.error_tolerance:
                return layer.weights, layer.bias

    def train_layers_on_dataset(
            self,
            data_number: int,
            input_outer_layer,
            hidden_layer_first,
            hidden_layer_second,
            output_outer_layer
    ):
        """
        Обучает несколько слоев на наборе данных.

        :param data_number: Номер данных.
        :param input_outer_layer: Внешний входной слой.
        :param hidden_layer_first: Первый скрытый слой.
        :param hidden_layer_second: Второй скрытый слой.
        :param output_outer_layer: Выходной внешний слой.
        """
        for i in range(len(self.dataset[self.data_name])):
            self.train(input_outer_layer, data_number)
            self.train(hidden_layer_first, data_number)
            self.train(hidden_layer_second, data_number)
            self.train(output_outer_layer, data_number)
            print(
                f'Обучение грани куба {data_number} завершено, результат: '
                f'{sum(output_outer_layer.get_layer_dataset()) * 10:.0f}\n'
            )
            data_number += 1
