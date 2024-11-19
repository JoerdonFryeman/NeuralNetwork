import pickle
from configuration import logger
from data import Data


class MachineLearning(Data):
    def __init__(self):
        """
        Инициализирует объект MachineLearning с начальными параметрами.
        Атрибуты:
        - epochs (int): Количество эпох для обучения.
        - learning_rate (float): Скорость обучения.
        - error_tolerance (float): Допустимый уровень ошибки.
        - regularization (float): Параметр регуляризации.
        - lasso_regularization (bool): Использовать Lasso регуляризацию.
        - ridge_regularization (bool): Использовать Ridge регуляризацию.
        """
        self.epochs: int = 1000
        self.learning_rate: float = 0.001
        self.error_tolerance: float = 0.001
        self.regularization: float = 0.001
        self.lasso_regularization: bool = False
        self.ridge_regularization: bool = True

    @staticmethod
    def _save_weights_and_biases(weights: dict[str, list[list[float]]], biases: dict[str, list[float]]) -> None:
        """
        Сохраняет веса и смещения в файл.

        :param weights: Словарь весов, где ключи - имена слоев, значения - веса слоев.
        :param biases: Словарь смещений, где ключи - имена слоев, значения - смещения слоев.
        """
        data: dict = {'weights': weights, 'biases': biases}
        try:
            with open('weights_and_biases.pkl', 'wb') as file:
                pickle.dump(data, file)
            logger.info('Данные успешно сохранены!')
        except Exception as e:
            logger.error(f'Произошла ошибка: {e}')

    @staticmethod
    def _calculate_error(predicted: float, target: float) -> float:
        """
        Вычисляет ошибку между предсказанным и целевым значениями.

        :param predicted: Предсказанное значение.
        :param target: Целевое значение.
        :return: Ошибка в процентном соотношении.
        """
        return ((predicted - target) / target) * 100

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
    ) -> None:
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

    def update_weights(self, layer, gradient: float, lasso: bool, ridge: bool) -> None:
        """
        Обновляет веса слоя с использованием заданных параметров Elastic Net регуляризации.

        :param layer: Объект слоя.
        :param gradient: Градиент.
        :param lasso: Использовать Lasso регуляризацию.
        :param ridge: Использовать Ridge регуляризацию.
        """
        for i in range(len(layer.weights)):
            for j in range(len(layer.weights[i])):
                regularization_term: float = 0.0
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

    def __get_train_visualisation(self, epoch, prediction, target, layer):
        """
        Выводит визуализацию процесса обучения.

        :param epoch: Эпоха.
        :param prediction: Предсказанное значение.
        :param target: Целевое значение.
        :param layer: Объект слоя.
        """
        if epoch % 10 == 0:
            print(
                f'Epoch: {epoch}, error: {self._calculate_error(prediction, target):.1f}%, '
                f'prediction: {prediction * 10:.4f}, result: {sum(layer.get_layer_dataset()):.4f}'
            )

    def train(self, layer, data_number: int) -> tuple[list[list[float]], float]:
        """
        Обучает слой на основании данных.

        :param layer: Объект слоя.
        :param data_number: Номер данных.
        :return: Кортеж с обновленными весами и смещением (bias) слоя.
        """
        for epoch in range(self.epochs):
            layer.input_dataset = self.get_data_sample()
            prediction: float = sum(layer.get_layer_dataset())
            target: float = self.get_normalized_target_value(data_number)
            gradient: float = prediction - target
            self.update_weights(layer, gradient, self.lasso_regularization, self.ridge_regularization)
            self.__get_train_visualisation(epoch, prediction, target, layer)
            if abs(prediction - target) < self.error_tolerance:
                return layer.weights, layer.bias

    @staticmethod
    def __get_train_layers_on_dataset_visualisation(data_number, output_outer_layer):
        """
        Выводит визуальное представление результатов обучения для текущего набора данных.

        :param data_number: Номер данных.
        :param output_outer_layer: Выходной слой.
        """
        print(
            f'\nОбучение грани куба {data_number} завершено, результат: '
            f'{sum(output_outer_layer.get_layer_dataset()) * 10:.0f}\n'
        )

    def train_layers_on_dataset(
            self, data_number: int, hidden_layer_first, hidden_layer_second, output_outer_layer
    ) -> None:
        """
        Обучает несколько слоев на наборе данных.

        :param data_number: Номер данных.
        :param hidden_layer_first: Первый скрытый слой.
        :param hidden_layer_second: Второй скрытый слой.
        :param output_outer_layer: Выходной слой.
        """
        weights: dict[str, list[list[float]]] = {}
        biases: dict[str, list[float]] = {}

        for i in range(len(self.dataset[self.data_name])):
            hidden_layer_first.input_dataset = self.get_data_sample()
            self.train(hidden_layer_first, data_number)

            hidden_layer_second.input_dataset = hidden_layer_first.get_layer_dataset()
            self.train(hidden_layer_second, data_number)

            output_outer_layer.input_dataset = hidden_layer_second.get_layer_dataset()
            self.train(output_outer_layer, data_number)

            self.__get_train_layers_on_dataset_visualisation(data_number, output_outer_layer)
            data_number += 1

        weights['hidden_layer_first'] = hidden_layer_first.weights
        weights['hidden_layer_second'] = hidden_layer_second.weights
        weights['output_outer_layer'] = output_outer_layer.weights

        biases['hidden_layer_first'] = hidden_layer_first.bias
        biases['hidden_layer_second'] = hidden_layer_second.bias
        biases['output_outer_layer'] = output_outer_layer.bias

        self._save_weights_and_biases(weights, biases)
