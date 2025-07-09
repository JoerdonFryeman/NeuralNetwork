from data.data import Data
from tools.visualisation import Visualisation
from .weights import Weights


class Train(Visualisation, Weights, Data):
    """Класс отвечает за процесс обучения нейронной сети."""

    def _get_target(self, data_key: str) -> float:
        """
        Возвращает значение целевого объекта.

        :param data_key: Ключ словаря данных.

        :return: Целевое значение объекта.
        """
        dataset_and_target = self.dataset[self.data_name]
        target_values = {key: float(dataset_and_target[key][1]) / 10 for key in dataset_and_target}
        return target_values.get(data_key, 0.0)

    def _train(
            self, data_key: str, layer, epochs: int, learning_rate: float, learning_decay: float,
            error_tolerance: float, regularization: float, lasso_regularization: bool, ridge_regularization: bool
    ) -> tuple[list[list[float]], float]:
        """
        Обучает слой на основании данных.

        :param layer: Объект слоя.
        :param data_key: Ключ словаря данных.
        :param epochs: Количество эпох для обучения.
        :param learning_rate: Скорость обучения.
        :param learning_decay: Уменьшение скорости обучения.
        :param error_tolerance: Допустимый уровень ошибки.
        :param regularization: Параметр регуляризации.
        :param lasso_regularization: Использовать Lasso регуляризацию.
        :param ridge_regularization: Использовать Ridge регуляризацию.

        :return: Кортеж с обновленными весами и смещением (bias) слоя.
        """
        for epoch in range(epochs):
            layer.input_dataset = self.get_data_sample(self.serial_class_number, self.serial_data_number)
            prediction: float = sum(layer.get_layer_dataset())
            target: float = self._get_target(data_key)
            loss: float = prediction - target
            self._update_weights(
                layer, loss, lasso_regularization, ridge_regularization, learning_rate, regularization
            )
            self.get_train_visualisation(epoch, self._calculate_error, prediction, target, layer)
            learning_rate = self._calculate_learning_decay(epoch, epochs, learning_rate, learning_decay)
            if abs(prediction - target) < error_tolerance:
                return layer.weights, layer.bias
        return layer.weights, layer.bias

    def train_layers_on_dataset(
            self, hidden_layer_first, hidden_layer_second, epochs: int, learning_rate: float, learning_decay: float,
            error_tolerance: float, regularization: float, lasso_regularization: bool, ridge_regularization: bool
    ) -> None:
        """
        Обучает несколько слоев на наборе данных.

        :param hidden_layer_first: Первый скрытый слой.
        :param hidden_layer_second: Второй скрытый слой.
        :param epochs: Количество эпох для обучения.
        :param learning_rate: Скорость обучения.
        :param learning_decay: Уменьшение скорости обучения.
        :param error_tolerance: Допустимый уровень ошибки.
        :param regularization: Параметр регуляризации.
        :param lasso_regularization: Использовать Lasso регуляризацию.
        :param ridge_regularization: Использовать Ridge регуляризацию.
        """
        weights: dict[str, list[list[float]]] = {}
        biases: dict[str, list[float]] = {}

        for data_key, data_samples in self.dataset[self.data_name].items():
            for _ in data_samples[0]:
                hidden_layer_first.input_dataset = self.get_data_sample(
                    self.serial_class_number, self.serial_data_number
                )
                self._train(
                    data_key, hidden_layer_first, epochs, learning_rate, learning_decay,
                    error_tolerance, regularization, lasso_regularization, ridge_regularization
                )
                hidden_layer_second.input_dataset = hidden_layer_first.get_layer_dataset()
                self._train(
                    data_key, hidden_layer_second, epochs, learning_rate, learning_decay,
                    error_tolerance, regularization, lasso_regularization, ridge_regularization
                )
                self.get_train_layers_on_dataset_visualisation(data_key, hidden_layer_second)

        weights['hidden_layer_first'] = hidden_layer_first.weights
        weights['hidden_layer_second'] = hidden_layer_second.weights

        biases['hidden_layer_first'] = hidden_layer_first.bias
        biases['hidden_layer_second'] = hidden_layer_second.bias

        self._save_weights_and_biases('weights_biases_and_data', 'weights_and_biases', weights, biases)
