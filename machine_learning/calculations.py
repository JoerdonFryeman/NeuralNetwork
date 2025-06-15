class Calculations:
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
    def _calculate_gradient_descent(
            input_dataset: list[float], learning_rate: float,
            gradient: float, weights: list[list[float]], i: int, j: int
    ) -> None:
        """
        Обновляет вес с использованием градиентного спуска.

        :param input_dataset: Входной набор данных.
        :param learning_rate: Скорость обучения.
        :param gradient: Градиент.
        :param weights: Список весов.
        :param i: Индекс первой координаты веса.
        :param j: Индекс второй координаты веса.
        """
        weights[i][j] -= learning_rate * gradient * input_dataset[j]

    @staticmethod
    def _calculate_learning_decay(epoch: int, epochs: int, learning_rate: float, learning_decay: float) -> float:
        """
        Вычисляет уменьшение скорости обучения (learning_rate) в зависимости от текущей эпохи.

        :param epoch: Текущая эпоха.
        :param epochs: Общее количество эпох.
        :param learning_rate: Текущая скорость обучения.
        :param learning_decay: Коэффициент уменьшения скорости обучения.

        :return: Обновленная скорость обучения.
        """
        if epoch % (epochs // 4) == 0 and epoch != 0:
            learning_rate *= learning_decay
        return learning_rate
