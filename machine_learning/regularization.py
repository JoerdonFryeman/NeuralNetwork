class Regularization:
    @staticmethod
    def _get_lasso_regularization(regularization: float, weights: list[list[float]], i: int, j: int) -> float:
        """
        Вычисляет Lasso регуляризацию для данного веса.
        Добавляет абсолютное значение величины коэффициентов как штраф к функции потерь.

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
        Добавляет квадрат величины коэффициентов как штраф к функции потерь.

        :param regularization: Параметр регуляризации.
        :param weights: Список весов.
        :param i: Индекс первой координаты веса.
        :param j: Индекс второй координаты веса.

        :return: Значение Ridge регуляризации.
        """
        return regularization * weights[i][j]
