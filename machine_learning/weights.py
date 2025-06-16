import os
import pickle

from config_files.configuration import logger
from .regularization import Regularization
from .calculations import Calculations


class Weights(Calculations, Regularization):
    @staticmethod
    def _save_weights_and_biases(
            directory: str, name: str, weights: dict[str, list[list[float]]], biases: dict[str, list[float]]
    ) -> None:
        """
        Сохраняет веса и смещения в файл.

        :param directory: Название каталога.
        :param name: Имя файла, в который будут загружены веса и смещения.
        :param weights: Словарь весов, где ключи - имена слоев, значения - веса слоев.
        :param biases: Словарь смещений, где ключи - имена слоев, значения - смещения слоев.
        """
        data: dict = {'weights': weights, 'biases': biases}
        try:
            try:
                os.mkdir(directory)
            except FileExistsError:
                pass
            with open(f'{directory}/{name}.pkl', 'wb') as file:
                pickle.dump(data, file)
            logger.info('Данные успешно сохранены!')
        except Exception as e:
            logger.error(f'Произошла ошибка: {e}')

    def _update_weights(
            self, layer, gradient: float, lasso_regularization: bool,
            ridge_regularization: bool, learning_rate: float, regularization: float
    ) -> None:
        """
        Обновляет веса слоя с использованием заданных параметров Elastic Net регуляризации.

        :param layer: Объект слоя.
        :param gradient: Градиент.
        :param lasso_regularization: Использовать Lasso регуляризацию.
        :param ridge_regularization: Использовать Ridge регуляризацию.
        :param learning_rate: Скорость обучения.
        :param regularization: Параметр регуляризации.
        """
        for i in range(len(layer.weights)):
            for j in range(len(layer.weights[i])):
                regularization_term: float = 0.0
                if lasso_regularization:
                    regularization_term += self._get_lasso_regularization(regularization, layer.weights, i, j)
                if ridge_regularization:
                    regularization_term += self._get_ridge_regularization(regularization, layer.weights, i, j)
                self._calculate_gradient_descent(
                    layer.input_dataset, learning_rate, gradient + regularization_term, layer.weights, i, j
                )
                if not lasso_regularization and not ridge_regularization:
                    self._calculate_gradient_descent(
                        layer.input_dataset, learning_rate, gradient, layer.weights, i, j
                    )
        layer.bias -= learning_rate * gradient
