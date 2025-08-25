import os
import pickle

from base.base import logger


class Weights:
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

    @staticmethod
    def _update_weights(
            layer, losses: float, lasso_regularization: bool,
            ridge_regularization: bool, learning_rate: float, regularization: float
    ) -> None:
        """
        Обновляет веса слоя с использованием заданных параметров Elastic Net регуляризации.

        :param layer: Объект слоя.
        :param losses: Результат работы функции потерь.
        :param lasso_regularization: Использовать Lasso регуляризацию.
        :param ridge_regularization: Использовать Ridge регуляризацию.
        :param learning_rate: Скорость обучения.
        :param regularization: Параметр регуляризации.
        """
        for i in range(len(layer.weights)):
            for j in range(len(layer.weights[i])):
                regularization_term: float = 0.0

                if lasso_regularization:
                    regularization_term += regularization * (1 if layer.weights[i][j] > 0 else -1)
                if ridge_regularization:
                    regularization_term += regularization * layer.weights[i][j]

                if lasso_regularization or ridge_regularization:
                    layer.weights[i][j] -= learning_rate * (losses * layer.input_dataset[j] + regularization_term)
                else:
                    layer.weights[i][j] -= learning_rate * losses * layer.input_dataset[j]

        layer.bias -= learning_rate * losses
