from config_files.configuration import get_json_data
from data import Data
from support_functions import ActivationFunctions


class Visualisation(ActivationFunctions, Data):
    """Класс предоставляет функции для визуализации процесса обучения и результатов работы нейронной сети."""

    @staticmethod
    def get_train_visualisation(epoch, calculate_error, prediction, target, layer):
        """
        Выводит визуализацию процесса обучения.

        :param calculate_error: Метод вычисления ошибки.
        :param epoch: Эпоха.
        :param prediction: Предсказанное значение.
        :param target: Целевое значение.
        :param layer: Объект слоя.
        """
        if epoch % 50 == 0:
            print(
                f'Эпоха: {epoch}, ошибка: {calculate_error(prediction, target):.1f}%, '
                f'прогноз: {prediction * 10:.4f}, результат: {sum(layer.get_layer_dataset()):.4f}'
            )

    @staticmethod
    def get_train_layers_on_dataset_visualisation(data_class_name, output_layer):
        """
        Выводит визуальное представление результатов обучения для текущего набора данных.

        :param data_class_name: Порядковый номер класса данных.
        :param output_layer: Выходной слой.
        """
        print(
            f'\nОбучение класса данных {data_class_name} завершено, результат: '
            f'{sum(output_layer.get_layer_dataset()) * 10:.0f}\n'
        )

    def _calculate_classification(self, output_sum: float, results: dict, margin: float = float('inf')) -> int:
        """Находит значение, наиболее близкое к output_sum с учётом margin"""
        data_class_name, min_difference = None, margin
        for name, result in results.items():
            difference = abs(output_sum - result[self.data_number - 1])
            if difference < min_difference:
                min_difference = difference
                data_class_name = name
        return data_class_name

    def _get_classification(self, output_sum: float):
        """Хранит словарь со значениями выходных данных каждого класса."""
        averages = {
            1: [0.0000, 0.0000],
            2: [0.0000, 0.0000],
            3: [0.0000, 0.0000],
            4: [0.0000, 0.0000],
            5: [0.0000, 0.0000],
            6: [0.0000, 0.0000],
        }
        return self._calculate_classification(output_sum, averages)

    def _print_visualisation(self, output_sum: float) -> None:
        """
        Выводит графическое представление результата и интерпретирует значение.

        :param output_sum: Сумма выходных данных.
        """
        data_class_name = self._get_classification(output_sum)
        horizontal_line_first, horizontal_line_second = 54, 20
        if data_class_name is not None:
            print(f'\n┌{"─" * horizontal_line_first}┐')
            print(f'│{" " * horizontal_line_second}{" " * 14}{" " * horizontal_line_second}│')
            for number in get_json_data('numbers')[str(data_class_name)]:
                print(f'│{" " * horizontal_line_second}{number}{" " * horizontal_line_second}│')
            print(f'│{" " * horizontal_line_second}{" " * 14}{" " * horizontal_line_second}│')
            print(f'└{"─" * horizontal_line_first}┘')
        else:
            print('Не могу интерпретировать значение результата!')

    def get_visualisation(self, input_dataset: list[float], layers: dict[str, any], output_layer: float) -> None:
        """
        Выводит визуальное представление нейронной сети.

        :param input_dataset: Входной набор данных.
        :param layers: Словарь слоев сети, где ключ - имя слоя, а значение - объект слоя.
        :param output_layer: Выходные данные.
        """
        print(f'Класс: {self.__class__.__name__}')
        print(f'Всего слоёв: {len(layers)}')
        print(f'Количество входных данных: {len(input_dataset)}\n')

        for name, layer in layers.items():
            print(f'Слой: {name}')
            result = layer.get_layer_dataset()
            print(f'Данные слоя: {[float(f"{i:.2f}") for i in result]}\n')

        print(f'Выходные данные: {output_layer:.4f}')
        self._print_visualisation(float(f'{output_layer:.4f}'))
