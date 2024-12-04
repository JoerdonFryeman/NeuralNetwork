from config_files.configuration import get_json_data
from support_functions import ActivationFunctions, OtherFunctions


class Visualisation(ActivationFunctions, OtherFunctions):
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
    def get_train_layers_on_dataset_visualisation(data_number, output_layer):
        """
        Выводит визуальное представление результатов обучения для текущего набора данных.

        :param data_number: Номер данных.
        :param output_layer: Выходной слой.
        """
        print(
            f'\nОбучение класса данных {data_number} завершено, результат: '
            f'{sum(output_layer.get_layer_dataset()) * 10:.0f}\n'
        )

    @staticmethod
    def _find_closest_average(output_sum: float, averages: dict[int, float], margin: float = float('inf')) -> int:
        """Находит среднее значение, наиболее близкое к output_sum с учётом margin"""
        data_class_name, min_difference = None, margin
        for name, average in averages.items():
            difference = abs(output_sum - average)
            if difference < min_difference:
                min_difference = difference
                data_class_name = name
        return data_class_name

    def _calculate_classes_average(self, output_sum: float):
        """Вычисляет среднеарифметическое значение для каждого класса данных."""
        averages = {
            1: self.calculate_average([0.00, 0.00]),
            2: self.calculate_average([0.00, 0.00]),
            3: self.calculate_average([0.00, 0.00]),
            4: self.calculate_average([0.00, 0.00]),
            5: self.calculate_average([0.00, 0.00]),
            6: self.calculate_average([0.00, 0.00]),
        }
        return self._find_closest_average(output_sum, averages)

    def _print_visualisation(self, output_sum: float) -> None:
        """
        Выводит графическое представление результата и интерпретирует значение.

        :param output_sum: Сумма выходных данных.
        """
        data_class_name = self._calculate_classes_average(output_sum)
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

        print(f'Выходные данные: {output_layer:.2f}')
        self._print_visualisation(float(f'{output_layer:.2f}'))
