from configuration import get_json_data
from data import Data
from support_functions import ActivationFunctions


class Visualisation(ActivationFunctions, Data):
    """Класс содержит методы визуализации процесса обучения и результатов работы нейронной сети."""

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
    def get_train_layers_on_dataset_visualisation(data_class_name: int, output_layer):
        """
        Выводит визуальное представление результатов обучения для текущего набора данных.

        :param data_class_name: Порядковый номер класса данных.
        :param output_layer: Выходные данные.
        """
        print(
            f'\nОбучение класса данных {data_class_name} завершено, результат: '
            f'{sum(output_layer.get_layer_dataset()) * 10:.0f}\n'
        )

    def _get_number_visualisation(self, output_layer: float) -> None:
        """
        Выводит графическое представление результата.

        :param output_layer: Выходные данные.
        """
        data_class_name: int = self.calculate_classification(output_layer)
        horizontal_line_first: int = 54
        horizontal_line_second: int = 20

        if data_class_name is not None:
            print(f'┌{"─" * horizontal_line_first}┐')
            print(f'│{" " * horizontal_line_second}{" " * 14}{" " * horizontal_line_second}│')
            for number in get_json_data('config_files', 'numbers')[str(data_class_name)]:
                print(f'│{" " * horizontal_line_second}{number}{" " * horizontal_line_second}│')
            print(f'│{" " * horizontal_line_second}{" " * 14}{" " * horizontal_line_second}│')
            print(f'└{"─" * horizontal_line_first}┘')
        else:
            print('Не могу интерпретировать значение результата!')

    def get_info_visualisation(
            self, input_dataset: list[int | float], layers: dict[str, any], output_layer: float
    ) -> None:
        """
        Выводит визуальное представление нейронной сети и результат.

        :param input_dataset: Входной набор данных.
        :param layers: Словарь слоев сети, где ключ - имя слоя, а значение - объект слоя.
        :param output_layer: Выходные данные.
        """
        print(f'Всего слоёв: {len(layers) + 1}')
        print(f'Количество классов данных: {self.get_data_dict_value('data_class_name')}')
        print(f'Количество данных в каждом классе: {self.get_data_dict_value('data_number')}\n')

        print(f'Количество входных данных: {len(input_dataset)}\n')

        for name, layer in layers.items():
            print(f'Слой: {name}')
            result = layer.get_layer_dataset()
            print(f'Данные слоя: {[float(f"{i:.2f}") for i in result]}\n')

        print(f'Слой: output_layer\nДанные: {output_layer:.10f}\n')
        print('Интерпретация данных:\n')
        self._get_number_visualisation(float(f'{output_layer:.10f}'))
