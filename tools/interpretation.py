from base.base import get_json_data
from data.classification import Classification


class Interpretation(Classification):
    """Класс для интерпретации результатов нейронной сети."""

    @staticmethod
    def get_number_visualisation(data_class_name: str) -> None:
        """
        Выводит визуальное представление чисел для заданного класса данных.

        :param data_class_name: Название класса данных, для которого необходимо вывести визуализацию чисел.
        """
        horizontal_line_first: int = 54
        horizontal_line_second: int = 20

        print(f'┌{"─" * horizontal_line_first}┐')
        print(f'│{" " * horizontal_line_second}{" " * 14}{" " * horizontal_line_second}│')
        for number in get_json_data('config_files/ascii_arts', 'numbers')[data_class_name]:
            print(f'│{" " * horizontal_line_second}{number}{" " * horizontal_line_second}│')
        print(f'│{" " * horizontal_line_second}{" " * 14}{" " * horizontal_line_second}│')
        print(f'└{"─" * horizontal_line_first}┘')

    @staticmethod
    def get_dice_visualization(data_class_name: str) -> None:
        """
        Выводит визуальное представление грани кубика для заданного класса данных.

        :param data_class_name: Название класса данных, представляющее номер грани кубика.

        :raises Exception: Если значение параметра data_class_name не находится в диапазоне от 1 до 6.
        """
        horizontal_line: int = 9

        if 1 <= int(data_class_name) <= 6:
            print(f'┌{"─" * horizontal_line}┐')
            for number in get_json_data('config_files/ascii_arts', 'dice')[data_class_name]:
                print(f'│{number}│')
            print(f'└{"─" * horizontal_line}┘')
        else:
            raise Exception('Не могу визуализировать значение результата!')

    @staticmethod
    def get_answer(data_class_name: str) -> None:
        """
        Выводит ответ в виде ASCII-арта в зависимости от класса данных.

        :param data_class_name: Название класса данных, определяющее, какой ответ будет выведен.
        """
        data_number = len(get_json_data('weights_biases_and_data', 'output_layer_data')) // 3

        if data_class_name in [str(i) for i in range(1, data_number + 1)]:
            print(f"\n{get_json_data('config_files/ascii_arts', 'answer')['yes']}")
        elif data_class_name in [str(i) for i in range(data_number + 1, data_number * 2 + 1)]:
            print(f"\n{get_json_data('config_files/ascii_arts', 'answer')['hmm']}")
        elif data_class_name in [str(i) for i in range(data_number * 2 + 1, data_number * 3 + 1)]:
            print(f"\n{get_json_data('config_files/ascii_arts', 'answer')['no']}")
        else:
            print(f"\n{get_json_data('config_files/ascii_arts', 'answer')['sorry']}")

    def get_interpretation(self, output_layer: float, func) -> None:
        """
        Интерпретирует значение результата и вызывает переданную функцию.

        :param output_layer: Значение выходного слоя, которое необходимо интерпретировать.
        :param func: Функция, которая будет вызвана для интерпретации результата.

        :raises Exception: Если значение результата не может быть интерпретировано, выводится сообщение об ошибке.
        """
        data_class_name: str = self.calculate_classification(output_layer)
        if data_class_name is not None:
            func(data_class_name)
        else:
            print('Не могу интерпретировать значение результата!')
