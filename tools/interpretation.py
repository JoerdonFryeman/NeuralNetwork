from config_files.configuration import get_json_data
from data.classification import Classification


class Interpretation(Classification):
    @staticmethod
    def get_number_visualisation(data_class_name: str) -> None:
        horizontal_line_first: int = 54
        horizontal_line_second: int = 20

        print(f'┌{"─" * horizontal_line_first}┐')
        print(f'│{" " * horizontal_line_second}{" " * 14}{" " * horizontal_line_second}│')
        for number in get_json_data('config_files', 'numbers')[data_class_name]:
            print(f'│{" " * horizontal_line_second}{number}{" " * horizontal_line_second}│')
        print(f'│{" " * horizontal_line_second}{" " * 14}{" " * horizontal_line_second}│')
        print(f'└{"─" * horizontal_line_first}┘')

    @staticmethod
    def get_dice_visualization(data_class_name: str) -> None:
        horizontal_line: int = 9

        if 1 <= int(data_class_name) <= 6:
            print(f'┌{"─" * horizontal_line}┐')
            for number in get_json_data('config_files', 'dice')[data_class_name]:
                print(f'│{number}│')
            print(f'└{"─" * horizontal_line}┘')
        else:
            print('Не могу визуализировать значение результата!')

    @staticmethod
    def get_answer(data_class_name: str) -> None:
        if data_class_name in ['1', '2', '3']:
            print('Да!')
        elif data_class_name == '4':
            print('Возможно...')
        else:
            print('Нет!')

    def get_interpretation(self, output_layer: float, func: callable) -> None:
        data_class_name: str = self.calculate_classification(output_layer)
        if data_class_name is not None:
            func(data_class_name)
        else:
            print('Не могу интерпретировать значение результата!')
