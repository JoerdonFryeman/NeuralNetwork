from config_files.configuration import get_json_data
from data.classification import Classification
from games.rps import RPS


class Interpretation(RPS, Classification):
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
        if data_class_name in ['1', '2', '3']:
            print(get_json_data('config_files/ascii_arts', 'answer')['yes'])
        elif data_class_name == '4':
            print(get_json_data('config_files/ascii_arts', 'answer')['maybe'])
        else:
            print(get_json_data('config_files/ascii_arts', 'answer')['no'])

    def get_rock_paper_scissors(self, data_class_name: str):
        """
        Определяет и выводит ответ нейронной сети для игры "Камень, ножницы, бумага".

        :param data_class_name: Название класса данных, определяющее, какой ответ будет выведен.

        :raises Exception: Если значение параметра data_class_name не соответствует ожидаемым значениям.
        :return: Возвращает строку, представляющую выбранное действие ("Бумага", "Камень" или "Ножницы").
        """
        if data_class_name == '1':
            print(f'Нейронная сеть отвечает: {get_json_data('config_files/ascii_arts', 'rps')['paper']}')
            self.run_rps('Бумага')
            return 'Бумага'
        elif data_class_name == '2':
            print(f'Нейронная сеть отвечает: {get_json_data('config_files/ascii_arts', 'rps')['rock']}')
            self.run_rps('Камень')
            return 'Камень'
        elif data_class_name == '3':
            print(f'Нейронная сеть отвечает: {get_json_data('config_files/ascii_arts', 'rps')['scissors']}')
            self.run_rps('Ножницы')
            return 'Ножницы'
        else:
            raise Exception('Не могу визуализировать значение результата!')

    def get_interpretation(self, output_layer: float, func: callable) -> None:
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
