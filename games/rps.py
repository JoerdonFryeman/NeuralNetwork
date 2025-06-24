from random import choice

from config_files.base import get_json_data, save_json_data


class RPS:
    """Класс для реализации игры "Камень, ножницы, бумага" с использованием нейронной сети."""

    rps_dict = {
        1: [[1.0, 0.0, 0.0], 'Камень'], 2: [[0.0, 1.0, 0.0], 'Ножницы'], 3: [[0.0, 0.0, 1.0], 'Бумага'],
        4: [[1.0, 0.0, 0.0], 'Ножницы'], 5: [[0.0, 1.0, 0.0], 'Бумага'], 6: [[0.0, 0.0, 1.0], 'Камень'],
        7: [[1.0, 0.0, 0.0], 'Бумага'], 8: [[0.0, 1.0, 0.0], 'Камень'], 9: [[0.0, 0.0, 1.0], 'Ножницы']
    }

    result = {1: 'Ничья!', 2: 'Нейросеть проиграла!', 3: 'Нейросеть выиграла!'}

    @staticmethod
    def get_computer_choice() -> list[float]:
        """
        Генерирует случайный выбор компьютера для игры "Камень, ножницы, бумага".

        :return: Возвращает список, представляющий выбор компьютера.
        """
        computer = choice(([1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]))
        if computer == [1.0, 0.0, 0.0]:
            print(f'\nОтвет компьютера: {get_json_data('config_files/ascii_arts', 'rps')['rock']}')
        elif computer == [0.0, 1.0, 0.0]:
            print(f'\nОтвет компьютера: {get_json_data('config_files/ascii_arts', 'rps')['scissors']}')
        elif computer == [0.0, 0.0, 1.0]:
            print(f'\nОтвет компьютера: {get_json_data('config_files/ascii_arts', 'rps')['paper']}')
        save_json_data('weights_biases_and_data', 'input_dataset', computer)
        return computer

    @staticmethod
    def get_result(computer: dict, neural_network: str, result: str, *args) -> None:
        """
        Определяет и выводит результат игры на основе выбора компьютера и нейронной сети.
        Метод сравнивает выбор компьютера и результат нейронной сети с переданными аргументами.

        :param computer: Словарь, содержащий информацию о выборе компьютера. Ожидается,
        :param neural_network: Строка, представляющая результат нейронной сети.
        :param result: Строка, представляющая результат, который будет выведен, если условия совпадения выполнены.
        :param args: Дополнительные аргументы.

        :return: Возвращает None.
        """
        if computer['classes']['1'][0][0] == args[0] and neural_network == args[1]:
            print(result)
        return None

    def run_rps(self, neural_network: str) -> None:
        """
        Запускает игру "Камень, ножницы, бумага" с использованием нейронной сети.

        Метод загружает выбор компьютера из файла и сравнивает его с результатами
        нейронной сети для всех возможных комбинаций действий. Если совпадения
        обнаружены, выводится соответствующий результат.

        :param neural_network: Строка, представляющая результат нейронной сети.

        :return: Возвращает None.
        """
        computer = get_json_data('weights_biases_and_data', 'input_dataset')
        counter = 1
        for i in range(1, 4):
            for j in range(1, 4):
                self.get_result(computer, neural_network, self.result[i], *self.rps_dict[counter])
                counter += 1
