from support_functions import ActivationFunctions


class Visualisation(ActivationFunctions):
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
        if epoch % 100 == 0:
            print(
                f'Epoch: {epoch}, error: {calculate_error(prediction, target):.1f}%, '
                f'prediction: {prediction * 10:.4f}, result: {sum(layer.get_layer_dataset()):.4f}'
            )

    @staticmethod
    def get_train_layers_on_dataset_visualisation(data_number, output_layer):
        """
        Выводит визуальное представление результатов обучения для текущего набора данных.

        :param data_number: Номер данных.
        :param output_layer: Выходной слой.
        """
        print(
            f'\nОбучение грани куба {data_number} завершено, результат: '
            f'{sum(output_layer.get_layer_dataset()) * 10:.0f}\n'
        )

    @staticmethod
    def __print_visualisation(output_sum: float) -> None:
        """
        Выводит графическое представление результата и интерпретирует значение.

        :param output_sum: Сумма выходных данных.
        """
        dice_faces = {
            1: ["         ", "    ●    ", "         "],
            2: ["      ●  ", "         ", "  ●      "],
            3: ["      ●  ", "    ●    ", "  ●      "],
            4: ["  ●   ●  ", "         ", "  ●   ●  "],
            5: ["  ●   ●  ", "    ●    ", "  ●   ●  "],
            6: ["  ●   ●  ", "  ●   ●  ", "  ●   ●  "],
        }

        margin = 0.0004  # Приемлемое отклонение

        if abs(output_sum - 0.0329) < margin:
            face = 1
        elif abs(output_sum - 0.0309) < margin:
            face = 2
        elif abs(output_sum - 0.0514) < margin:
            face = 3
        elif abs(output_sum - 0.0591) < margin:
            face = 4
        elif abs(output_sum - 0.0648) < margin:
            face = 5
        elif abs(output_sum - 0.0304) < margin:
            face = 6
        else:
            print('Не могу интерпретировать значение результата!')
            return

        if face:
            print(f'\n┌{"─" * 9}┐')
            for line in dice_faces[face]:
                print(f'|{line}|')
            print(f'└{"─" * 9}┘')

    def get_visualisation(self, input_dataset: list[float], layers: dict[str, any]) -> None:
        """
        Выводит визуальное представление нейронной сети.

        :param input_dataset: Входной набор данных.
        :param layers: Словарь слоев сети, где ключ - имя слоя, а значение - объект слоя.
        """
        print(f'Класс: {self.__class__.__name__}')
        print(f'Всего слоёв: {len(layers)}')
        print(f'Количество входных данных: {len(input_dataset)}\n')

        for name, layer in layers.items():
            print(f'Слой: {name}')
            result = layer.get_layer_dataset()
            print(f'Данные слоя: {[float(f"{i:.2f}") for i in result]}\n')

            if name == 'hidden_layer_second':
                output_sum = self.get_sigmoid(sum(result))
                print(f'Выходные данные: {output_sum:.4f}')
                self.__print_visualisation(output_sum)
