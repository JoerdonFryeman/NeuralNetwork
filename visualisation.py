class Visualisation:
    @staticmethod
    def __print_visualisation(output_sum: float) -> None:
        """
        Выводит графическое представление результата.

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

        if output_sum < 0.5:
            face = 1
        elif 0.5 < output_sum < 0.9:
            face = 2
        elif 0.9 < output_sum < 1.2:
            face = 3
        elif 1.25 < output_sum < 1.26:
            face = 5
        elif 1.27 < output_sum < 1.28:
            face = 4
        elif 2.8 < output_sum < 3.0:
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

            if name == 'output_outer_layer':
                output_sum = float(sum(result))
                print(f'Выходные данные: {output_sum:.4f}')
                self.__print_visualisation(output_sum)
