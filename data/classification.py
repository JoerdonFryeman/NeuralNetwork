from config_files.configuration import get_json_data, logger


class Classification:
    @staticmethod
    def calculate_classification(output_layer: float, margin: float = float('inf')) -> int:
        """
        Находит наиболее близкое к output_sum с учётом margin значение.

        :param output_layer: Выходное значение.
        :param margin: Минимальная разность.

        :return: Имя "класса" данных в виде порядкового номера.
        """
        serial_class_number = None
        min_difference: float = float(f'{margin:.10f}')
        try:
            results: dict = get_json_data('weights_biases_and_data', 'output_layer_data')
            # Прохождение в цикле по всем ключам и значениям словаря results.
            for name, result in results.items():
                for value in result:
                    # Вычисляется абсолютная разность.
                    difference: float = abs(output_layer - value)
                    # Условие для обновления ближайшего класса.
                    if difference < min_difference:
                        min_difference = difference
                        # Обновляется ближайший класс.
                        serial_class_number = name
        except FileNotFoundError as e:
            logger.error(f'Произошла ошибка: {e}')
        return serial_class_number
