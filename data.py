from configuration import get_json_data, save_json_data


class Data:
    """Класс предназначен для работы с массивами данных."""

    __slots__ = ('data_number', 'serial_class_number', 'serial_data_number', 'dataset')

    def __init__(self):
        """
        Инициализирует объекты класса с параметрами по умолчанию.

        :param data_number (str): Ключ-номер массива данных.
        :param serial_class_number (int): Начальный порядковый номер класса данных.
        :param serial_data_number (int): Начальный порядковый номер данных.
        :param dataset (dict): Загружаемый массив данных.
        """
        self.data_number: str = 'classes'
        self.serial_class_number: int = 1
        self.serial_data_number: int = 1
        self.dataset: dict = get_json_data('weights_biases_and_data', 'input_dataset')

    def get_data_dict_value(self, value_type: str) -> int:
        """
        В зависимости от флага возвращает количество данных или имя "класса" данных.

        :param value_type: Флаг "типа" данных.
        :return: Количество данных или имя "класса" данных.
        """
        if value_type == 'serial_data_number':
            return len(dict(enumerate(self.dataset[self.data_number].get(str(self.serial_class_number), []), 1)))
        elif value_type == 'serial_class_number':
            return len(dict(enumerate(self.dataset[self.data_number])).keys())
        else:
            raise ValueError(f'Неизвестный тип значений: {value_type}')

    def get_data_dict(self, class_name: int) -> dict:
        """
        Возвращает словарь данных.

        :param class_name: Порядковый номер класса данных.
        :return: Словарь с данными, где ключи - порядковые номера классов изображений.
        """
        return dict(enumerate(self.dataset[self.data_number].get(str(class_name), []), 1))

    def create_output_layer_data(self, output_layer: list[float], file_exist: bool = True) -> None:
        """
        Создаёт словарь с выходными данными.

        :param output_layer: Список выходных данных.
        :param file_exist: Флаг наличия или отсутствия файла выходных данных.
        """
        output_layer_data: dict[str, list[float]] = {}
        serial_class_number: int = self.get_data_dict_value('serial_class_number')
        # Прохождение по индексам от 0 до значения serial_class_number.
        for i in range(serial_class_number):
            # Для каждого индекса i создаётся ключ в словаре (строка от "1" до значения serial_class_number).
            # Далее он заполняется значениями, полученными с помощью среза.
            # output_data[i::serial_class_number] где выбираются элементы, начиная с i и шагом со значением serial_class_number.
            if file_exist:
                output_layer_data[str(i + 1)] = output_layer[i::serial_class_number]
            else:
                output_layer_data[str(i + 1)] = [0.0 * self.get_data_dict_value('serial_data_number')]
        save_json_data('weights_biases_and_data', 'output_layer_data', output_layer_data)

    def calculate_classification(self, output_layer: float, margin: float = float('inf')) -> int:
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
        except FileNotFoundError:
            self.create_output_layer_data([0.0 * self.get_data_dict_value('serial_data_number')], False)
        return serial_class_number

    def load_output_layer_data(self, init_network, training_mode: bool) -> None:
        """
        Загружает сохранённые выходные данные.

        :param init_network: Ссылка на функцию инициализации нейросети.
        :param training_mode: Флаг режима обучения.
        """
        output_layer_data: list[int | float] = []
        # В циклах, равных количеству данных в классе и количеству самих классов, загружаются выходные данные.
        for _ in range(self.get_data_dict_value('serial_data_number')):
            for _ in range(self.get_data_dict_value('serial_class_number')):
                result: int | float = init_network()
                output_layer_data.append(result)
                self.serial_class_number += 1
            self.serial_class_number: int = 1
            self.serial_data_number += 1
        # В режиме обучения запускается метод создания словаря входных данных.
        if training_mode:
            self.create_output_layer_data(output_layer_data)

    def get_data_sample(self, class_name: int, serial_data_number: int) -> any:
        """
        Возвращает данные для текущего изображения.

        :param class_name: Порядковый номер класса данных.
        :param serial_data_number: Номер данных, для которых нужно нормированное значение.

        :return: Данные для текущего изображения.
        """
        result = self.get_data_dict(class_name).get(serial_data_number)
        if result is None:
            raise ValueError(f'Номер {serial_data_number} или номер {class_name} за пределами диапазона!')
        return result

    def get_target_value_by_key(self, value_by_key: str) -> float:
        """
        Возвращает значение целевого объекта на основе ключа словаря.

        :param value_by_key: Ключ словаря данных.
        :return: Целевое значение целевого объекта.
        """
        target_values = {key: float(key) / 10 for key in self.dataset[self.data_number]}
        return target_values.get(value_by_key, 0.0)
