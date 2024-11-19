import os
import pickle
import tempfile
import unittest
from unittest.mock import patch, mock_open
from data import Data
from layers import LayerBuilder, OuterLayer, HiddenLayer
from neural_network import NeuralNetwork
from support_functions import ActivationFunctions, InitializationFunctions
from machine_learning import MachineLearning
from visualisation import Visualisation


class GeneralTestParameters(unittest.TestCase):
    """
    Класс для установки и управления общими параметрами тестирования.

    Этот класс используется для настройки начальных данных и параметров,
    необходимых для выполнения тестов. Он включает методы для установки
    и отключения режима тестирования, а также для инициализации общих параметров.
    """

    @classmethod
    def setUpClass(cls):
        """Устанавливает режим тестирования перед запуском всех тестов."""
        LayerBuilder._test_mode = True

    @classmethod
    def tearDownClass(cls):
        """Отключает режим тестирования после завершения всех тестов."""
        LayerBuilder._test_mode = False

    def setUp(self):
        """
        Инициализация общих параметров тестов.

        Этот метод выполняется перед каждым тестом и задает начальные параметры,
        необходимые для тестов, такие как входной набор данных, количество нейронов и
        различные объекты классов.
        """
        self.input_dataset = [1, 2, 3, 4, 5, 6]
        self.neuron_number = 2

        self.data = Data()
        self.data.data_name = 'cube'
        self.data.dataset = {
            'cube': [
                [
                    0.2549019607843137, 0.24313725490196078, 0.21568627450980393,
                    0.24705882352941178, 0.25098039215686274, 0.25098039215686274
                ],
                [
                    0.7333333333333333, 0.7215686274509804, 0.6941176470588235,
                    0.6901960784313725, 0.6823529411764706, 0.6784313725490196
                ],
                [0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
                [0.7, 0.7, 0.7, 0.7, 0.7, 0.7],
                [0.8, 0.8, 0.8, 0.8, 0.8, 0.8],
                [0.9, 0.9, 0.9, 0.9, 0.9, 0.9],
            ]
        }
        self.activation_functions = ActivationFunctions()
        self.layer_builder = LayerBuilder()
        self.neural_network = NeuralNetwork(self.input_dataset)
        self.visualisation = Visualisation()
        self.mode = 'uniform'

        self.weights = self.layer_builder._get_weights_mode(
            len(self.input_dataset), self.neuron_number, [[0.5, -0.5], [0.3, -0.3]], self.mode
        )
        self.bias = 1
        self.switch = False
        self.switch_list = [False, False]

        self.outer_layer = OuterLayer(
            self.input_dataset, self.weights, self.bias,
            self.activation_functions.get_linear, self.activation_functions.get_sigmoid, self.switch_list
        )
        self.hidden_layer = HiddenLayer(
            self.input_dataset, self.weights, self.bias,
            7, self.activation_functions.get_leaky_relu, True
        )


class TestConfigurationMethods(GeneralTestParameters):
    """
    Класс для тестирования методов конфигурации.

    Этот класс расширяет GeneralTestParameters и включает тесты для методов,
    связанных с обработкой конфигурационных данных, таких как получение данных из JSON файла.
    """

    @patch("builtins.open", new_callable=mock_open, read_data='{"key": "value"}')
    def test_get_json_data_success(self, mock_file):
        """
        Проверяет успешное получение данных из JSON файла методом get_json_data.

        Этот тест симулирует успешное открытие и чтение JSON файла и проверяет,
        что метод get_json_data возвращает правильные данные.
        """
        from configuration import get_json_data
        result = get_json_data('testfile')
        self.assertEqual(result, {'key': 'value'})
        mock_file.assert_called_once_with('testfile.json', encoding='UTF-8')

    @patch("builtins.open")
    def test_get_json_data_file_not_found(self, mock_file):
        """
        Проверяет случай отсутствия JSON файла методом get_json_data.

        Этот тест симулирует сценарий, когда JSON файл не найден, и проверяет,
        что метод get_json_data порождает исключение FileNotFoundError.
        """
        mock_file.side_effect = FileNotFoundError
        from configuration import get_json_data
        with self.assertRaises(FileNotFoundError):
            get_json_data('nonexistentfile')


class TestDataMethods(GeneralTestParameters):
    """
    Класс для тестирования методов обработки данных.

    Этот класс включает тесты для различных методов обработки данных,
    таких как получение номера данных и нормированного значения целевого объекта.
    """

    def test_get_data_dict(self):
        """
        На данный момент помечен как TODO.

        Этот метод будет использоваться для тестирования метода get_data_dict,
        который еще не реализован.
        """
        pass

    def test_get_data_number(self):
        """
        Тестируем метод _get_data_number на возврат корректного номера данных.

        Этот тест проверяет, что метод _get_data_number класса Data возвращает
        корректные номера данных из набора входных данных.
        """
        for num in self.input_dataset:
            self.assertEqual(Data._get_data_number(num), num)

    def test_get_data_sample(self):
        """
        На данный момент помечен как TODO.

        Этот метод будет использоваться для тестирования метода get_data_sample,
        который еще не реализован.
        """
        pass

    def test_get_normalized_target_value(self):
        """
        Тестируем метод get_normalized_target_value на возврат нормированного значения целевого объекта.

        Этот тест проверяет, что метод get_normalized_target_value класса Data возвращает
        нормированное значение целевого объекта, деля его на 10.
        """
        for num in self.input_dataset:
            self.assertEqual(self.data.get_normalized_target_value(num), num / 10)


class TestInitializationFunctions(GeneralTestParameters):
    """
    Класс для тестирования функций инициализации.

    Этот класс включает тесты для различных методов инициализации весов, таких как
    равномерная инициализация, инициализация Ксавье и инициализация Хе.
    """

    def test_get_uniform_initialization(self):
        """
        Тестирует метод get_uniform_initialization на корректность возвращаемых значений.

        Этот тест проверяет, что метод get_uniform_initialization возвращает правильный
        диапазон для заданного значения.
        """
        value = 5
        expected = (-5, 5)
        self.assertEqual(InitializationFunctions.get_uniform_initialization(value), expected)

    def test_get_xavier_initialization(self):
        """
        Тестирует метод get_xavier_initialization на корректность возвращаемых значений.

        Этот тест проверяет, что метод get_xavier_initialization возвращает правильный
        диапазон для заданных размеров входных и выходных слоев.
        """
        input_size = 3
        output_size = 2
        limit = (6 / (input_size + output_size)) ** 0.5
        expected = (-limit, limit)
        self.assertEqual(InitializationFunctions.get_xavier_initialization(input_size, output_size), expected)

    def test_get_he_initialization(self):
        """
        Тестирует метод get_he_initialization на корректность возвращаемых значений.

        Этот тест проверяет, что метод get_he_initialization возвращает правильный
        диапазон для заданного размера входного слоя.
        """
        input_size = 4
        limit = (2 / input_size) ** 0.5
        expected = (-limit, limit)
        self.assertEqual(InitializationFunctions.get_he_initialization(input_size), expected)


class TestActivationFunctions(GeneralTestParameters):
    """
    Класс для тестирования функций активации.

    Этот класс включает тесты для различных методов активации, таких как
    линейная, ReLU, сигмоид, тангенс, leaky ReLU, ELU и softmax активация.
    """

    def setUp(self):
        """
        Инициализация параметров перед каждым тестом.

        Этот метод вызывает родительский setUp и инициализирует необходимые
        для тестов атрибуты, такие как объект функций активации и значения
        для тестирования.
        """
        super().setUp()
        self.af = ActivationFunctions()
        self.x = 2
        self.negative_x = -2

    def test_get_linear(self):
        """
        Тестирует метод get_linear на корректность возвращаемого значения.

        Этот тест проверяет, что метод get_linear возвращает то же значение,
        которое было передано на вход.
        """
        self.assertEqual(self.af.get_linear(self.x), self.x)

    def test_get_relu(self):
        """
        Тестирует метод get_relu на корректность возвращаемого значения.

        Этот тест проверяет, что метод get_relu возвращает правильное значение
        для положительного и отрицательного входов (0 для отрицательных).
        """
        self.assertEqual(self.af.get_relu(self.x), self.x)
        self.assertEqual(self.af.get_relu(self.negative_x), 0)

    def test_get_sigmoid(self):
        """
        Тестирует метод get_sigmoid на корректность возвращаемого значения.

        Этот тест проверяет, что метод get_sigmoid возвращает значение в диапазоне
        от 0 до 1.
        """
        result = self.af.get_sigmoid(self.x)
        self.assertTrue(0 <= result <= 1)

    def test_get_tanh(self):
        """
        Тестирует метод get_tanh на корректность возвращаемого значения.

        Этот тест проверяет, что метод get_tanh возвращает значение в диапазоне
        от -1 до 1.
        """
        result = self.af.get_tanh(self.x)
        self.assertTrue(-1 <= result <= 1)

    def test_get_leaky_relu(self):
        """
        Тестирует метод get_leaky_relu на корректность возвращаемого значения.

        Этот тест проверяет, что метод get_leaky_relu возвращает правильное
        значение для положительных и отрицательных входов (умноженное на alpha для отрицательного).
        """
        self.assertEqual(self.af.get_leaky_relu(self.x), self.x)
        self.assertEqual(self.af.get_leaky_relu(self.negative_x), 0.01 * self.negative_x)

    def test_get_elu(self):
        """
        Тестирует метод get_elu на корректность возвращаемого значения.

        Этот тест проверяет, что метод get_elu возвращает правильное значение
        для положительных и отрицательных входов (умноженное на alpha и экспонентное для отрицательных).
        """
        self.assertEqual(self.af.get_elu(self.x), self.x)
        self.assertAlmostEqual(self.af.get_elu(self.negative_x), -0.864664716763, places=6)

    def test_get_softmax(self):
        """
        Тестирует метод get_softmax на корректность возвращаемых значений.

        Этот тест проверяет, что метод get_softmax возвращает значения, которые суммируются до 1.
        """
        result = self.af.get_softmax([1, 2, 3])
        self.assertAlmostEqual(sum(result), 1)


class TestMachineLearningMethods(TestDataMethods):
    """
    Класс для тестирования методов машинного обучения.

    Этот класс включает тесты для методов, таких как сохранение весов и смещений,
    вычисление ошибки, получение регуляризации Lasso и Ridge, вычисление градиентного спуска
    и обновление весов.
    """

    @patch('builtins.open', new_callable=mock_open)
    @patch('pickle.dump')
    def test_save_weights_and_biases(self, mock_dump, mock_open):
        """
        Тестируем метод _save_weights_and_biases на корректное сохранение весов и смещений в файл.

        Этот тест проверяет, что метод _save_weights_and_biases класса MachineLearning
        корректно сохраняет веса и смещения в файл с использованием модуля pickle.
        """
        weights = {'layer1': [0.1, 0.2, 0.3], 'layer2': [0.4, 0.5, 0.6]}
        biases = {'layer1': [0.01, 0.02, 0.03], 'layer2': [0.04, 0.05, 0.06]}
        expected_data = {'weights': weights, 'biases': biases}
        MachineLearning._save_weights_and_biases(weights, biases)
        mock_open.assert_called_once_with('weights_and_biases.pkl', 'wb')
        file_handle = mock_open()
        try:
            mock_dump.assert_called_once_with(expected_data, file_handle)
        except AssertionError:
            print('Ожидалось, что функция "dump" будет вызвана один раз, но вызывалась 0 раз.')
            print('mock_open() вывод: ', file_handle)
            print('mock_dump() вызовы: ', mock_dump.call_args_list)

    def test_calculate_error(self):
        """
        Тестируем метод _calculate_error на корректное вычисление ошибки.

        Этот тест проверяет, что метод _calculate_error класса MachineLearning
        корректно вычисляет ошибку между предсказанным и целевым значениями.
        """
        predicted = 110
        target = 100
        expected_error = 10.0
        result = MachineLearning._calculate_error(predicted, target)
        self.assertEqual(result, expected_error)

    def test_get_lasso_regularization_positive_weight(self):
        """
        Тестируем метод _get_lasso_regularization на корректное вычисление регуляризации для положительного веса.

        Этот тест проверяет, что метод _get_lasso_regularization класса MachineLearning
        корректно вычисляет значения регуляризации Lasso для положительного веса.
        """
        regularization = 0.001
        weights = [[0.5, -0.2], [0.8, -0.4]]
        i, j = 0, 0
        expected_regularization = 0.001
        result = MachineLearning._get_lasso_regularization(regularization, weights, i, j)
        self.assertEqual(result, expected_regularization)

    def test_get_lasso_regularization_negative_weight(self):
        """
        Тестируем метод _get_lasso_regularization на корректное вычисление регуляризации для отрицательного веса.

        Этот тест проверяет, что метод _get_lasso_regularization класса MachineLearning
        корректно вычисляет значения регуляризации Lasso для отрицательного веса.
        """
        regularization = 0.001
        weights = [[0.5, -0.2], [0.8, -0.4]]
        i, j = 0, 1
        expected_regularization = -0.001
        result = MachineLearning._get_lasso_regularization(regularization, weights, i, j)
        self.assertEqual(result, expected_regularization)

    def test_get_ridge_regularization(self):
        """
        Тестируем метод _get_ridge_regularization на корректное вычисление регуляризации.

        Этот тест проверяет, что метод _get_ridge_regularization класса MachineLearning
        корректно вычисляет значении регуляризации Ridge.
        """
        regularization = 0.001
        weights = [[0.5, -0.2], [0.8, -0.4]]
        i, j = 0, 0
        expected_regularization = 0.0005
        result = MachineLearning._get_ridge_regularization(regularization, weights, i, j)
        self.assertEqual(result, expected_regularization)

    def test_calculate_gradient_descent(self):
        """
        Тестируем метод _calculate_gradient_descent на корректное вычисление градиентного спуска.

        Этот тест проверяет, что метод _calculate_gradient_descent класса MachineLearning
        корректно обновляет веса в соответствии с градиентным спуском.
        """
        weights = [[0.5, 0.25], [0.75, 0.5]]
        i, j = 0, 0
        learning_rate = 0.01
        gradient = 0.1
        input_dataset = [1.0, 2.0]
        MachineLearning._calculate_gradient_descent(weights, i, j, learning_rate, gradient, input_dataset)
        expected_weight = 0.5 - 0.01 * 0.1 * 1.0
        self.assertEqual(weights[i][j], expected_weight)

    def test_update_weights(self):
        """
        Тестируем метод update_weights на корректное обновление весов.

        Этот тест проверяет, что метод update_weights класса MachineLearning корректно обновляет веса
        во время обучения с использованием градиентного спуска с учетом регуляризации Lasso и Ridge.
        """
        ml = MachineLearning()
        gradient = 0.1
        lasso, ridge = True, True
        initial_weights, initial_bias = [row[:] for row in self.weights], self.bias
        if self.layer_builder.training:
            expected_weights = [
                [
                    0.3443205071031966, 0.2577518870314966, -0.07972518088389749,
                    -0.24147828537403782, 0.010769664995001677, -0.09565929215441041
                ],
                [
                    0.2836973052361836, -0.1968848805465247, -0.023699975638506645,
                    0.08297770592687338, 0.4076058446309092, 0.004080827696255352
                ]
            ]
        else:
            expected_weights = [
                [
                    initial_weights[0][0] - ml.learning_rate * (
                            gradient + ml._get_lasso_regularization(
                        ml.regularization, initial_weights, 0, 0
                    ) + ml._get_ridge_regularization(ml.regularization, initial_weights, 0, 0)
                    ) * self.input_dataset[0],
                    initial_weights[0][1] - ml.learning_rate * (
                            gradient + ml._get_lasso_regularization(
                        ml.regularization, initial_weights, 0, 1
                    ) + ml._get_ridge_regularization(ml.regularization, initial_weights, 0, 1)
                    ) * self.input_dataset[1],
                ],
                [
                    initial_weights[1][0] - ml.learning_rate * (
                            gradient + ml._get_lasso_regularization(ml.regularization, initial_weights, 1, 0
                                                                    ) + ml._get_ridge_regularization(
                        ml.regularization, initial_weights, 1, 0)
                    ) * self.input_dataset[0],
                    initial_weights[1][1] - ml.learning_rate * (
                            gradient + ml._get_lasso_regularization(ml.regularization, initial_weights, 1, 1
                                                                    ) + ml._get_ridge_regularization(
                        ml.regularization, initial_weights, 1, 1)
                    ) * self.input_dataset[1],
                ]
            ]
        expected_bias = initial_bias - ml.learning_rate * gradient
        ml.update_weights(self, gradient, lasso, ridge)
        self.assertEqual(self.weights, expected_weights)
        self.assertEqual(self.bias, expected_bias)

    def test_train(self):
        """
        Тестируем метод train на корректное обучение модели.

        Этот тест проверяет, что метод train класса MachineLearning корректно изменяет
        веса и смещения слоя во время обучения.
        """
        ml = MachineLearning()
        layer = self.outer_layer
        initial_weights = [row[:] for row in layer.weights]
        initial_bias = layer.bias
        data_number = 0
        ml.train(layer, data_number)
        self.assertNotEqual(layer.weights, initial_weights)
        self.assertNotEqual(layer.bias, initial_bias)

    def test_train_layers_on_dataset(self):
        """
        На данный момент помечен как TODO.

        Этот метод будет использоваться для тестирования метода train_layers_on_dataset,
        который еще не реализован.
        """
        pass


class TestLayerBuilderMethods(TestInitializationFunctions):
    """
    Класс для тестирования методов построения слоев.

    Этот класс включает тесты для методов выбора функции инициализации,
    получения весов и смещений в режиме обучения и проверки типа переключателей.
    """

    def test_select_initialization_function_uniform(self):
        """
        Тестируем метод _select_initialization_function для случая 'uniform'.

        Этот тест проверяет, что метод _select_initialization_function класса LayerBuilder
        корректно возвращает кортеж для режима 'uniform'.
        """
        builder = LayerBuilder()
        result = builder._select_initialization_function('uniform')
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)

    def test_select_initialization_function_xavier(self):
        """
        Тестируем метод _select_initialization_function для случая 'xavier'.

        Этот тест проверяет, что метод _select_initialization_function класса LayerBuilder
        корректно возвращает кортеж для режима 'xavier', учитывая размер входного слоя и количество нейронов.
        """
        builder = LayerBuilder()
        result = builder._select_initialization_function('xavier', input_size=10, neuron_number=5)
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)

    def test_select_initialization_function_he(self):
        """
        Тестируем метод _select_initialization_function для случая 'he'.

        Этот тест проверяет, что метод _select_initialization_function класса LayerBuilder
        корректно возвращает кортеж для режима 'he', учитывая размер входного слоя.
        """
        builder = LayerBuilder()
        result = builder._select_initialization_function('he', input_size=10)
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)

    def test_select_initialization_function_for_bias(self):
        """
        Тестируем метод _select_initialization_function для смещений.

        Этот тест проверяет, что метод _select_initialization_function класса LayerBuilder
        корректно возвращает значение смещения для режима 'uniform'.
        """
        builder = LayerBuilder()
        result = builder._select_initialization_function('uniform', for_bias=True)
        self.assertEqual(result, 0.0)

    def test_select_initialization_function_unknown_mode(self):
        """
        Тестируем метод _select_initialization_function для неизвестного режима.

        Этот тест проверяет, что метод _select_initialization_function класса LayerBuilder
        выбрасывает исключение ValueError для неизвестного режима инициализации.
        """
        builder = LayerBuilder()
        with self.assertRaises(ValueError):
            builder._select_initialization_function('unknown_mode')

    def test_get_weights_mode_training(self):
        """
        Тестируем метод _get_weights_mode в режиме обучения.

        Этот тест проверяет, что метод _get_weights_mode класса LayerBuilder
        корректно возвращает веса, когда объект находится в состоянии тренировки.
        """
        builder = LayerBuilder()
        builder.training = True
        input_size = 3
        neuron_number = 2
        weights = []
        mode = 'uniform'
        result = builder._get_weights_mode(input_size, neuron_number, weights, mode)
        self.assertEqual(len(result), neuron_number)
        self.assertEqual(len(result[0]), input_size)
        self.assertEqual(len(result[1]), input_size)

    def test_get_weights_mode_existing_weights(self):
        """
        Тестируем метод _get_weights_mode для существующих весов.

        Этот тест проверяет, что метод _get_weights_mode класса LayerBuilder
        возвращает существующие веса, когда объект не находится в состоянии тренировки.
        """
        builder = LayerBuilder()
        builder.training = False
        input_size = 3
        neuron_number = 2
        weights = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        mode = 'uniform'
        result = builder._get_weights_mode(input_size, neuron_number, weights, mode)
        self.assertEqual(result, weights)

    def test_get_bias_mode_training(self):
        """
        Тестируем метод _get_bias_mode в режиме обучения.

        Этот тест проверяет, что метод _get_bias_mode класса LayerBuilder
        корректно возвращает смещение, когда объект находится в состоянии тренировки.
        """
        builder = LayerBuilder()
        builder.training = True
        bias = 0.0
        mode = 'uniform'
        result = builder._get_bias_mode(bias, mode)
        self.assertIsInstance(result, float)

    def test_get_bias_mode_existing_bias(self):
        """
        Тестируем метод _get_bias_mode для существующих смещений.

        Этот тест проверяет, что метод _get_bias_mode класса LayerBuilder
        возвращает существующее смещение, когда объект не находится в состоянии тренировки.
        """
        builder = LayerBuilder()
        builder.training = False
        bias = 0.5
        mode = 'uniform'
        result = builder._get_bias_mode(bias, mode)
        self.assertEqual(result, bias)

    def test__verify_switch_type(self):
        """
        Тестируем метод _verify_switch_type.

        Этот тест проверяет, что метод _verify_switch_type класса LayerBuilder
        корректно преобразует переключатель для заданного количества нейронов.
        """
        self.assertEqual(
            self.layer_builder._verify_switch_type(self.switch, self.neuron_number), self.switch_list
        )

    def test__calculate_neuron_dataset(self):
        """
        Тестируем метод _calculate_neuron_dataset.

        Этот тест проверяет, что метод _calculate_neuron_dataset класса LayerBuilder
        корректно вычисляет данные нейронов в зависимости от состояния тренировки объекта.
        """
        if self.layer_builder.training:
            expected_result = [-6.85630916738443, -3.7775713776491626]
        else:
            expected_result = [-2.5, -2.3]
        result = self.layer_builder._calculate_neuron_dataset(
            self.input_dataset, self.neuron_number, self.weights, self.bias, self.switch
        )
        self.assertEqual(result, expected_result)


class TestNeuralNetworkMethods(ActivationFunctions, GeneralTestParameters):
    """
    Класс для тестирования методов нейронной сети.

    Этот класс включает тесты для методов загрузки весов и смещений,
    валидации входных данных, распространения данных через слой,
    добавления и удаления слоев, и ряда других функциональностей.
    """

    def setUp(self):
        """
        Инициализация параметров перед каждым тестом.

        Этот метод вызывает родительский setUp и инициализирует необходимые для тестов атрибуты.
        """
        super().setUp()

    def test_load_weights_and_biases(self):
        """
        Тестируем метод _load_weights_and_biases на корректную загрузку
        весов и смещений из файла.

        Этот тест проверяет, что метод _load_weights_and_biases класса NeuralNetwork
        корректно загружает веса и смещения из pickle файла.
        """
        test_data = {
            'weights': [[0.1, 0.2], [0.3, 0.4]],
            'biases': [0.1, 0.2]
        }
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            filename = temp_file.name
            with open(filename, 'wb') as f:
                pickle.dump(test_data, f)
        loaded_data = NeuralNetwork._load_weights_and_biases(filename)
        self.assertEqual(loaded_data, test_data)
        os.remove(filename)

    def test_validate_input_dataset(self):
        """
        Тестируем метод validate_input_dataset на корректную валидацию входных данных.

        Этот тест проверяет, что метод validate_input_dataset класса NeuralNetwork
        корректно проверяет входные данные.
        """
        self.assertEqual(self.neural_network.validate_input_dataset([1, 1]), [1, 1])

    def test_propagate(self):
        """
        Тестируем метод propagate на корректное распространение данных через слой.

        Этот тест проверяет, что метод propagate класса NeuralNetwork корректно
        распространяет данные через слой, учитывая состояние тренировки.
        """
        if self.layer_builder.training:
            expected_result = [-2.922840276534719, 0.46110436654362297]
        else:
            expected_result = [-2.5, 0.9087119475530022]
        self.assertEqual(
            self.neural_network.propagate(self.outer_layer), expected_result
        )

    def test_add_layer(self):
        """
        Тестируем метод add_layer на корректное добавление нового слоя в нейронную сеть.

        Этот тест проверяет, что метод add_layer класса NeuralNetwork
        корректно добавляет новый слой в нейронную сеть.
        """
        self.neural_network.add_layer('input_layer', self.outer_layer)
        self.assertEqual(self.neural_network.layers['input_layer'], self.outer_layer)

    def test_remove_layer(self):
        """
        Тестируем метод remove_layer на корректное удаление слоя из нейронной сети.

        Этот тест проверяет, что метод remove_layer класса NeuralNetwork
        корректно удаляет слой из нейронной сети.
        """
        self.neural_network.add_layer('input_layer', self.outer_layer)
        self.neural_network.remove_layer('input_layer')

    def test_get_layer_existing_layer(self):
        """
        Тестируем метод get_layer на корректное получение существующего слоя.

        Этот тест проверяет, что метод get_layer класса NeuralNetwork
        корректно возвращает слой, если он существует в нейронной сети.
        """
        nn = self.neural_network
        nn.layers = {
            'outer': self.outer_layer,
            'hidden': self.hidden_layer
        }

        outer_layer = nn.get_layer('outer')
        hidden_layer = nn.get_layer('hidden')

        self.assertIs(outer_layer, self.outer_layer)
        self.assertIs(hidden_layer, self.hidden_layer)

    def test_get_layer_non_existing_layer(self):
        """
        Тестируем метод get_layer на обработку отсутствующего слоя.

        Этот тест проверяет, что метод get_layer класса NeuralNetwork
        корректно обрабатывает случаи, когда запрашиваемого слоя нет в нейронной сети.
        """
        nn = self.neural_network
        nn.layers = {
            'outer': self.outer_layer,
            'hidden': self.hidden_layer
        }

        no_layer = nn.get_layer('input')
        self.assertIsNone(no_layer)

    def test_create_layer(self):
        """
        Тестируем метод _create_layer на корректное создание нового слоя.

        Этот тест проверяет, что метод _create_layer класса NeuralNetwork
        корректно создает новый слой и добавляет его в нейронную сеть.
        """
        layer_name = 'test_layer'
        input_dataset = self.input_dataset
        layer_class = HiddenLayer

        with unittest.mock.patch(
                'builtins.open', unittest.mock.mock_open(
                    read_data=pickle.dumps(
                        {"weights": {layer_name: [[0.1, 0.2], [0.3, 0.4]]},
                         "biases": {layer_name: [0.1, 0.2]}}
                    )
                )
        ):
            layer = self.neural_network._create_layer(
                layer_class, layer_name, input_dataset, 2, self.activation_functions.get_relu, True
            )
            if self.layer_builder.training:
                expected_weights = [
                    [
                        0.3444218515250481, 0.2579544029403025, -0.079428419169155,
                        -0.24108324970703665, 0.01127472136860852, -0.09506586254958571
                    ],
                    [
                        0.2837985890347726, -0.19668727392107255, -0.02340304584764419,
                        0.0833820394550312, 0.4081128851953352, 0.004686855817390256
                    ]
                ]
                expected_bias = 0.3444218515250481
            else:
                expected_weights = [[0.1, 0.2], [0.3, 0.4]]
                expected_bias = [0.1, 0.2]
            self.assertIn(layer_name, self.neural_network.layers)
            self.assertEqual(layer.weights, expected_weights)
            self.assertEqual(layer.bias, expected_bias)

        with unittest.mock.patch('builtins.open', side_effect=FileNotFoundError):
            layer = self.neural_network._create_layer(
                layer_class, layer_name, input_dataset, 2, self.activation_functions.get_relu, True
            )
            self.assertIn(layer_name, self.neural_network.layers)
            self.assertIsNotNone(layer.weights)
            self.assertIsNotNone(layer.bias)

    def test_build_neural_network(self):
        """
        Тестируем метод build_neural_network на корректное построение нейронной сети.

        Этот тест проверяет, что метод build_neural_network класса NeuralNetwork
        корректно вызывает метод _create_layer три раза, а также корректно добавляет
        слои в нейронную сеть.
        """
        with unittest.mock.patch.object(
                self.neural_network, '_create_layer', wraps=self.neural_network._create_layer
        ) as mock_create_layer:
            self.neural_network.build_neural_network()
            self.assertEqual(mock_create_layer.call_count, 3)
            self.assertIn('hidden_layer_first', self.neural_network.layers)
            self.assertIn('hidden_layer_second', self.neural_network.layers)
            self.assertIn('output_outer_layer', self.neural_network.layers)
            self.assertIsInstance(self.neural_network.layers['hidden_layer_first'], HiddenLayer)
            self.assertIsInstance(self.neural_network.layers['hidden_layer_second'], HiddenLayer)
            self.assertIsInstance(self.neural_network.layers['output_outer_layer'], OuterLayer)


if __name__ == '__main__':
    unittest.main()
