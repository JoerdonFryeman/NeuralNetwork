import os
import pickle
import tempfile
import unittest
from unittest.mock import patch, mock_open, MagicMock

from configuration import get_json_data
from data import Data
from main import Control
from layers import LayerBuilder, HiddenLayer
from neural_network import NeuralNetwork
from machine_learning import MachineLearning
from support_functions import ActivationFunctions, InitializationFunctions


class GeneralTestParameters(
    unittest.TestCase, Control, LayerBuilder,
    MachineLearning, ActivationFunctions, InitializationFunctions, Data
):
    """Класс для установки и управления общими параметрами тестирования."""

    @classmethod
    def setUpClass(cls):
        """Устанавливает режим тестирования перед запуском всех тестов."""
        cls.test_mode = True

    @classmethod
    def tearDownClass(cls):
        """Отключает режим тестирования после завершения всех тестов."""
        cls.test_mode = False

    def setUp(self):
        """Инициализация параметров перед каждым тестом."""

        self.input_dataset = [0.5, -0.5]  # Продолжайте использовать конкретные данные
        self.neuron_number = 2
        self.weights = [[0.5, 0.25], [0.75, 0.5]]
        self.bias = 0.1

        self.control = Control()
        self.neural_network = NeuralNetwork(self.control.training, self.control.init_func, self.input_dataset)
        self.test_layer = HiddenLayer(
            self.control.training, self.control.init_func, self.input_dataset, self.weights,
            self.bias, self.neuron_number, self.get_tanh, True, self.test_mode
        )


class TestConfigurationMethods(GeneralTestParameters):
    """Класс для тестирования методов конфигурации."""

    @patch('builtins.open', new_callable=mock_open, read_data='{"key": "value"}')
    def test_get_json_data_success(self, mock_file):
        result = get_json_data('test_file')
        self.assertEqual(result, {'key': 'value'})
        mock_file.assert_called_once_with('config_files/test_file.json', encoding='UTF-8')

    @patch('builtins.open', side_effect=FileNotFoundError)
    def test_get_json_data_file_not_found(self, mock_file):
        with self.assertRaises(FileNotFoundError) as context:
            get_json_data('non_existing_file')
        self.assertEqual(str(context.exception), 'Файл не найден!')
        mock_file.assert_called_once_with('config_files/non_existing_file.json', encoding='UTF-8')


class TestDataMethods(GeneralTestParameters):
    """Класс для тестирования методов обработки данных."""

    @patch.object(Data, 'dataset', {'numbers': {'1': ['sample1', 'sample2', 'sample3']}})
    def test_get_data_dict(self):
        expected_data_dict = {1: 'sample1', 2: 'sample2', 3: 'sample3'}
        result = self.get_data_dict()
        self.assertEqual(result, expected_data_dict)

    @patch.object(Data, 'dataset', {'numbers': {'1': ['sample1', 'sample2', 'sample3']}})
    @patch.object(Data, 'data_number', 2)
    def test_get_data_sample(self):
        expected_sample = 'sample2'
        result = self.get_data_sample()
        self.assertEqual(result, expected_sample)

    @patch.object(Data, 'dataset', {'numbers': {'1': ['sample0', 'sample1', 'sample2', 'sample3']}})
    @patch.object(Data, 'data_number', 2)
    def test_get_normalized_target_value(self):
        expected_normalized_value = 0.3
        result = self.get_normalized_target_value(3)
        self.assertAlmostEqual(result, expected_normalized_value)

    def test_get_target_value_by_key_valid_key(self):
        key = '3'
        expected_value = 0.3
        result = self.get_target_value_by_key(key)
        self.assertEqual(result, expected_value, f"Метод вернул неправильное значение для ключа {key}")

    def test_get_target_value_by_key_invalid_key(self):
        key = '10'
        expected_value = 1.0
        result = self.get_target_value_by_key(key)
        self.assertEqual(result, expected_value, f"Метод вернул неправильное значение для несуществующего ключа {key}")

    def test_get_target_value_by_key_edge_case_key(self):
        key = '6'
        expected_value = 0.6
        result = self.get_target_value_by_key(key)
        self.assertEqual(result, expected_value, f"Метод вернул неправильное значение для ключа {key}")


class TestInitializationFunctions(GeneralTestParameters):
    """Тесты для методов инициализации весов в нейронных сетях."""

    def test_get_uniform(self):
        result = self.get_uniform(0.5)
        expected = (-0.5, 0.5)
        self.assertEqual(result, expected, "Ошибка в методе get_uniform")

    def test_get_xavier(self):
        result = self.get_xavier(3, 2)
        limit = (6 / (3 + 2)) ** 0.5
        expected = (-limit, limit)
        self.assertEqual(result, expected, "Ошибка в методе get_xavier")

    def test_get_he(self):
        result = self.get_he(3)
        limit = (2 / 3) ** 0.5
        expected = (-limit, limit)
        self.assertEqual(result, expected, "Ошибка в методе get_he")


class TestActivationFunctions(GeneralTestParameters):
    """Тесты для активационных функций нейронных сетей."""

    def test_get_linear(self):
        self.assertEqual(self.get_linear(0.5), 0.5, "Ошибка get_linear")

    def test_get_relu(self):
        self.assertEqual(self.get_relu(0.5), 0.5, "Ошибка get_relu для положительного x")
        self.assertEqual(self.get_relu(-0.5), 0.0, "Ошибка get_relu для отрицательного x")

    def test_get_sigmoid(self):
        self.assertAlmostEqual(self.get_sigmoid(0), 0.5, places=5, msg="Ошибка get_sigmoid для x=0")

    def test_get_tanh(self):
        self.assertAlmostEqual(self.get_tanh(0), 0.0, places=5, msg="Ошибка get_tanh для x=0")

    def test_get_leaky_relu(self):
        self.assertEqual(self.get_leaky_relu(0.5), 0.5, "Ошибка get_leaky_relu для положительного x")
        self.assertEqual(self.get_leaky_relu(-0.5), -0.005, "Ошибка get_leaky_relu для отрицательного x")

    def test_get_elu(self):
        self.assertEqual(self.get_elu(1.0), 1.0, "Ошибка get_elu для положительного x")
        self.assertAlmostEqual(self.get_elu(-1.0), -0.6321, places=4, msg="Ошибка get_elu для отрицательного x")

    def test_get_softmax_single_value(self):
        result = self.get_softmax(1.0)
        self.assertEqual(result, [1.0], "Ошибка get_softmax для единственного значения")

    def test_get_softmax_list(self):
        result = self.get_softmax([1.0, 2.0, 3.0])
        expected = [0.0900, 0.2447, 0.6652]
        for r, e in zip(result, expected):
            self.assertAlmostEqual(r, e, places=4, msg="Ошибка get_softmax для списка")


class TestMachineLearningMethods(TestDataMethods):
    """Тесты для методов MachineLearning."""

    @patch('builtins.open', new_callable=mock_open)
    @patch('pickle.dump')
    def test_save_weights_and_biases(self, mock_dump, mock_open_instance):
        weights = {'layer1': [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], 'layer2': [[0.7, 0.8], [0.9, 1.0], [1.1, 1.2]]}
        biases = {'layer1': [0.01, 0.02, 0.03], 'layer2': [0.04, 0.05, 0.06]}
        expected_data = {'weights': weights, 'biases': biases}
        self._save_weights_and_biases('weights_biases_and_data/weights_biases_and_data.pkl', weights, biases)
        mock_open_instance.assert_called_once_with('weights_biases_and_data/weights_biases_and_data.pkl', 'wb')
        file_handle = mock_open_instance()
        mock_dump.assert_called_once_with(expected_data, file_handle)

    def test_calculate_error(self):
        result = self._calculate_error(110, 100)
        self.assertAlmostEqual(result, 10.0, places=4, msg="Ошибка calculate_error")

    def test_get_lasso_regularization_positive_weight(self):
        weights = [[0.5, -0.2], [0.8, -0.4]]
        expected_regularization = 0.001
        result = self._get_lasso_regularization(self.control.regularization, weights, 0, 0)
        self.assertEqual(result, expected_regularization, "Ошибка get_lasso_regularization")

    def test_get_lasso_regularization_negative_weight(self):
        weights = [[0.5, -0.2], [0.8, -0.4]]
        expected_regularization = -0.001
        result = self._get_lasso_regularization(self.control.regularization, weights, 0, 1)
        self.assertEqual(result, expected_regularization, "Ошибка get_lasso_regularization")

    def test_get_ridge_regularization(self):
        weights = [[0.5, -0.2], [0.8, -0.4]]
        expected_regularization = 0.0005
        result = self._get_ridge_regularization(self.control.regularization, weights, 0, 0)
        self.assertEqual(result, expected_regularization, "Ошибка get_ridge_regularization")

    def test_calculate_gradient_descent(self):
        weights = [[0.5, 0.25], [0.75, 0.5]]
        input_dataset = [1.0, 2.0]
        self._calculate_gradient_descent(input_dataset, self.control.learning_rate, 0.1, weights, 0, 0)
        expected_weight = 0.5 - 0.01 * 0.1 * 1.0
        self.assertAlmostEqual(weights[0][0], expected_weight, places=2, msg="Ошибка calculate_gradient_descent")

    def test_calculate_learning_decay(self):
        result = self._calculate_learning_decay(0, 20, 0.1, 0.9)
        self.assertEqual(result, 0.1)
        result = self._calculate_learning_decay(5, 20, 0.1, 0.9)
        self.assertEqual(result, 0.1 * 0.9)
        result = self._calculate_learning_decay(6, 20, 0.1 * 0.9, 0.9)
        self.assertEqual(result, 0.1 * 0.9)
        result = self._calculate_learning_decay(10, 20, 0.1 * 0.9, 0.9)
        self.assertEqual(result, 0.1 * 0.9 * 0.9)
        result = self._calculate_learning_decay(11, 20, 0.1 * 0.9 * 0.9, 0.9)
        self.assertEqual(result, 0.1 * 0.9 * 0.9)
        result = self._calculate_learning_decay(15, 20, 0.1 * 0.9 * 0.9, 0.9)
        self.assertEqual(result, 0.1 * 0.9 * 0.9 * 0.9)

    def test_update_weights(self):
        gradient = 0.1
        lasso, ridge = True, True
        initial_weights, initial_bias = [row[:] for row in self.weights], self.bias
        if self.control.training:
            expected_weights = [[0.49994925, 0.250050625], [0.749949125, 0.50005075]]
        else:
            expected_weights = [
                [
                    initial_weights[0][0] - self.control.learning_rate * (
                            gradient + self._get_lasso_regularization(
                        self.control.regularization, initial_weights, 0, 0
                    ) + self._get_ridge_regularization(self.control.regularization, initial_weights, 0, 0)
                    ) * self.input_dataset[0],
                    initial_weights[0][1] - self.control.learning_rate * (gradient + self._get_lasso_regularization(
                        self.control.regularization, initial_weights, 0, 1
                    ) + self._get_ridge_regularization(self.control.regularization, initial_weights, 0, 1)
                                                                          ) * self.input_dataset[1],
                ],
                [
                    initial_weights[1][0] - self.control.learning_rate * (
                            gradient + self._get_lasso_regularization
                    (self.control.regularization, initial_weights, 1, 0) + self._get_ridge_regularization(
                        self.control.regularization, initial_weights, 1, 0)
                    ) * self.input_dataset[0],
                    initial_weights[1][1] - self.control.learning_rate * (gradient + self._get_lasso_regularization(
                        self.control.regularization, initial_weights, 1, 1) + self._get_ridge_regularization(
                        self.control.regularization, initial_weights, 1, 1)
                                                                          ) * self.input_dataset[1],
                ]
            ]
        expected_bias = initial_bias - self.control.learning_rate * gradient
        self._update_weights(self, gradient, lasso, ridge, self.control.learning_rate, self.control.regularization)
        self.assertEqual(self.weights, expected_weights)
        self.assertEqual(self.bias, expected_bias)

    def test_train_method(self):
        self.test_layer = MagicMock()
        self.test_layer.input_dataset = self.input_dataset
        self.test_layer.get_layer_dataset.return_value = list(map(sum, zip(*self.weights)))

        self.get_data_sample = MagicMock(return_value=self.input_dataset)
        self.get_target_value_by_key = MagicMock(return_value=0.25)
        self._update_weights = MagicMock()
        self.get_train_visualisation = MagicMock()
        self._calculate_error = MagicMock(return_value=0.05)
        self._calculate_learning_decay = MagicMock(side_effect=lambda e, ep, lr, ld: lr * ld)

        data_key = 'test_key'
        epochs = 10
        learning_rate = 0.01
        learning_decay = 0.9
        error_tolerance = 0.05
        regularization = 0.01
        lasso_regularization = True
        ridge_regularization = True

        result = self._train(
            data_key, self.test_layer, epochs, learning_rate, learning_decay, error_tolerance,
            regularization, lasso_regularization, ridge_regularization
        )

        self.assertIsNotNone(result, "Метод _train не должен возвращать None")

        self.assertIsInstance(result, tuple, "Метод должен возвращать кортеж")
        self.assertEqual(len(result), 2, "Кортеж должен содержать два элемента: weights и bias")

        weights, bias = result

        self.assertIsNotNone(weights, "Веса должны быть определены")
        self.assertIsNotNone(bias, "Смещение должно быть определено")

        self.get_data_sample.assert_called()
        self.get_target_value_by_key.assert_called_with(data_key)
        self._update_weights.assert_called()
        self.get_train_visualisation.assert_called()
        self._calculate_learning_decay.assert_called()


class TestLayerBuilderMethods(TestInitializationFunctions):
    """Класс для тестирования методов построения слоев."""

    def test_select_initialization_function_uniform(self):
        result = self._select_init_func('uniform')
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)

    def test_select_initialization_function_xavier(self):
        result = self._select_init_func('xavier', input_size=10, neuron_number=5)
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)

    def test_select_initialization_function_he(self):
        result = self._select_init_func('he', input_size=10)
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)

    def test_select_initialization_function_for_bias(self):
        result = self._select_init_func('uniform', for_bias=True)
        self.assertEqual(result, 0.0)

    def test_select_initialization_function_unknown_mode(self):
        with self.assertRaises(ValueError):
            self._select_init_func('unknown_mode')

    def test_get_weights_mode_training(self):
        self.training = True
        input_size = len(self.input_dataset)
        neuron_number = self.neuron_number
        weights = None
        mode = self.control.init_func
        result = self.select_weights_mode(self.training, input_size, neuron_number, weights, mode, self.test_mode)
        self.assertEqual(len(result), neuron_number)
        self.assertEqual(len(result[0]), input_size)
        self.assertEqual(len(result[1]), input_size)

    def test_get_weights_mode_existing_weights(self):
        self.training = False
        input_size = len(self.input_dataset)
        neuron_number = self.neuron_number
        weights = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        mode = self.control.init_func
        result = self.select_weights_mode(self.training, input_size, neuron_number, weights, mode, self.test_mode)
        self.assertEqual(result, weights)

    def test_get_bias_mode_training(self):
        self.training = True
        bias = 0.0
        mode = self.control.init_func
        result = self.select_bias_mode(self.training, bias, mode, self.test_mode)
        self.assertIsInstance(result, float)

    def test_get_bias_mode_existing_bias(self):
        self.training = False
        bias = 0.5
        mode = self.control.init_func
        result = self.select_bias_mode(self.training, bias, mode, self.test_mode)
        self.assertEqual(result, bias)

    def test__calculate_neuron_dataset(self):
        if self.control.training:
            weights = self.weights
            expected_result = [0.5716699659103408, 0.5716699659103408]
        else:
            weights = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
            expected_result = [0.2913126124515488, 0.2913126124515488]
        result = self.calculate_neuron_dataset(
            self.input_dataset, self.neuron_number, weights, self.bias, self.get_tanh, True, self.test_mode
        )
        self.assertEqual(result, expected_result)


class TestNeuralNetworkMethods(GeneralTestParameters):
    """Класс для тестирования методов нейронной сети."""

    def test_load_weights_and_biases(self):
        test_data = {'weights': self.weights, 'biases': [self.bias, self.bias + 0.1]}
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            filename = temp_file.name
            with open(filename, 'wb') as file:
                pickle.dump(test_data, file)
        loaded_data = self.neural_network._load_weights_and_biases(filename)
        self.assertEqual(loaded_data, test_data)
        os.remove(filename)

    def test_validate_input_dataset(self):
        self.assertEqual(self.neural_network._validate_input_dataset([1, 1]), [1, 1])

    def test_propagate(self):
        if self.control.training:
            expected_result = [0.20868983227415003, 0.37649705581299875]
        else:
            expected_result = [0.5716699659103408, 0.5716699659103408]
        self.assertEqual(
            self.neural_network._propagate(self.test_layer), expected_result
        )

    def test_add_layer(self):
        self.neural_network._add_layer('test_layer', self.test_layer)
        self.assertEqual(self.neural_network.layers['test_layer'], self.test_layer)

    def test_create_layer(self):
        layer_name = 'test_layer'
        input_dataset = self.input_dataset
        layer_class = HiddenLayer

        with unittest.mock.patch(
                'builtins.open', unittest.mock.mock_open(
                    read_data=pickle.dumps(
                        {"weights": {layer_name: self.weights}, "biases": {layer_name: [self.bias, self.bias + 0.1]}}
                    )
                )
        ):
            layer = self.neural_network._create_layer(
                layer_class, layer_name, input_dataset, 2, self.get_tanh, True, self.test_mode
            )
            if self.control.training:
                expected_weights = [
                    [0.8436577925009958, 0.6318566641080297], [-0.194559098040328, -0.5905309473142218]
                ]
                expected_bias = 0.0
            else:
                expected_weights = self.weights
                expected_bias = [self.bias, self.bias + 0.1]
            self.assertIn(layer_name, self.neural_network.layers)
            self.assertEqual(layer.weights, expected_weights)
            self.assertEqual(layer.bias, expected_bias)

        with unittest.mock.patch('builtins.open', side_effect=FileNotFoundError):
            self.assertIn(layer_name, self.neural_network.layers)
            self.assertIsNotNone(self.test_layer.weights)
            self.assertIsNotNone(self.test_layer.bias)

    def test_build_neural_network(self):
        with unittest.mock.patch.object(
                self.neural_network, '_create_layer', wraps=self.neural_network._create_layer
        ) as mock_create_layer:
            self.neural_network.build_neural_network(
                self.control.epochs, self.control.learning_rate, self.control.learning_decay,
                self.control.error_tolerance, self.control.regularization, self.control.lasso_regularization,
                self.control.ridge_regularization, self.test_mode
            )
            self.assertEqual(mock_create_layer.call_count, 2)
            self.assertIn('hidden_layer_first', self.neural_network.layers)
            self.assertIn('hidden_layer_second', self.neural_network.layers)
            self.assertIsInstance(self.neural_network.layers['hidden_layer_first'], HiddenLayer)
            self.assertIsInstance(self.neural_network.layers['hidden_layer_second'], HiddenLayer)


if __name__ == '__main__':
    unittest.main()
