import unittest
from neural_network import LayerBuilder, ActivationFunctions, InputLayer, NeuralNetwork


class TestLayerBuilder(unittest.TestCase):
    """
    Класс для тестирования функциональности LayerBuilder, InputLayer и NeuralNetwork.
    """

    @classmethod
    def setUpClass(cls):
        """
        Устанавливает режим тестирования перед запуском всех тестов.
        """
        LayerBuilder._test_mode = True

    @classmethod
    def tearDownClass(cls):
        """
        Отключает режим тестирования после завершения всех тестов.
        """
        LayerBuilder._test_mode = False

    def setUp(self):
        """
        Инициализация общих параметров тестов.
        """
        self.input_dataset = [1.0, 1]
        self.input_size = len(self.input_dataset)
        self.neuron_number = 2
        self.layer_builder = LayerBuilder()
        self.weights = self.layer_builder._initialize_weights(self.input_size, self.neuron_number)
        self.bias = 1
        self.switch = False
        self.switch_list = [False, False]
        self.activation_functions = ActivationFunctions()
        self.input_layer = InputLayer(
            self.input_dataset, self.activation_functions._get_linear, self.activation_functions._get_sigmoid
        )
        self.neural_network = NeuralNetwork([1, 1])

    def test__initialize_weights(self):
        """
        Проверяет корректность генерации весов для нейронов.
        """
        expected_weights = [[0.06888437030500963, 0.051590880588060495], [-0.015885683833831002, -0.048216649941407334]]
        result = self.layer_builder._initialize_weights(self.input_size, self.neuron_number)
        self.assertEqual(result, expected_weights)

    def test__verify_switch_type(self):
        """
        Проверяет верификацию типа переключателя для нейронов.
        """
        self.assertEqual(
            self.layer_builder._verify_switch_type(self.switch, self.neuron_number), self.switch_list
        )

    def test__calculate_neuron_dataset(self):
        """
        Проверяет корректность вычисления данных нейронов.
        """
        expected_result = [-1.8795247491069298, -2.0641023337752387]
        result = self.layer_builder._calculate_neuron_dataset(
            self.input_dataset, self.neuron_number, self.weights, self.bias, self.switch
        )
        self.assertEqual(result, expected_result)

    def test__get_linear(self):
        """
        Проверяет линейную активационную функцию.
        """
        self.assertEqual(self.activation_functions._get_linear(1), 1)

    def test__get_relu(self):
        """
        Проверяет ReLU активационную функцию.
        """
        self.assertEqual(self.activation_functions._get_relu(1), 1)

    def test__get_sigmoid(self):
        """
        Проверяет сигмоидную активационную функцию.
        """
        self.assertEqual(self.activation_functions._get_sigmoid(1), 0.2689414233455059)

    def test__get_tanh(self):
        """
        Проверяет активационную функцию тангенс гиперболический (tanh).
        """
        self.assertEqual(self.activation_functions._get_tanh(1), 0.9640158262858859)

    def test_validate_input_dataset(self):
        """
        Проверяет валидацию входных данных.
        """
        self.assertEqual(self.neural_network.validate_input_dataset([1, 1]), [1, 1])

    def test_propagate(self):
        """
        Проверяет распространение данных через слой.
        """
        self.assertEqual(self.neural_network.propagate(self.input_layer), [0.12498513944051354, 0.5148937039504645])

    def test_add_layer(self):
        """
        Проверяет добавление слоя в нейронную сеть.
        """
        self.neural_network.add_layer('input_layer', self.input_layer)
        self.assertEqual(self.neural_network.layers['input_layer'], self.input_layer)

    def test_remove_layer(self):
        """
        Проверяет удаление слоя из нейронной сети.
        """
        self.neural_network.add_layer('input_layer', self.input_layer)
        self.neural_network.remove_layer('input_layer')
        self.assertNotIn('input_layer', self.neural_network.layers)

    def test_get_layer(self):
        """
        Проверяет получение слоя по его имени.
        """
        self.neural_network.add_layer('input_layer', self.input_layer)
        self.assertEqual(self.neural_network.get_layer('input_layer'), self.input_layer)

    def test_build_neural_network(self):
        """
        Проверяет процесс построения нейронной сети.
        """
        self.neural_network.build_neural_network()
        self.assertEqual(len(self.neural_network.layers), 4)

    def test_get_visualisation(self):
        """
        Проверяет визуализацию нейронной сети.
        """
        self.neural_network.build_neural_network()
        self.neural_network.get_visualisation()


if __name__ == '__main__':
    unittest.main()
