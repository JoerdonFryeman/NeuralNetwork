import unittest
from neural_network import LayerBuilder, InputLayer, NeuralNetwork


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
        self.input_data = [1, 1]
        self.neuron_number = 2
        self.bias = 1
        self.switch = False
        self.switch_list = [False, False]
        self.layer_builder = LayerBuilder()
        self.input_layer = InputLayer(self.input_data, self.layer_builder._get_sigmoid)
        self.neural_network = NeuralNetwork([1, 1])

    def test__generate_weights_size(self):
        """
        Проверяет корректность генерации весов для нейронов.
        """
        expected_weights = [[0.6888437030500962, 0.515908805880605], [-0.15885683833831, -0.4821664994140733]]
        result = self.layer_builder._generate_weights_size(self.input_data, self.neuron_number)
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
        expected_result = [-0.7952474910692988, -2.641023337752383]
        result = self.layer_builder._calculate_neuron_dataset(
            self.input_data, self.neuron_number, self.bias, self.switch
        )
        self.assertEqual(result, expected_result)

    def test__get_linear(self):
        """
        Проверяет линейную активационную функцию.
        """
        self.assertEqual(self.layer_builder._get_linear(1), 1)

    def test__get_relu(self):
        """
        Проверяет ReLU активационную функцию.
        """
        self.assertEqual(self.layer_builder._get_relu(1), 1)

    def test__get_sigmoid(self):
        """
        Проверяет сигмоидную активационную функцию.
        """
        self.assertEqual(self.layer_builder._get_sigmoid(1), 0.2689414233455059)

    def test__get_tanh(self):
        """
        Проверяет активационную функцию тангенс гиперболический (tanh).
        """
        self.assertEqual(self.layer_builder._get_tanh(1), 0.9640158262858859)

    def test_validate_input_dataset(self):
        """
        Проверяет валидацию входных данных.
        """
        self.assertEqual(self.neural_network.validate_input_dataset([1, 1]), [1, 1])

    def test_propagate(self):
        """
        Проверяет распространение данных через слой.
        """
        self.assertEqual(self.neural_network.propagate(self.input_layer), [0.5431262981835264, 0.3237340505557927])

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
