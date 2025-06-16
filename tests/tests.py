import unittest
from unittest.mock import patch, mock_open

from config_files.configuration import get_json_data
from data.data import Data
from tools.support_functions import ActivationFunctions, InitializationFunctions


class GeneralTestParameters(unittest.TestCase, ActivationFunctions, InitializationFunctions, Data):
    """Класс для установки и управления общими параметрами тестирования."""

    def setUp(self):
        """Инициализация параметров перед каждым тестом."""
        pass


class TestConfigurationMethods(GeneralTestParameters):
    """Класс для тестирования методов конфигурации."""

    @patch('builtins.open', new_callable=mock_open, read_data='{"key": "value"}')
    def test_get_json_data_success(self, mock_file):
        result = get_json_data('config_files', 'test_file')
        self.assertEqual(result, {'key': 'value'})
        mock_file.assert_called_once_with('config_files/test_file.json', encoding='UTF-8')

    @patch('builtins.open', side_effect=FileNotFoundError)
    def test_get_json_data_file_not_found(self, mock_file):
        with self.assertRaises(FileNotFoundError) as context:
            get_json_data('config_files', 'non_existing_file')
        self.assertEqual(str(context.exception), 'Файл не найден!')
        mock_file.assert_called_once_with('config_files/non_existing_file.json', encoding='UTF-8')


class TestDataMethods(GeneralTestParameters):
    """Класс для тестирования методов обработки данных. На данный момент помечен как TODO."""
    pass


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


if __name__ == '__main__':
    unittest.main()
