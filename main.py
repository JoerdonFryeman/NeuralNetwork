from data import Data
from configuration import logger
from neural_network import NeuralNetwork


def main():
    """
    Основная функция, которая запускает процесс создания и визуализации нейронной сети.
    """
    try:
        data = Data()
        network = NeuralNetwork(data.get_data_sample())
        network.build_neural_network()
        network.get_visualisation()
    except ValueError as error:
        logger.error(f'Validation error: {error}')


if __name__ == '__main__':
    main()
