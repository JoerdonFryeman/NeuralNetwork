from configuration import logger
from neural_network import NeuralNetwork


def main():
    """
    Основная функция, которая запускает процесс создания и визуализации нейронной сети.
    """
    try:
        network = NeuralNetwork([1, 1, 1])
        network.build_neural_network()
        network.get_visualisation()
    except ValueError as error:
        logger.error(f'Validation error: {error}')


if __name__ == '__main__':
    main()
