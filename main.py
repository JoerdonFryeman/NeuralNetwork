from data import Data
from configuration import logger
from neural_network import NeuralNetwork

training: bool = False
initialization: str = 'uniform'


def main():
    """Основная функция, которая запускает процесс создания и визуализации нейронной сети."""
    try:
        data = Data()
        network = NeuralNetwork(
            training, initialization,
            data.get_data_sample()
        )
        network.build_neural_network()
    except ValueError as error:
        logger.error(f'Проверка выдала ошибку: {error}')


if __name__ == '__main__':
    main()
