import re
from base.base import get_json_data, save_json_data


class TextEncoder:
    """Класс для работы с текстовыми данными."""

    def __init__(self, target_mode: bool):
        self.target_mode: bool = target_mode

    @staticmethod
    def get_encoded_letters(letter: str) -> list[float]:
        """
        Кодирует буквы в числовой формат, деля код символа на 10000.

        :param letter: Строка, содержащая буквы для кодирования.
        :return: Список чисел, представляющих закодированные буквы.
        """
        return [ord(i) / 10000 for i in letter]

    @staticmethod
    def get_target(key: int) -> float:
        """
        Возвращает целевое значение в зависимости от ключа.

        :param key: Целочисленный ключ, определяющий целевое значение.
        :return: Целевое значение (float) для данного ключа.
        """
        targets = {1: 0.9, 2: 0.5, 3: 0.1}
        return targets.get(key, 0.1)

    @staticmethod
    def get_text_cleaner(character: str) -> str:
        """
        Очищает строку от нежелательных символов.

        :param character: Строка, которую необходимо очистить.
        :return: Очищенная строка без нежелательных символов.
        """
        unwanted_characters = r'`~@#$%^&*()_\-+=}{$:;"<>\|\\№•«»‘’“”\t\n±×÷§©®™°µ¶'
        pattern = f'[{re.escape(unwanted_characters)}]+'
        return re.sub(pattern, '', character)

    def encode_sentence(self) -> None:
        """
        Кодирует предложения и сохраняет данные в формате JSON.

        В зависимости от режима работы (target_mode) метод либо запрашивает ввод от пользователя,
        либо загружает данные из JSON-файлов. Закодированные данные сохраняются в файл.
        """
        classes_data = {}

        if not self.target_mode:
            text_data = {"user_input": [input("Введите сообщение: ")]}
        else:
            text_data = {
                "yes": get_json_data('learning_data/text', 'yes')["yes"],
                "hmm": get_json_data('learning_data/text', 'hmm')["hmm"],
                "no": get_json_data('learning_data/text', 'no')["no"]
            }

        for i, label in enumerate(text_data, 1):
            encoded_sentences = []
            for sentence in text_data[label]:
                encoded_sentences.append(self.get_encoded_letters(sentence))
            classes_data[str(i)] = [encoded_sentences, self.get_target(i)]

        save_json_data('weights_biases_and_data', 'input_dataset', {"classes": classes_data})
