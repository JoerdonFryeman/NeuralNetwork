import re
from base.base import get_json_data, save_json_data


class TextEncoder:
    def __init__(self, target_mode):
        self.target_mode: bool = target_mode

    @staticmethod
    def encode_unicode(letter) -> float:
        return ord(letter) / 10000

    @staticmethod
    def get_filter(word):
        unwanted_characters = r' `~@#$%^&*()_\-+=}{$$$$:;"<>\|\\№•«»‘’“”\t\n±×÷§©®™°µ¶'
        pattern = f'[{re.escape(unwanted_characters)}]+'
        return re.sub(pattern, '', word)

    def get_text_data(self):
        if self.target_mode:
            return get_json_data('learning_data/text', 'text')['text']
        return [input(f"Введите сообщение: ")]

    def get_target(self, key: int) -> float:
        if self.target_mode:
            if key < 20:
                return 1.0
            elif key < 40:
                return 0.7
            elif key < 60:
                return 0.4
            elif key < 80:
                return 0.1
            raise Exception('Неправильное количество входных данных!')
        else:
            return 0.1

    def encode_sentence(self):
        text = [self.get_filter(w.lower()) for w in self.get_text_data()]
        data = {str(j + 1): [[[self.encode_unicode(i) for i in text[j]]], self.get_target(j)] for j in range(len(text))}
        save_json_data('weights_biases_and_data', 'input_dataset', {"classes": data})
