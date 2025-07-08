import re
from base.base import get_json_data, save_json_data


class TextEncoder:
    def __init__(self, target_mode):
        self.target_mode: bool = target_mode

    @staticmethod
    def get_encoded_letter(letter) -> float:
        return ord(letter) / 10000

    @staticmethod
    def get_text_cleaner(character: str) -> str:
        unwanted_characters = r'`~@#$%^&*()_\-+=}{$:;"<>\|\\№•«»‘’“”\t\n±×÷§©®™°µ¶'
        pattern = f'[{re.escape(unwanted_characters)}]+'
        return re.sub(pattern, '', character)

    def get_text_data(self):
        if self.target_mode:
            return [
                *get_json_data('learning_data/text', 'yes')['yes'],
                *get_json_data('learning_data/text', 'hmm')['hmm'],
                *get_json_data('learning_data/text', 'no')['no'],
            ]
        return [input(f"Введите сообщение: ")]

    def get_target(self, key: int, ) -> float:
        if self.target_mode:
            if key < 40:
                return 0.9
            elif key < 80:
                return 0.5
            elif key < 120:
                return 0.1
            raise Exception('Неправильное количество входных данных!')
        return 0.1

    def encode_sentence(self):
        classes_data = {}
        text_data = [self.get_text_cleaner(w) for w in self.get_text_data()]
        for key in range(len(text_data)):
            encod_sent = []
            for i in text_data[key]:
                encod_sent.append(self.get_encoded_letter(i))
            classes_data[str(key + 1)] = [[encod_sent], self.get_target(key)]
        save_json_data('weights_biases_and_data', 'input_dataset', {"classes": classes_data})
