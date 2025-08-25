import re

from base.base import get_json_data, save_json_data
from tools.support_functions import SupportFunctions, NormalizationFunctions


class TextEncoder(NormalizationFunctions, SupportFunctions):
    def __init__(self, target_mode):
        self.target_mode: bool = target_mode

    @staticmethod
    def get_value(name_key: str) -> int:
        return len(get_json_data('learning_data/text', name_key)[name_key])

    @staticmethod
    def get_text_cleaner(character: str) -> str:
        unwanted_characters = r'`~@#$%^&*()—_\+=}{$:"<>\|\\№•«»‘’“”\t\n±×÷§©®™°µ¶'
        pattern = f'[{re.escape(unwanted_characters)}]+'
        return re.sub(pattern, '', character).lower()

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
            if key < self.get_value('yes'):
                return 0.7
            elif key < self.get_value('hmm') * 2:
                return 0.4
            elif key < self.get_value('no') * 3:
                return 0.1
            raise Exception('Неправильное количество входных данных!')
        else:
            return 0.1

    def get_encoded_sentence(self, sentence: str) -> list:
        words_and_punctuation = re.findall(r'\w+|[!?.,;-]', sentence)
        return [float(f'{self.generate_word_token(self.get_text_cleaner(i)):.4f}') for i in words_and_punctuation]

    def generate_word_token(self, word: str, alpha: float = 0.3) -> float:
        min_val = min(ord(i) for i in word) - alpha
        max_val = max(ord(i) for i in word) - alpha
        encoded_letters = [self.normalize_min_max(ord(i), min_val, max_val) for i in word]
        return self.calculate_average(encoded_letters)

    def encode_sentence(self):
        text_data = [i for i in self.get_text_data()]
        classes_data = {str(i + 1): [[[*self.get_encoded_sentence(str(text_data[i]))]], self.get_target(i)] for i in range(len(text_data))}
        save_json_data('weights_biases_and_data', 'input_dataset', {"classes": classes_data})
