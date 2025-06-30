import re
from config_files.base import get_json_data, save_json_data


class TextEncoder:
    @staticmethod
    def encode_unicode(letter) -> float:
        return ord(letter) / 100000

    @staticmethod
    def get_filter(word):
        unwanted_characters = r' `~@#$%^&*()_\-+=}{$$$$:;"<>\|\\№•«»‘’“”\t\n±×÷§©®™°µ¶'
        pattern = f'[{re.escape(unwanted_characters)}]+'
        return re.sub(pattern, '', word)

    @staticmethod
    def get_target(key: int) -> float:
        target_mode: bool = False

        if target_mode:
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

    def encode_text_to_unicode(self):
        text = [self.get_filter(w.lower()) for w in get_json_data('learning_data/text', 'text')['text']]
        if not text:
            raise ValueError("Список вопросов не может быть пустым.")
        data = {str(j + 1): [[[self.encode_unicode(i) for i in text[j]]], self.get_target(j)] for j in range(len(text))}
        save_json_data('weights_biases_and_data', 'input_dataset', {"classes": data})
