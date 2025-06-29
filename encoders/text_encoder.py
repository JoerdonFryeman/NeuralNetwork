import re
from config_files.base import get_json_data, save_json_data


class TextEncoder:
    @staticmethod
    def encode_letter_to_unicode(letter) -> float:
        return ord(letter) / 100000

    @staticmethod
    def get_filter(word):
        unwanted_characters = r' `.,~@#$%^&*()_\-+=}{$$$$:;"<>\|\\№•«»‘’“”\t\n±×÷§©®™°µ¶'
        pattern = f'[{re.escape(unwanted_characters)}]+'
        return re.sub(pattern, '', word)

    def encode_text_to_unicode(self):
        text = [self.get_filter(w.lower()) for w in get_json_data('learning_data/text', 'text')['text']]
        if not text:
            raise ValueError("Список вопросов не может быть пустым.")
        data = {str(j + 1): [[[self.encode_letter_to_unicode(i) for i in text[j]]], 0.2] for j in range(len(text))}
        save_json_data('weights_biases_and_data', 'input_dataset', {"classes": data})
