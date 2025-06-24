from config_files.base import save_json_data


class TextEncoder:
    @staticmethod
    def encode_letter_to_unicode(letter) -> float:
        return ord(letter) / 10000

    def encode_text_to_unicode(self):
        word = 'Ты человек'
        weights = [self.encode_letter_to_unicode(i) for i in word]
        data = {"classes": {"1": [[weights], 0.01]}}
        save_json_data('weights_biases_and_data', 'input_dataset', data)
