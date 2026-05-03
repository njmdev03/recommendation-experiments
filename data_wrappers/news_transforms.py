class TokenizerTransform:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, text):
        return self.tokenizer.encode(text)

class PadTransform:
    def __init__(self, max_len, pad_id=0):
        self.max_len = max_len
        self.pad_id = pad_id

    def __call__(self, input):
        input = input[:self.max_len]
        pad_len = self.max_len - len(input)

        return {
            "output": input + [self.pad_id] * pad_len,
            "mask": [1]*len(input) + [0]*pad_len
        }