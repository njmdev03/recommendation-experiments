class TokenizerTransform:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, text):
        return self.tokenizer.encode(text)

class PadTransform:
    def __init__(self, max_len, pad_id=0):
        self.max_len = max_len
        self.pad_id = pad_id

    def __call__(self, tokens):
        tokens = tokens[:self.max_len]

        mask = [1] * len(tokens)
        pad_len = self.max_len - len(tokens)

        tokens = tokens + [self.pad_id] * pad_len
        mask = mask + [0] * pad_len

        return {
            "input_ids": tokens,
            "word_mask": mask
        }