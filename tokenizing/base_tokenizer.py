from abc import ABC


class BaseTokenizer(ABC):
    def train(self, texts):
        pass

    def load(self):
        pass

    def save(self):
        pass

    def encode(self, text):
        raise NotImplementedError

    def decode(self, ids):
        raise NotImplementedError

    def get_vocab(self):
        raise NotImplementedError

class BaseWordTokenizer(BaseTokenizer):
    pass

class BaseSubWordTokenizer(BaseTokenizer):
    pass