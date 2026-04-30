import re
from collections import Counter
import json
from pathlib import Path
import os

from tokenizing.base_tokenizer import BaseWordTokenizer


class WordTokenizer(BaseWordTokenizer):
    def __init__(self, specials = ["<pad>", "<unk>", "<bos>", "<eos>"], lower=True, max_vocab_size=50000):
        self.lower = lower
        self.max_vocab_size = max_vocab_size
        self.specials = specials

        self.stoi = {}
        self.itos = {}
        self.freqs = {}

    def __len__(self):
        return len(self.itos)

    def _tokenize(self, text):
        if self.lower:
            text = text.lower()
        text = re.sub(r"[^\w\s]", "", text)
        return text.split()

    def train(self, texts, min_freq=1):
        counter = Counter()

        for t in texts:
            counter.update(self._tokenize(t))

        vocab = [
            w for w, c in counter.items()
            if c >= min_freq
        ]

        vocab = vocab[: self.max_vocab_size]

        all_tokens = self.specials + vocab

        self.stoi = {tok: i for i, tok in enumerate(all_tokens)}
        self.itos = {i: tok for tok, i in self.stoi.items()}
        self.freqs = counter

    def encode(self, text, unk="<unk>"):
        tokens = self._tokenize(text)
        return [
            self.stoi.get(tok, self.stoi[unk])
            for tok in tokens
        ]

    def decode(self, ids):
        return " ".join([self.itos[i] for i in ids])

    def get_vocab(self):
        return self.stoi

    def save(self, path, indent=2):
        os.makedirs(os.path.dirname(path), exist_ok=True)

        with open(path, "w") as f:
            json.dump({
                "freqs": self.freqs,
                "stoi": self.stoi,
                "specials": self.specials,
                "lower": self.lower
            }, f, indent=indent)

    def load(self, path):
        if not Path(path).exists():
            return False

        with open(path) as f:
            data = json.load(f)

        self.stoi = data["stoi"]
        self.specials = data["specials"]
        self.lower = data["lower"]
        self.freqs = data["freqs"]
        self.itos = {i: tok for tok, i in self.stoi.items()}

        return True