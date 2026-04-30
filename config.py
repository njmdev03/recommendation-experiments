from enum import Enum
import torch


class Datasets(Enum):
    MIND = "MIND"

class Tokenizers(Enum):
    WORD = "Word"

class Embeddings(Enum):
    SIMPLE = "Simple"


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyper Parameters
LEARNING_RATE = 1e-4
EPOCHS = 1
BATCH_SIZE = 2

# Dataset Config
DATASET = Datasets.MIND

# Tokenizing
TOKENIZER = Tokenizers.WORD

# Vocab saving
VOCAB = f"./out/vocabs/{TOKENIZER.value}-vocab.json"

# Embeddings
EMBEDDING = Embeddings.SIMPLE
EMBEDDING_SIZE = 100
