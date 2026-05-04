from enum import Enum, auto
import torch
from string import Template
from functools import partial

import metrics


class Metrics(Enum):
    AUC = "AUC"
    MRR = "MRR"
    NDCG_5 = "NDCG@5"
    NDCG_10 = "NDCG@10"

METRIC_REGISTRY = {
    Metrics.AUC: metrics.compute_auc,
    Metrics.MRR: metrics.compute_mrr,
    Metrics.NDCG_5: partial(metrics.compute_ndcg, k=5),
    Metrics.NDCG_10: partial(metrics.compute_ndcg, k=10)
}

class Datasets(Enum):
    MIND = "MIND",
    MIND_SPLIT = "MIND Split"

class Tokenizers(Enum):
    WORD = "Word",
    BPE = "BPE",
    WORD_PIECE = "WordPiece"
    SENTENCE_PIECE = "SentencePiece"

class Embeddings(Enum):
    SIMPLE = "Simple",
    GLOVE = "GloVe",
    FAST_TEXT = "Fast Text",
    WORD_2_VEC = "Word2Vec"

class Glove(Enum):
    GLOVE_6B_50 = auto()
    GLOVE_6B_100 = auto()
    GLOVE_6B_200 = auto()
    GLOVE_6B_300 = auto()

    TWITTER_27B_25 = auto()
    TWITTER_27B_50 = auto()
    TWITTER_27B_100 = auto()
    TWITTER_27B_200 = auto()

    GLOVE_42B_300 = auto()

    WIKIGIGA_300 = auto()
    WIKIGIGA_50 = auto()

class Models(Enum):
    BASIC = "Basic"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Experiment Management
RUN_NAME = "mind_baseline_v2"
LOG_DIR = f"./out/{RUN_NAME}/runs"
CKPT_DIR = f"./out/{RUN_NAME}/checkpoints"
CKPT_NAME = Template(f"Epoch-$epoch.pt")

RESUME_PATH = None

# Hyper Parameters
LEARNING_RATE = 1e-4
EPOCHS = 4
BATCH_SIZE = 16
ACCUM_STEPS = 1
USE_MIX_PRE = True

# Evaluation
VAL_BATCH_SIZE = BATCH_SIZE
VAL_OUTPUT_DIR = f"./out/{RUN_NAME}/val"
VAL_SKIP_DONE = True

# Metrics
TRAINING_METRICS = [Metrics.MRR, Metrics.AUC]
TRAINING_METRIC_EVERY_N = 100
TRAIN_VAL_METRICS = []
VAL_METRICS = [Metrics.AUC, Metrics.MRR, Metrics.NDCG_5, Metrics.NDCG_10]

# Dataset Config
DATASET = Datasets.MIND_SPLIT

MAX_HIST = 50
MAX_CAND  = 20
MAX_LEN  = 128

# Tokenizing
TOKENIZER = Tokenizers.WORD

# Vocab saving
VOCAB = f"./out/{RUN_NAME}/vocabs/{TOKENIZER.value}-vocab.json"

# Embeddings
EMBEDDING = Embeddings.SIMPLE
EMBEDDING_SIZE = 128
GLOVE_TYPE = Glove.TWITTER_27B_50

# Model
MODEL = Models.BASIC
COMPILE = False

# Profiling (ish)
SYNC_PROFILES = False
