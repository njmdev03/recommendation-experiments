from enum import Enum, auto
from functools import partial

import metrics


# Move Enums here to keep config files clean
class Metrics(Enum):
    AUC = "AUC"
    MRR = "MRR"
    NDCG_5 = "NDCG@5"
    NDCG_10 = "NDCG@10"

class Datasets(Enum):
    MIND = "MIND"
    MIND_SPLIT = "MIND Split"

class Tokenizers(Enum):
    WORD = "Word"
    BPE = "BPE"
    WORD_PIECE = "WordPiece"
    SENTENCE_PIECE = "SentencePiece"

class Embeddings(Enum):
    SIMPLE = "Simple"
    GLOVE = "GloVe"
    FAST_TEXT = "Fast Text"
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
    NRMS = "NRMS"
    TRANSFORMER = "Transformer"

METRIC_REGISTRY = {
    Metrics.AUC: metrics.compute_auc,
    Metrics.MRR: metrics.compute_mrr,
    Metrics.NDCG_5: partial(metrics.compute_ndcg, k=5),
    Metrics.NDCG_10: partial(metrics.compute_ndcg, k=10)
}
