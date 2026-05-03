from enum import Enum
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
    MIND = "MIND"

class Tokenizers(Enum):
    WORD = "Word"

class Embeddings(Enum):
    SIMPLE = "Simple"

class Models(Enum):
    BASIC = "Basic"

# Experiment Management
RUN_NAME = "mind_baseline_v1"
LOG_DIR = f"./out/{RUN_NAME}/runs"
CKPT_DIR = f"./out/{RUN_NAME}/checkpoints"
CKPT_NAME = Template(f"Epoch-$epoch.pt")

RESUME_PATH = "./out/mind_baseline_v1/checkpoints/Epoch-0.pt"

# Profiling (ish)
SYNC_PROFILES = False

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyper Parameters
LEARNING_RATE = 1e-4
EPOCHS = 2
BATCH_SIZE = 8
ACCUM_STEPS = 1
USE_MIX_PRE = True

# Evaluation
VAL_BATCH_SIZE = BATCH_SIZE
VAL_OUTPUT_DIR = f"./out/{RUN_NAME}/val"

# Model
MODEL = Models.BASIC
COMPILE = False

# Dataset Config
DATASET = Datasets.MIND

MAX_HIST = 50
MAX_IMP  = 20
MAX_LEN  = 128

# Tokenizing
TOKENIZER = Tokenizers.WORD

# Vocab saving
VOCAB = f"./out/{RUN_NAME}/vocabs/{TOKENIZER.value}-vocab.json"

# Embeddings
EMBEDDING = Embeddings.SIMPLE
EMBEDDING_SIZE = 128

# Metrics
TRAINING_METRICS = [Metrics.MRR]
TRAINING_METRIC_EVERY_N = 100
TRAIN_VAL_METRICS = []
VAL_METRICS = [Metrics.MRR, Metrics.NDCG_5, Metrics.NDCG_10]
