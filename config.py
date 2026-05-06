import torch
from string import Template
from functools import partial

import metrics
from config_utils import Metrics, Datasets, Tokenizers, Embeddings, Glove, Models, METRIC_REGISTRY

# Experiment Management
RUN_NAME = "mind_baseline_v2"
LOG_DIR = f"./out/{RUN_NAME}/runs"
CKPT_DIR = f"./out/{RUN_NAME}/checkpoints"
CKPT_NAME = Template(f"Epoch-$epoch.pt")

RESUME_PATH = None

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyper Parameters
LEARNING_RATE = 1e-4
EPOCHS = 1
BATCH_SIZE = 16
ACCUM_STEPS = 1
USE_MIX_PRE = True

# Evaluation
VAL_BATCH_SIZE = BATCH_SIZE
VAL_OUTPUT_DIR = f"./out/{RUN_NAME}/val"
VAL_SKIP_DONE = True

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
EMBEDDING_SIZE = 128 # Ignored for glove, glove dimensions are used instead
GLOVE_TYPE = Glove.GLOVE_6B_200
FREEZE = False

# Model
MODEL = Models.NRMS

HEAD_DIM = 16 # NRMS
NUM_HEADS = 16

# NUM_HEADS = 8 # Transformer
# NUM_LAYERS = 4
# POSITIONAL = True

COMPILE = False

# Profiling (ish)
SYNC_PROFILES = False
