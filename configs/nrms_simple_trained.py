import torch
from string import Template

from config_utils import Metrics, Datasets, Tokenizers, Embeddings, Glove, Models


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

RUN_NAME = "nrms_simple_trained"
LOG_DIR = f"./out/{RUN_NAME}/runs"
CKPT_DIR = f"./out/{RUN_NAME}/checkpoints"
CKPT_NAME = Template(f"Epoch-$epoch.pt")
RESUME_PATH = None

LEARNING_RATE = 1e-4
EPOCHS = 1
BATCH_SIZE = 16
ACCUM_STEPS = 1
USE_MIX_PRE = True

VAL_BATCH_SIZE = BATCH_SIZE
VAL_OUTPUT_DIR = f"./out/{RUN_NAME}/val"
VAL_SKIP_DONE = True

TRAINING_METRICS = [Metrics.MRR, Metrics.AUC]
TRAINING_METRIC_EVERY_N = 100
TRAIN_VAL_METRICS = []
VAL_METRICS = [Metrics.AUC, Metrics.MRR, Metrics.NDCG_5, Metrics.NDCG_10]

DATASET = Datasets.MIND_SPLIT
MAX_HIST = 50
MAX_CAND  = 20
MAX_LEN  = 128
TOKENIZER = Tokenizers.WORD
VOCAB = f"./out/{RUN_NAME}/vocabs/{TOKENIZER.value}-vocab.json"

EMBEDDING = Embeddings.SIMPLE
EMBEDDING_SIZE = 128
FREEZE = False

MODEL = Models.NRMS
HEAD_DIM = 16
NUM_HEADS = 16

COMPILE = False
SYNC_PROFILES = False
