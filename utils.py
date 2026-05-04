import torch
from torch import nn
from pathlib import Path
from enum import Enum
import numpy as np

import config as conf

from data_wrappers.mind import MINDDataset, collate_fn as base_collate
from data_wrappers.mind_split import MINDBehaviors, MINDNews, collate_fn as split_collate
from data_wrappers.pipeline import NewsPipeline, BehaviorPipeline
from data_wrappers.news_transforms import *
from data_wrappers.beh_transforms import *
from tokenizing.basic_word import WordTokenizer
from models import NewsRecModel


def get_checkpoint_path(epoch):
    return Path(conf.CKPT_DIR) / Path(conf.CKPT_NAME.substitute(epoch=epoch))

def get_dataset(train=True, tok=None):
    if conf.DATASET == conf.Datasets.MIND:
        data = MINDDataset(split="train" if train else "dev")

        if not tok:
            tok = get_tokenizer([
                data.get_news_text(nid, tokenize=False)
                for nid in data.get_news_ids()
            ])

        data.tok = tok
        return data, tok

    if conf.DATASET == conf.Datasets.MIND_SPLIT:
        news_dataset = MINDNews(split="train" if train else "dev")

        if not tok:
            tok = get_tokenizer([
                news_dataset[nid] for nid in news_dataset.get_ids()
            ])

        news_pipeline = NewsPipeline(
            news_dataset,
            transforms=[
                TokenizerTransform(tok),
                PadTransform(max_len=conf.MAX_LEN),
            ]
        )

        behaviors_dataset = MINDBehaviors(split="train" if train else "dev")

        behavior_pipeline = BehaviorPipeline(
            behaviors_dataset,
            transforms=[
                ShuffleCandidates(),
                PadBehavior(max_hist=conf.MAX_HIST, max_cand=conf.MAX_CAND),
                NewsLookup(news_pipeline),
                ToTensor()
            ]
        )

        return behavior_pipeline, tok

def get_tokenizer(data):
    if conf.TOKENIZER == conf.Tokenizers.WORD:
        tok = WordTokenizer(specials = ["<pad>", "<unk>", "<bos>", "<eos>", "<no_his>"])

        if not tok.load(conf.VOCAB):
            tok.train(data)
            tok.save(conf.VOCAB)

        return tok

def load_glove(glove_path):
    embeddings_index = {}
    with open(glove_path, encoding="utf-8") as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], dtype="float32")
            embeddings_index[word] = vector
    return embeddings_index


def build_embedding_matrix(tokenizer, embeddings_index, embedding_dim):
    embedding_matrix = np.zeros((len(tokenizer), embedding_dim))

    for word, idx in tokenizer.word_index.items():
        if idx >= len(tokenizer):
            continue
        vector = embeddings_index.get(word)
        if vector is not None:
            embedding_matrix[idx] = vector
        else:
            # optional: random init for OOV words
            embedding_matrix[idx] = np.random.normal(scale=0.6, size=(embedding_dim,))

    return embedding_matrix

def get_embedding(tok, embedding_size_hint, freeze=False):
    if conf.EMBEDDING == conf.Embeddings.SIMPLE:
        emb = nn.Embedding(len(tok), embedding_size_hint)

        emb.weight.requires_grad = not freeze

        return emb, embedding_size_hint

    if conf.EMBEDDING == conf.Embeddings.GLOVE:
        # --- 6B ---
        if conf.GLOVE_TYPE == conf.Glove.GLOVE_6B_50:
            glove_path = "./data/glove/glove.6B/glove.6B.50d.txt"
            embedding_dim = 50

        elif conf.GLOVE_TYPE == conf.Glove.GLOVE_6B_100:
            glove_path = "./data/glove/glove.6B/glove.6B.100d.txt"
            embedding_dim = 100

        elif conf.GLOVE_TYPE == conf.Glove.GLOVE_6B_200:
            glove_path = "./data/glove/glove.6B/glove.6B.200d.txt"
            embedding_dim = 200

        elif conf.GLOVE_TYPE == conf.Glove.GLOVE_6B_300:
            glove_path = "./data/glove/glove.6B/glove.6B.300d.txt"
            embedding_dim = 300

        # --- Twitter 27B ---
        elif conf.GLOVE_TYPE == conf.Glove.TWITTER_27B_25:
            glove_path = "./data/glove/glove.twitter.27B/glove.twitter.27B.25d.txt"
            embedding_dim = 25

        elif conf.GLOVE_TYPE == conf.Glove.TWITTER_27B_50:
            glove_path = "./data/glove/glove.twitter.27B/glove.twitter.27B.50d.txt"
            embedding_dim = 50

        elif conf.GLOVE_TYPE == conf.Glove.TWITTER_27B_100:
            glove_path = "./data/glove/glove.twitter.27B/glove.twitter.27B.100d.txt"
            embedding_dim = 100

        elif conf.GLOVE_TYPE == conf.Glove.TWITTER_27B_200:
            glove_path = "./data/glove/glove.twitter.27B/glove.twitter.27B.200d.txt"
            embedding_dim = 200

        # --- Common Crawl 42B ---
        elif conf.GLOVE_TYPE == conf.Glove.GLOVE_42B_300:
            glove_path = "./data/glove/glove.42B.300d.txt"
            embedding_dim = 300

        # --- Wiki + Gigaword ---
        elif conf.GLOVE_TYPE == conf.Glove.WIKIGIGA_50:
            glove_path = "./data/glove/glove.6B/glove.6B.50d.txt"
            embedding_dim = 50

        elif conf.GLOVE_TYPE == conf.Glove.WIKIGIGA_300:
            glove_path = "./data/glove/glove.6B/glove.6B.300d.txt"
            embedding_dim = 300

        embeddings_index = load_glove(glove_path)

        embedding_matrix = build_embedding_matrix(
            tok,
            embeddings_index,
            embedding_dim
        )

        embedding_tensor = torch.tensor(embedding_matrix, dtype=torch.float32)

        return nn.Embedding.from_pretrained(
            embedding_tensor,
            freeze=freeze
        ), embedding_dim

def get_model(vocab_size, embedding_size, embedding):
    if conf.MODEL == conf.Models.BASIC:
        model = NewsRecModel(vocab_size=vocab_size)

        if conf.COMPILE:
            model = torch.compile(model)

        return model.to(conf.DEVICE)

def get_collate():
    if conf.DATASET == conf.Datasets.MIND:
        return base_collate

    if conf.DATASET == conf.Datasets.MIND_SPLIT:
        return split_collate
