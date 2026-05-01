import config as conf

from enum import Enum
import torch
from torch import nn
from torch.utils.data.dataset import Dataset

from data_wrappers.mind import MINDNews, MINDBehaviors
from tokenizing.basic_word import WordTokenizer


def get_dataset():
    if conf.DATASET == conf.Datasets.MIND:
        return MINDNews(split="train"), MINDBehaviors(split="train")

def get_tokenizer(data):
    if conf.TOKENIZER == conf.Tokenizers.WORD:
        tok = WordTokenizer(specials = ["<pad>", "<unk>", "<bos>", "<eos>", "<no_his>"])

        if not tok.load(conf.VOCAB):
            tok.train(data)
            tok.save(conf.VOCAB)

        return tok

# def get_tgt_tokenizer(data):
#     if conf.TOKENIZER == conf.Tokenizers.WORD:
#         tok = WordTokenizer()

#         if not tok.load(conf.TGT_VOCAB):
#             tok.train(data)
#             tok.save(conf.TGT_VOCAB)

#         return tok

def get_embedding(vocab_size, embedding_size_hint):
    if conf.EMBEDDING == conf.Embeddings.SIMPLE:
        return nn.Embedding(vocab_size, embedding_size_hint)
