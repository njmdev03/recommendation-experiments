import torch
from torch import nn
from pathlib import Path

import config as conf

from data_wrappers.mind import MINDDataset
from tokenizing.basic_word import WordTokenizer
from models import NewsRecModel


def get_checkpoint_path(epoch):
    return Path(conf.CKPT_DIR) / Path(conf.CKPT_NAME.substitute(epoch=epoch))

def get_model(vocab_size):
    if conf.MODEL == conf.Models.BASIC:
        model = NewsRecModel(vocab_size=vocab_size)

        if conf.COMPILE:
            model = torch.compile(model)

        return model.to(conf.DEVICE)

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

def get_tokenizer(data):
    if conf.TOKENIZER == conf.Tokenizers.WORD:
        tok = WordTokenizer(specials = ["<pad>", "<unk>", "<bos>", "<eos>", "<no_his>"])

        if not tok.load(conf.VOCAB):
            tok.train(data)
            tok.save(conf.VOCAB)

        return tok

def get_embedding(vocab_size, embedding_size_hint):
    if conf.EMBEDDING == conf.Embeddings.SIMPLE:
        return nn.Embedding(vocab_size, embedding_size_hint)
