import torch
from torch import nn
from pathlib import Path

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

    if conf.DATASET == conf.Datasets.MIND_SPLIT:
        news_dataset = MINDNews()

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

        behaviors_dataset = MINDBehaviors()

        behavior_pipeline = BehaviorPipeline(
            behaviors_dataset,
            transforms=[
                ReorderCandidates(),
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

def get_embedding(vocab_size, embedding_size_hint):
    if conf.EMBEDDING == conf.Embeddings.SIMPLE:
        return nn.Embedding(vocab_size, embedding_size_hint)

def get_collate():
    if conf.DATASET == conf.Datasets.MIND:
        return base_collate

    if conf.DATASET == conf.Datasets.MIND_SPLIT:
        return split_collate