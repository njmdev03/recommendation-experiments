import torch
import pandas as pd

import config as conf


def collate_fn(batch, dataset, pad_token_id):
        B = len(batch)

        hist_ids = torch.full((B, conf.MAX_HIST, conf.MAX_LEN), pad_token_id, dtype=torch.long)
        imp_ids  = torch.full((B, conf.MAX_IMP,  conf.MAX_LEN), pad_token_id, dtype=torch.long)

        hist_mask = torch.zeros((B, conf.MAX_HIST, conf.MAX_LEN), dtype=torch.long)
        imp_mask  = torch.zeros((B, conf.MAX_IMP,  conf.MAX_LEN), dtype=torch.long)

        labels = []

        for i, (hist_ids_list, cand_ids_list, label) in enumerate(batch):
            pos = label.index(1)

            cands = cand_ids_list[:conf.MAX_IMP]

            if pos >= conf.MAX_IMP:
                # replace last item with positive
                cands[-1] = cand_ids_list[pos]
                pos = conf.MAX_IMP - 1

            labels.append(pos)

            # history
            for j, nid in enumerate(hist_ids_list[:conf.MAX_HIST]):
                tokens = dataset.get_news_text(nid)[:conf.MAX_LEN]
                L = len(tokens)

                hist_ids[i, j, :L] = torch.tensor(tokens)
                hist_mask[i, j, :L] = 1

            # candidates
            for j, nid in enumerate(cands):
                tokens = dataset.get_news_text(nid)[:conf.MAX_LEN]
                L = len(tokens)

                imp_ids[i, j, :L] = torch.tensor(tokens)
                imp_mask[i, j, :L] = 1

        labels = torch.tensor(labels)
        assert (labels >= 0).all() and (labels < conf.MAX_IMP).all(), labels

        return {
            "hist_ids": hist_ids,
            "hist_mask": hist_mask,
            "imp_ids": imp_ids,
            "imp_mask": imp_mask,
            "labels": labels
        }

class MINDDataset:
    def __init__(self, split="train", size="small", no_hist_tok="<no_his>", news_source=["title", "abstract"]):
        news_cols = ['news_id', 'category', 'subcategory', 'title',
                     'abstract', 'url', 'title_entities', 'abstract_entities']
        self.news_df = pd.read_csv(f'data/MIND{size}_{split}/news.tsv',
                                    sep='\t', names=news_cols)

        beh_cols = ['impression_id', 'user_id', 'time', 'history', 'impressions']
        self.behaviors_df = pd.read_csv(f'data/MIND{size}_{split}/behaviors.tsv',
                                        sep='\t', names=beh_cols)

        self.news_df = self.news_df.set_index("news_id")
        self.news_df = self.news_df.fillna("")
        self.behaviors_df = self.behaviors_df.fillna(no_hist_tok)

        self.no_hist_tok = no_hist_tok

        self.news_source = news_source

    def __len__(self):
        return self.behaviors_df.__len__()

    def get_news_ids(self):
        return self.news_df.index.to_numpy()

    def get_news_text(self, news_id):
        if news_id == self.no_hist_tok or news_id not in self.news_df.index:
            text = self.no_hist_tok
        else:
            row = self.news_df.loc[news_id]

            text = ""

            for col_name in self.news_source:
                text += str(row[col_name]) + " "

            text = text.strip()

        return text

    def __getitem__(self, idx):
        beh = self.behaviors_df.loc[idx]

        hist_ids = beh.loc["history"].split(" ")
        impressions = beh.loc["impressions"].split(" ")
        cand_ids = [str(x).split("-")[0] for x in impressions]
        labels = [int(str(x).split("-")[1]) for x in impressions]

        return hist_ids, cand_ids, labels