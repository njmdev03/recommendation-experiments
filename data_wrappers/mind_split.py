import torch
import pandas as pd


class MINDNews:
    def __init__(self, split="train", size="small", no_hist_tok="<no_his>", fields=["title", "abstract"]):
        news_cols = ['news_id', 'category', 'subcategory', 'title',
                     'abstract', 'url', 'title_entities', 'abstract_entities']
        self.news_df = pd.read_csv(f'data/MIND{size}_{split}/news.tsv',
                                    sep='\t', names=news_cols)

        self.news_df = self.news_df.set_index("news_id")
        self.news_df = self.news_df.fillna("")

        self.no_hist_tok = no_hist_tok

        self.fields = fields

    def __len__(self):
        return self.news_df.__len__()

    def get_ids(self):
        return self.news_df.index.to_numpy()

    def __getitem__(self, news_id):
        if news_id == self.no_hist_tok or news_id not in self.news_df.index:
            text = self.no_hist_tok
        else:
            row = self.news_df.loc[news_id]

            text = " ".join(str(row[f]) for f in self.fields).strip()

        return text

class MINDBehaviors:
    def __init__(self, split="train", size="small", no_hist_tok="<no_his>"):
        beh_cols = ['impression_id', 'user_id', 'time', 'history', 'impressions']
        self.behaviors_df = pd.read_csv(f'data/MIND{size}_{split}/behaviors.tsv',
                                        sep='\t', names=beh_cols)

        self.behaviors_df = self.behaviors_df.fillna(no_hist_tok)

        self.no_hist_tok = no_hist_tok

    def __len__(self):
        return self.behaviors_df.__len__()

    def __getitem__(self, idx):
        beh = self.behaviors_df.loc[idx]

        hist_ids = beh.loc["history"].split(" ")
        impressions = beh.loc["impressions"].split(" ")
        cand_ids = [str(x).split("-")[0] for x in impressions]
        labels = [int(str(x).split("-")[1]) for x in impressions]

        return {
            "history_ids": hist_ids,
            "candidate_ids": cand_ids,
            "labels": labels
        }

def collate_fn(batch):
    history = torch.stack([b["history"] for b in batch])
    candidates = torch.stack([b["candidates"] for b in batch])
    labels = torch.stack([b["labels"] for b in batch])

    return history, candidates, labels