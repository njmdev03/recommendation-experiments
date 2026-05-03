import pandas as pd


class MINDDataset:
    def __init__(self, news_tokenizer=None, split="train", size = "small", no_hist_tok="<no_his>"):
        news_cols = ['news_id', 'category', 'subcategory', 'title',
                     'abstract', 'url', 'title_entities', 'abstract_entities']
        self.news_df = pd.read_csv(f'data/MINDsmall_{split}/news.tsv',
                                    sep='\t', names=news_cols)

        beh_cols = ['impression_id', 'user_id', 'time', 'history', 'impressions']
        self.behaviors_df = pd.read_csv(f'data/MINDsmall_{split}/behaviors.tsv',
                                        sep='\t', names=beh_cols)

        self.news_df = self.news_df.set_index("news_id")
        self.news_df = self.news_df.fillna("")
        self.behaviors_df = self.behaviors_df.fillna(no_hist_tok)

        self.no_hist_tok = no_hist_tok

        self.tok = news_tokenizer

    def __len__(self):
        return self.behaviors_df.__len__()

    def get_news_ids(self):
        return self.news_df.index.to_numpy()

    def get_news_text(self, news_id, tokenize=True):
        if news_id == self.no_hist_tok or news_id not in self.news_df.index:
            text = self.no_hist_tok
        else:
            row = self.news_df.loc[news_id]

            text = str(row["title"]) + " " + str(row["abstract"])

        if tokenize:
            return self.tok.encode(text)
        else:
            return text

    def __getitem__(self, idx):
        beh = self.behaviors_df.loc[idx]

        hist_ids = beh.loc["history"].split(" ")
        impressions = beh.loc["impressions"].split(" ")
        cand_ids = [str(x).split("-")[0] for x in impressions]
        labels = [int(str(x).split("-")[1]) for x in impressions]

        return hist_ids, cand_ids, labels