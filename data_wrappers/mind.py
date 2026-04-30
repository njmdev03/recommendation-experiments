import pandas as pd


class MINDNews:
    def __init__(self, split="train"):
        news_cols = ['news_id', 'category', 'subcategory', 'title',
                     'abstract', 'url', 'title_entities', 'abstract_entities']
        self.news_df = pd.read_csv('data/MINDsmall_train/news.tsv',
                                    sep='\t', names=news_cols)

        self.news_df = self.news_df.set_index("news_id")
        self.news_df = self.news_df.fillna("")

    def __len__(self):
        return self.news_df.__len__()

    def __getitem__(self, key):
        return self.news_df.loc[key]

    def iter(self, col_id):
        return self.news_df[col_id]

class MINDBehaviors:
    def __init__(self, split="train"):
        beh_cols = ['impression_id', 'user_id', 'time', 'history', 'impressions']
        self.behaviors_df = pd.read_csv('data/MINDsmall_train/behaviors.tsv',
                                        sep='\t', names=beh_cols)

    def __len__(self):
        return self.behaviors_df.__len__()

    def __getitem__(self, idx):
        return self.behaviors_df.loc[idx]

class MINDDataset:
    def __init__(self, news_tokenizer, split="train"):
        news_cols = ['news_id', 'category', 'subcategory', 'title',
                     'abstract', 'url', 'title_entities', 'abstract_entities']
        self.news_df = pd.read_csv('data/MINDsmall_train/news.tsv',
                                    sep='\t', names=news_cols)

        beh_cols = ['impression_id', 'user_id', 'time', 'history', 'impressions']
        self.behaviors_df = pd.read_csv('data/MINDsmall_train/behaviors.tsv',
                                        sep='\t', names=beh_cols)

        self.news_df = self.news_df.set_index("news_id")
        self.news_df = self.news_df.fillna("")
        self.behaviors_df = self.behaviors_df.fillna("")

        self.tok = news_tokenizer

    def __len__(self):
        return self.behaviors_df.__len__()

    def __getitem__(self, idx):
        beh = self.behaviors_df.loc[idx]

        # Get the impressions array
        imps = beh.loc["impressions"].split(" ")

        # Split the impressions into articles and positive/negative sample state
        imp_ids = []
        sample = []

        for imp in imps:
            s = str(imp).split("-")
            imp_ids.append(s[0])
            sample.append(int(s[1]))

        # Tokenize the impressions
        tok_imps = []

        for imp in imp_ids:
            tok_imps.append(self.tok.encode(f"{self.news_df.loc[imp].loc["title"]} {self.news_df.loc[imp].loc["abstract"]}"))

        # Tokenize the history
        tok_hist = []

        for item in beh.loc["history"].split(" "):
            tok_hist.append(self.tok.encode(f"{self.news_df.loc[item].loc["title"]} {self.news_df.loc[item].loc["abstract"]}"))

        return tok_hist, tok_imps, sample
