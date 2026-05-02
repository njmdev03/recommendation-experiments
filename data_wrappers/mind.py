import pandas as pd


# class MINDNews:
#     def __init__(self, split="train"):
#         news_cols = ['news_id', 'category', 'subcategory', 'title',
#                      'abstract', 'url', 'title_entities', 'abstract_entities']
#         self.news_df = pd.read_csv('data/MINDsmall_train/news.tsv',
#                                     sep='\t', names=news_cols)

#         self.news_df = self.news_df.set_index("news_id")
#         self.news_df = self.news_df.fillna("")

#     def __len__(self):
#         return self.news_df.__len__()

#     def __getitem__(self, key):
#         return self.news_df.loc[key]

#     def iter(self, col_id):
#         return self.news_df[col_id]

# class MINDBehaviors:
#     def __init__(self, split="train"):
#         beh_cols = ['impression_id', 'user_id', 'time', 'history', 'impressions']
#         self.behaviors_df = pd.read_csv('data/MINDsmall_train/behaviors.tsv',
#                                         sep='\t', names=beh_cols)

#     def __len__(self):
#         return self.behaviors_df.__len__()

#     def __getitem__(self, idx):
#         return self.behaviors_df.loc[idx]

class MINDDataset:
    def __init__(self, news_tokenizer=None, split="train", size = "small", no_hist_tok="<no_his>"):
        news_cols = ['news_id', 'category', 'subcategory', 'title',
                     'abstract', 'url', 'title_entities', 'abstract_entities']
        self.news_df = pd.read_csv('data/MINDsmall_train/news.tsv',
                                    sep='\t', names=news_cols)

        beh_cols = ['impression_id', 'user_id', 'time', 'history', 'impressions']
        self.behaviors_df = pd.read_csv('data/MINDsmall_train/behaviors.tsv',
                                        sep='\t', names=beh_cols)

        self.news_df = self.news_df.set_index("news_id")
        self.news_df = self.news_df.fillna("")
        self.behaviors_df = self.behaviors_df.fillna(no_hist_tok)

        self.no_hist_tok = no_hist_tok

        self.tok = news_tokenizer

        # Bake news to dict
        # self.news_dict = {}

        # for row in self.news_df:
        #     text = row["title"] + " " + row["abstract"]
        #     self.news_dict[row["news_id"]] = text

    def __len__(self):
        return self.behaviors_df.__len__()

    def get_news_ids(self):
        return self.news_df.index.to_numpy()

    def get_news_text(self, news_id, tokenize=True):
        if news_id == self.no_hist_tok:
            text = self.no_hist_tok
        else:
            row = self.news_df.loc[news_id]

            text = row["title"] + " " + row["abstract"]

        # text = self.news_dict[news_id]

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

        beh = self.behaviors_df.loc[idx]

        # Get the impressions array
        candidates = beh.loc["impressions"].split(" ")

        # Split the impressions into articles and positive/negative sample state
        cand_ids = []
        label = []

        for cand in candidates:
            s = str(cand).split("-")
            cand_ids.append(s[0])
            label.append(int(s[1]))

        # Tokenize the impressions
        tok_cand = []

        for cand in cand_ids:
            tok_cand.append(self.tok.encode(f"{self.news_df.loc[cand].loc["title"]} {self.news_df.loc[cand].loc["abstract"]}"))

        # Tokenize the history
        tok_hist = []

        if beh.loc["history"].split(" ") != ['']:
            for item in beh.loc["history"].split(" "):
                tok_hist.append(self.tok.encode(f"{self.news_df.loc[item].loc["title"]} {self.news_df.loc[item].loc["abstract"]}"))
        else:
            tok_hist = [self.tok.encode("<no_his>")]

        return tok_hist, tok_cand, label
