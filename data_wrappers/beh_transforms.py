import torch
import random


class ShuffleCandidates:
    def __call__(self, sample):
        ids = sample["candidate_ids"]
        labels = sample["labels"]

        paired = list(zip(ids, labels))
        random.shuffle(paired)

        ids, labels = zip(*paired)

        sample["candidate_ids"] = list(ids)
        sample["labels"] = list(labels)
        return sample


class ReorderCandidates:
    def __call__(self, sample):
        ids = sample["candidate_ids"]
        labels = sample["labels"]

        paired = list(zip(ids, labels))
        paired.sort(key=lambda x: x[1], reverse=True)

        ids, labels = zip(*paired)

        sample["candidate_ids"] = list(ids)
        sample["labels"] = list(labels)
        return sample

class PadBehavior:
    def __init__(self, max_hist, max_cand, pad_token="<no_his>"):
        self.max_hist = max_hist
        self.max_cand = max_cand
        self.pad_token = pad_token

    def __call__(self, sample):
        h = sample["history_ids"][:self.max_hist]
        c = sample["candidate_ids"][:self.max_cand]
        l = sample["labels"][:self.max_cand]

        h += [self.pad_token] * (self.max_hist - len(h))
        c += [self.pad_token] * (self.max_cand - len(c))
        l += [0] * (self.max_cand - len(l))

        sample["history_ids"] = h
        sample["candidate_ids"] = c
        sample["labels"] = l

        return sample

def split(items):
    inputs = []
    masks = []

    for x in items:
        inputs.append(x["output"])
        masks.append(x["mask"])

    return inputs, masks

class NewsLookup:
    def __init__(self, news_pipeline):
        self.news_pipeline = news_pipeline

    def __call__(self, sample):
        # history
        hist_items = [self.news_pipeline.get(nid) for nid in sample["history_ids"]]
        hist_input, hist_mask = split(hist_items)

        # candidates
        cand_items = [self.news_pipeline.get(nid) for nid in sample["candidate_ids"]]
        cand_input, cand_mask = split(cand_items)

        return {
            "history": hist_input,
            "history_mask": hist_mask,

            "candidate": cand_input,
            "candidate_mask": cand_mask,

            # "history_ids": sample["history_ids"],
            # "candidate_ids": sample["candidate_ids"],

            "labels": sample["labels"]
        }

class ToTensor:
    def __call__(self, sample):
        return {
            "history": torch.tensor(sample["history"], dtype=torch.float), # dtype=torch.float
            "history_mask": torch.tensor(sample["history_mask"], dtype=torch.float), # dtype=torch.float
            "candidate": torch.tensor(sample["candidate"], dtype=torch.float), # dtype=torch.float
            "candidate_mask": torch.tensor(sample["candidate_mask"], dtype=torch.float), # dtype=torch.float
            # "history_ids": sample["history_ids"],
            # "candidate_ids": sample["candidate_ids"],
            "labels": torch.tensor(sample["labels"], dtype=torch.float) # dtype=torch.float
        }