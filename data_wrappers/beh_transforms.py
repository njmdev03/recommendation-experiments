import torch
import random


class ShuffleCandidates:
    def __call__(self, sample):
        ids = sample["candidate_ids"]
        labels = sample["labels"]

        mask = sample.get("candidate_mask")

        if mask is not None:
            paired = list(zip(ids, labels, mask))
            random.shuffle(paired)
            ids, labels, mask = zip(*paired)
            sample["candidate_mask"] = list(mask)
        else:
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

class SelectSinglePositive:
    """
    Converts multi-label -> single-label (index of first positive)
    Keeps label aligned with candidate_ids BEFORE padding.
    """
    def __call__(self, sample):
        labels = sample["labels"]

        pos = [i for i, l in enumerate(labels) if l == 1]
        if len(pos) == 0:
            return None

        sample["labels"] = pos[0]
        return sample

class PadBehavior:
    def __init__(self, max_hist, max_cand, pad_token="<no_his>"):
        self.max_hist = max_hist
        self.max_cand = max_cand
        self.pad_token = pad_token

    def __call__(self, sample):

        # -------------------
        # HISTORY
        # -------------------
        hist = sample["history_ids"][:self.max_hist]

        hist_mask = [1] * len(hist)
        pad_len_h = self.max_hist - len(hist)

        hist += [self.pad_token] * pad_len_h
        hist_mask += [0] * pad_len_h

        # -------------------
        # CANDIDATES
        # -------------------
        cand_ids = sample["candidate_ids"]
        labels = sample["labels"]

        # If we have a list of labels, ensure at least one positive is in the range we keep
        if isinstance(labels, list):
            pos_indices = [i for i, l in enumerate(labels) if l == 1]
            if not pos_indices:
                return None
            
            first_pos = pos_indices[0]
            if first_pos >= self.max_cand:
                # swap positive into the allowed range
                target_idx = self.max_cand - 1
                cand_ids[target_idx], cand_ids[first_pos] = cand_ids[first_pos], cand_ids[target_idx]
                labels[target_idx], labels[first_pos] = labels[first_pos], labels[target_idx]

        cand = cand_ids[:self.max_cand]
        cand_mask = [1] * len(cand)
        pad_len_c = self.max_cand - len(cand)

        cand += [self.pad_token] * pad_len_c
        cand_mask += [0] * pad_len_c

        # -------------------
        # LABEL
        # -------------------
        if isinstance(labels, list):
            labels = labels[:self.max_cand]
            labels += [0] * pad_len_c
        elif isinstance(labels, int):
            # If it's already an index, it must be valid
            if labels >= self.max_cand:
                return None

        # ensure padding did not break alignment assumption
        assert len(cand) == self.max_cand
        assert len(cand_mask) == self.max_cand

        return {
            "history_ids": hist,
            "history_mask": hist_mask,

            "candidate_ids": cand,
            "candidate_mask": cand_mask,

            "labels": labels
        }

class NewsLookup:
    def __init__(self, news_pipeline):
        self.news_pipeline = news_pipeline

    def __call__(self, sample):

        hist_items = [
            self.news_pipeline.get(nid)
            for nid in sample["history_ids"]
        ]

        cand_items = [
            self.news_pipeline.get(nid)
            for nid in sample["candidate_ids"]
        ]

        hist_input = [x["input_ids"] for x in hist_items]
        hist_word_mask = [x["word_mask"] for x in hist_items]

        cand_input = [x["input_ids"] for x in cand_items]
        cand_word_mask = [x["word_mask"] for x in cand_items]

        return {
            "history": hist_input,
            "history_word_mask": hist_word_mask,

            "candidate": cand_input,
            "candidate_word_mask": cand_word_mask,

            "history_mask": sample["history_mask"],
            "candidate_mask": sample["candidate_mask"],

            "labels": sample["labels"]
        }

class ToTensor:
    def __call__(self, sample):
        return {
            "history": torch.tensor(sample["history"], dtype=torch.long),
            "history_word_mask": torch.tensor(sample["history_word_mask"], dtype=torch.float),

            "candidate": torch.tensor(sample["candidate"], dtype=torch.long),
            "candidate_word_mask": torch.tensor(sample["candidate_word_mask"], dtype=torch.float),

            "history_mask": torch.tensor(sample["history_mask"], dtype=torch.float),
            "candidate_mask": torch.tensor(sample["candidate_mask"], dtype=torch.float),

            "labels": torch.tensor(sample["labels"], dtype=torch.long)
        }

class MakeLabelIndex:
    def __init__(self):
        pass

    def __call__(self, sample):
        labels = sample["labels"]

        # convert list → index ONCE
        if isinstance(labels, list):
            pos = [i for i, l in enumerate(labels) if l == 1]
            if len(pos) == 0:
                return None
            sample["labels"] = pos[0]

        return sample