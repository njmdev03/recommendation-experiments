import torch
import math
import numpy as np
from sklearn.metrics import roc_auc_score


def compute_auc(scores, labels):
    scores = scores.detach().cpu().numpy()
    labels = labels.detach().cpu().numpy()
    print("labels NaNs:", np.isnan(labels).sum())
    print("scores NaNs:", np.isnan(scores).sum())
    print("labels shape:", np.shape(labels))
    print("scores shape:", np.shape(scores))
    print("unique labels:", np.unique(labels)[:10])
    return roc_auc_score(labels, scores)

def compute_mrr(scores, labels):
    # sort by predicted scores (descending)
    sorted_indices = torch.argsort(scores, descending=True)
    sorted_labels = labels[sorted_indices]

    # find first relevant item
    for i, label in enumerate(sorted_labels):
        if label.item() == 1:
            return 1.0 / (i + 1)
    return 0.0

def dcg_at_k(labels, k):
    labels = labels[:k]
    return sum(
        (2**rel - 1) / math.log2(i + 2)
        for i, rel in enumerate(labels)
    )

def compute_ndcg(scores, labels, k):
    sorted_indices = torch.argsort(scores, descending=True)
    sorted_labels = labels[sorted_indices].tolist()

    dcg = dcg_at_k(sorted_labels, k)

    # ideal ranking
    ideal_labels = sorted(labels.tolist(), reverse=True)
    idcg = dcg_at_k(ideal_labels, k)

    return dcg / idcg if idcg > 0 else 0.0
