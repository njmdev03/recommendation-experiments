import torch
import numpy as np
from sklearn.metrics import roc_auc_score


def compute_auc(scores, batch):
    # scores: (B, N), labels: (B) index OR (B, N) one-hot
    labels = batch["labels"]
    scores_np = scores.detach().cpu().numpy()
    mask_np = batch["candidate_mask"].detach().cpu().numpy()

    B, N = scores_np.shape

    if labels.dim() == 1:
        # Convert index labels to one-hot for AUC calculation
        labels_one_hot = torch.zeros_like(scores)
        labels_one_hot.scatter_(1, labels.unsqueeze(1), 1)
        labels_np = labels_one_hot.detach().cpu().numpy()
    else:
        labels_np = labels.detach().cpu().numpy()

    aucs = []
    for i in range(B):
        m = mask_np[i] > 0
        y_true = labels_np[i][m]
        y_score = scores_np[i][m]

        if len(np.unique(y_true)) > 1:
            aucs.append(roc_auc_score(y_true, y_score))
    return np.mean(aucs) if aucs else 0.0

def compute_mrr(scores, batch):
    # scores: (B, N), labels: (B) index OR (B, N) one-hot
    labels = batch["labels"]
    mask = batch["candidate_mask"]

    # Mask out padded candidates with very low scores
    scores = scores.clone()
    scores[mask == 0] = -1e9

    # Get index of positive item
    if labels.dim() == 1:
        pos_idx = labels
    else:
        pos_idx = labels.argmax(dim=1)

    indices = torch.argsort(scores, dim=1, descending=True)

    # Find rank of positive item
    ranks = (indices == pos_idx.unsqueeze(1)).nonzero(as_tuple=True)[1]

    mrr = 1.0 / (ranks + 1).float()
    return mrr.mean().item()

def compute_ndcg(scores, batch, k):
    # scores: (B, N), labels: (B) index OR (B, N) one-hot
    labels = batch["labels"]
    mask = batch["candidate_mask"]

    scores = scores.clone()
    scores[mask == 0] = -1e9

    if labels.dim() == 1:
        pos_idx = labels
    else:
        pos_idx = labels.argmax(dim=1)

    indices = torch.argsort(scores, dim=1, descending=True)

    ranks = (indices == pos_idx.unsqueeze(1)).nonzero(as_tuple=True)[1]

    ndcgs = 1.0 / torch.log2(ranks.float() + 2.0)
    ndcgs[ranks >= k] = 0.0
    return ndcgs.mean().item()
