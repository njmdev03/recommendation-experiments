import torch
import numpy as np
from sklearn.metrics import roc_auc_score


def compute_auc(scores, batch):
    # scores: (B, N), labels: (B, N) one-hot
    labels_np = batch["labels"].detach().cpu().numpy()
    scores_np = scores.detach().cpu().numpy()
    # candidate_mask is (B, N, L), any(-1) gives (B, N)
    mask = batch["candidate_mask"].any(dim=-1).detach().cpu().numpy()

    B, N = scores_np.shape
    aucs = []
    for i in range(B):
        m = mask[i] > 0
        y_true = labels_np[i][m]
        y_score = scores_np[i][m]

        if len(np.unique(y_true)) > 1:
            aucs.append(roc_auc_score(y_true, y_score))
    return np.mean(aucs) if aucs else 0.0

def compute_mrr(scores, batch):
    # scores: (B, N), labels: (B, N) one-hot
    labels = batch["labels"]
    mask = batch["candidate_mask"].any(dim=-1)

    # Mask out padded candidates with very low scores
    scores = scores.clone()
    scores[mask == 0] = -1e9

    # Get index of positive item
    pos_idx = labels.argmax(dim=1)

    indices = torch.argsort(scores, dim=1, descending=True)

    # Find rank of positive item
    ranks = (indices == pos_idx.unsqueeze(1)).nonzero(as_tuple=True)[1]

    mrr = 1.0 / (ranks + 1).float()
    return mrr.mean().item()

def compute_ndcg(scores, batch, k):
    # scores: (B, N), labels: (B, N) one-hot
    labels = batch["labels"]
    mask = batch["candidate_mask"].any(dim=-1)

    scores = scores.clone()
    scores[mask == 0] = -1e9

    pos_idx = labels.argmax(dim=1)
    indices = torch.argsort(scores, dim=1, descending=True)

    ranks = (indices == pos_idx.unsqueeze(1)).nonzero(as_tuple=True)[1]

    ndcgs = 1.0 / torch.log2(ranks.float() + 2.0)
    ndcgs[ranks >= k] = 0.0
    return ndcgs.mean().item()
