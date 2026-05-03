import torch
import numpy as np
from sklearn.metrics import roc_auc_score


def compute_auc(scores, batch):
    # scores: (B, N), labels: (B,) indices
    labels = batch["labels"].detach().cpu().numpy()
    scores_np = scores.detach().cpu().numpy()
    mask = batch["imp_mask"].any(dim=-1).detach().cpu().numpy()

    B, N = scores_np.shape
    aucs = []
    for i in range(B):
        m = mask[i]
        y_true = np.zeros(N)
        y_true[labels[i]] = 1

        y_true = y_true[m]
        y_score = scores_np[i][m]

        if len(np.unique(y_true)) > 1:
            aucs.append(roc_auc_score(y_true, y_score))
        else:
            # If only one class is present, AUC is not well-defined.
            # However, in recommendation, this usually means something is wrong
            # with the batching/sampling if it happens often.
            pass
    return np.mean(aucs) if aucs else 0.0

def compute_mrr(scores, batch):
    # scores: (B, N), labels: (B,) indices
    labels = batch["labels"]
    indices = torch.argsort(scores, dim=1, descending=True)
    mask = (indices == labels.unsqueeze(1))
    ranks = mask.nonzero(as_tuple=True)[1]

    mrr = 1.0 / (ranks + 1).float()
    return mrr.mean().item()

def compute_ndcg(scores, batch, k):
    # scores: (B, N), labels: (B,) indices
    labels = batch["labels"]
    indices = torch.argsort(scores, dim=1, descending=True)
    mask = (indices == labels.unsqueeze(1))
    ranks = mask.nonzero(as_tuple=True)[1]

    ndcgs = 1.0 / torch.log2(ranks.float() + 2.0)
    ndcgs[ranks >= k] = 0.0
    return ndcgs.mean().item()
