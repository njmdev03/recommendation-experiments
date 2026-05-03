import torch
from torch import nn
from torch.utils.data.dataloader import DataLoader
from functools import partial
import glob
import os
import json
from pathlib import Path

import config as conf

import utils
from training import eval as evaluator
from training import checkpoint as checkpoints


def to_json_safe(obj):
    import numpy as np
    if isinstance(obj, dict):
        return {k: to_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    return obj

def collate_fn(batch, dataset, pad_token_id):
        B = len(batch)

        hist_ids = torch.full((B, conf.MAX_HIST, conf.MAX_LEN), pad_token_id, dtype=torch.long)
        imp_ids  = torch.full((B, conf.MAX_IMP,  conf.MAX_LEN), pad_token_id, dtype=torch.long)

        hist_mask = torch.zeros((B, conf.MAX_HIST, conf.MAX_LEN), dtype=torch.long)
        imp_mask  = torch.zeros((B, conf.MAX_IMP,  conf.MAX_LEN), dtype=torch.long)

        labels = []

        for i, (hist_ids_list, cand_ids_list, label) in enumerate(batch):
            pos = label.index(1)

            cands = cand_ids_list[:conf.MAX_IMP]

            if pos >= conf.MAX_IMP:
                # replace last item with positive
                cands[-1] = cand_ids_list[pos]
                pos = conf.MAX_IMP - 1

            labels.append(pos)

            # history
            for j, nid in enumerate(hist_ids_list[:conf.MAX_HIST]):
                tokens = dataset.get_news_text(nid)[:conf.MAX_LEN]
                L = len(tokens)

                hist_ids[i, j, :L] = torch.tensor(tokens)
                hist_mask[i, j, :L] = 1

            # candidates
            for j, nid in enumerate(cands):
                tokens = dataset.get_news_text(nid)[:conf.MAX_LEN]
                L = len(tokens)

                imp_ids[i, j, :L] = torch.tensor(tokens)
                imp_mask[i, j, :L] = 1

        labels = torch.tensor(labels)
        assert (labels >= 0).all() and (labels < conf.MAX_IMP).all(), labels

        return {
            "hist_ids": hist_ids,
            "hist_mask": hist_mask,
            "imp_ids": imp_ids,
            "imp_mask": imp_mask,
            "labels": labels
        }

# def score(user_vec, candidate_vecs):
#         # dot product
#         return torch.matmul(candidate_vecs, user_vec.unsqueeze(-1)).squeeze(-1)

def main():
    val_data, tok = utils.get_dataset(train=False)

    pad_token_id = tok.stoi["<pad>"]

    collate = partial(collate_fn, dataset=val_data, pad_token_id=pad_token_id)

    val_loader = DataLoader(
        val_data,
        batch_size=conf.VAL_BATCH_SIZE,
        shuffle=True,
        pin_memory=True,
        num_workers=4,
        persistent_workers=False,
        prefetch_factor=4,
        collate_fn=collate
    )

    # model = utils.get_model(len(tok))

    ckpt_paths = sorted(glob.glob(os.path.join(conf.CKPT_DIR, "*.pt")))

    for checkpoint in ckpt_paths:
        model = utils.get_model(len(tok))
        optimizer = torch.optim.Adam(model.parameters(), lr=conf.LEARNING_RATE)
        scaler = torch.amp.GradScaler(enabled=conf.USE_MIX_PRE)

        # Checkpoint resuming
        _ = checkpoints.load(
            checkpoint,
            model,
            optimizer,
            scaler
        )

        eval_metrics = evaluator.evaluate(
            model,
            val_loader,
            {
                metric.value: conf.METRIC_REGISTRY[metric]
                for metric in conf.VAL_METRICS
            }
        )

        # Build output dict
        result = {
            "checkpoint": checkpoint,
            "metrics": to_json_safe(eval_metrics)
        }

        # Create output filename based on checkpoint name
        ckpt_filename = os.path.basename(checkpoint)          # e.g. model_epoch10.pt
        json_filename = os.path.splitext(ckpt_filename)[0] + ".json"

        output_path = os.path.join(Path(conf.VAL_OUTPUT_DIR), json_filename)

        # Ensure directory exists
        os.makedirs(Path(conf.VAL_OUTPUT_DIR), exist_ok=True)

        # Write JSON
        with open(output_path, "w") as f:
            json.dump(result, f, indent=2)

        print(f"Saved: {output_path}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Evaluation interrupted by user")
