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

def main():
    val_data, tok = utils.get_dataset(train=False)

    pad_token_id = tok.stoi["<pad>"]
    
    collate_fn = utils.get_collate()
    
    if conf.DATASET == conf.Datasets.MIND_SPLIT:
        collate_fn = partial(collate_fn, dataset=val_data, pad_token_id=pad_token_id)

    val_loader = DataLoader(
        val_data,
        batch_size=conf.VAL_BATCH_SIZE,
        shuffle=True,
        pin_memory=True,
        num_workers=4,
        persistent_workers=False,
        prefetch_factor=4,
        collate_fn=collate_fn
    )

    # model = utils.get_model(len(tok))

    ckpt_paths = sorted(glob.glob(os.path.join(conf.CKPT_DIR, "*.pt")))

    print(f"Found checkpoints: {ckpt_paths}")

    if conf.VAL_SKIP_DONE:
        val_paths = sorted(glob.glob(os.path.join(conf.VAL_OUTPUT_DIR, "*.json")))

        print(f"Found evaluations: {val_paths}")

        for path in val_paths:
            filename = os.path.basename(path).split(".")[0]
            for i, c_path in enumerate(ckpt_paths):
                c_filename = os.path.basename(c_path).split(".")[0]
                if c_filename == filename:
                    ckpt_paths.pop(i)

    print(f"Checkpoint to be evaluated: {ckpt_paths}")

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
