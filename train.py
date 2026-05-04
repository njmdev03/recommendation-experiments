import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader
from functools import partial

import config as conf

import utils
from training import engine as train
from training import eval as evaluator
from training import checkpoint as checkpoints


def masked_bce_with_logits(logits, targets, candidate_mask):
    """
    logits: [B, C]
    targets: [B, C] (0/1)
    candidate_mask: [B, C] (1 = valid, 0 = padding)
    """

    loss = F.binary_cross_entropy_with_logits(
        logits,
        targets.float(),
        reduction="none"
    )

    loss = loss * candidate_mask.float()

    return loss.sum() / candidate_mask.sum().clamp_min(1)

def main():
    data, tok = utils.get_dataset()
    val_data, _ = utils.get_dataset(train=False, tok=tok)

    pad_token_id = tok.stoi["<pad>"]

    collate_fn = utils.get_collate()

    if conf.DATASET == conf.Datasets.MIND:
        collate_fn = partial(collate_fn, dataset=data, pad_token_id=pad_token_id)

    loader = DataLoader(
        data,
        batch_size=conf.BATCH_SIZE,
        shuffle=True,
        # pin_memory=True,
        # num_workers=4,
        # persistent_workers=True,
        # prefetch_factor=4,
        collate_fn=collate_fn
    )

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

    embedding, emb_dim = utils.get_embedding(tok, embedding_size_hint=conf.EMBEDDING_SIZE, freeze=conf.FREEZE)

    model = utils.get_model(
        embedding_size=emb_dim,
        embedding=embedding
    )

    criterion = nn.CrossEntropyLoss(ignore_index=-1)

    optimizer = torch.optim.Adam(model.parameters(), lr=conf.LEARNING_RATE)
    scaler = torch.amp.GradScaler(enabled=conf.USE_MIX_PRE)

    start_epoch = 1

    # Checkpoint resuming
    if conf.RESUME_PATH:
        start_epoch = checkpoints.load(
            conf.RESUME_PATH,
            model,
            optimizer,
            scaler
        ) + 1

        print(f"Resuming from {start_epoch}")

    for epoch in range(start_epoch, conf.EPOCHS + 1):
        print(f"Starting epoch {epoch}")

        train_metrics = None
        eval_metrics = None

        train_metrics = train.train_epoch(
            loader,
            model,
            optimizer,
            scaler,
            criterion,
            {
                metric.value: conf.METRIC_REGISTRY[metric]
                for metric in conf.TRAINING_METRICS
            }
        )

        checkpoints.save(
            utils.get_checkpoint_path(epoch),
            model,
            optimizer,
            scaler,
            epoch,
            {"train": train_metrics}
        )

        if conf.TRAIN_VAL_METRICS and len(conf.TRAIN_VAL_METRICS) > 0:
            eval_metrics = evaluator.evaluate(
                model,
                val_loader,
                {
                    metric.value: conf.METRIC_REGISTRY[metric]
                    for metric in conf.TRAIN_VAL_METRICS
                }
            )

            checkpoints.save(
                utils.get_checkpoint_path(epoch),
                model,
                optimizer,
                scaler,
                epoch,
                {"train": train_metrics, "eval": eval_metrics}
            )

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Training interrupted by user")
