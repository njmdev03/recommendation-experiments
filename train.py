import torch
from torch import nn
from torch.utils.data.dataloader import DataLoader
from functools import partial

import config as conf

import utils
from training import engine as train
from training import eval as evaluator
from training import checkpoint as checkpoints


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

def main():
    data, tok = utils.get_dataset()
    val_data, _ = utils.get_dataset(train=False, tok=tok)

    pad_token_id = tok.stoi["<pad>"]

    collate = partial(collate_fn, dataset=data, pad_token_id=pad_token_id)

    loader = DataLoader(
        data,
        batch_size=conf.BATCH_SIZE,
        shuffle=True,
        pin_memory=True,
        num_workers=4,
        persistent_workers=False,
        prefetch_factor=4,
        collate_fn=collate
    )

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

    model = utils.get_model(len(tok))

    criterion = nn.CrossEntropyLoss(ignore_index=pad_token_id)

    optimizer = torch.optim.Adam(model.parameters(), lr=conf.LEARNING_RATE)
    scaler = torch.amp.GradScaler(enabled=conf.USE_MIX_PRE)

    start_epoch = 0

    # Checkpoint resuming
    if conf.RESUME_PATH:
        start_epoch = checkpoints.load(
            conf.RESUME_PATH,
            model,
            optimizer,
            scaler
        ) + 1

        print(f"Resuming from {start_epoch}")

    for epoch in range(start_epoch, conf.EPOCHS):
        print(f"Starting epoch {epoch}")

        del train_metrics
        del eval_metrics

        train_metrics = train.train_epoch(
            loader,
            model,
            optimizer,
            scaler,
            criterion,
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
