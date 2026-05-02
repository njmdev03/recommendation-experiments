import torch
from torch import nn
from torch.utils.data.dataloader import DataLoader
from functools import partial

import config as conf

import utils
from training import engine as train
from training import eval as evaluator


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

def score(user_vec, candidate_vecs):
        # dot product
        return torch.matmul(candidate_vecs, user_vec.unsqueeze(-1)).squeeze(-1)

def main():
    data, tok = utils.get_dataset()

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

    model = utils.get_model(len(tok))

    criterion = nn.CrossEntropyLoss(ignore_index=pad_token_id)

    optimizer = torch.optim.Adam(model.parameters(), lr=conf.LEARNING_RATE)
    scaler = torch.amp.GradScaler(enabled=conf.USE_MIX_PRE)

    for epoch in range(conf.EPOCHS):
        print(f"Starting epoch {epoch}")

        train.train_epoch(loader, model, optimizer, scaler, criterion)

        metrics = evaluator.evaluate(model, loader, criterion)
        print(metrics)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Training interrupted by user")
