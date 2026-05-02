import config as conf

import torch
from torch import nn
from torch.profiler import profile, record_function, ProfilerActivity
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data.dataloader import DataLoader
import tqdm
from functools import partial

from data_wrappers.mind import MINDDataset
import models

import utils
# from models.simple_transformer import SimpleTransformer


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
    # Training dataset
    data = MINDDataset()

    print("Training Tokenizers")
    # Tokenizers
    tok = utils.get_tokenizer([data.get_news_text(nid, tokenize=False) for nid in data.get_news_ids()])

    pad_token_id = tok.stoi["<pad>"]
    data.tok = tok

    print("Build Embeddings")
    emb = utils.get_embedding(len(tok), conf.EMBEDDING_SIZE)

    collate = partial(
        collate_fn,
        dataset=data,
        pad_token_id=pad_token_id
    )

    dataloader = DataLoader(
        data,
        batch_size=conf.BATCH_SIZE,
        # shuffle=True,
        collate_fn=collate,
        pin_memory=True,
        num_workers=4,
        persistent_workers=False,
        prefetch_factor=4
    )

    # Model
    model = models.NewsRecModel(vocab_size=len(tok)).to(conf.DEVICE)
    if conf.COMPILE:
        model = torch.compile(model, backend="aot_eager")

    # Training
    criterion = nn.CrossEntropyLoss(ignore_index=pad_token_id)
    optimizer = torch.optim.Adam(model.parameters(), lr=conf.LEARNING_RATE)

    # prof = profile(
    #     activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    #     record_shapes=True,
    #     profile_memory=True)

    import time

    # prof.start()
    if conf.USE_MIX_PRE:
                scaler = torch.amp.GradScaler(str(conf.DEVICE))

    for epoch in range(conf.EPOCHS):
        print(f"Starting epoch {epoch}")
        model.train()

        for step, batch in enumerate(tqdm.tqdm(dataloader)):
            # if i % 10 == 0:
            #     torch.cuda.synchronize()
            #     print("allocated:", torch.cuda.memory_allocated() / 1e9, "GB")
            #     print("reserved :", torch.cuda.memory_reserved() / 1e9, "GB")

            t0 = time.time()

            # data already loaded here
            data_time = t0 - end if step > 0 else 0

            t1 = time.time()

            for k in batch:
                batch[k] = batch[k].to(conf.DEVICE, non_blocking=True)

            transfer_time = time.time() - t1

            # torch.cuda.synchronize()
            t2 = time.time()

            if conf.USE_MIX_PRE:
                with torch.amp.autocast(str(conf.DEVICE)):
                    scores = model(batch)
                    loss = criterion(scores, batch["labels"])

                loss = loss / conf.ACCUM_STEPS
                scaler.scale(loss).backward()

                if (step + 1) % conf.ACCUM_STEPS == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
            else:
                scores = model(batch)
                loss = criterion(scores, batch["labels"])

                loss = loss / conf.ACCUM_STEPS
                loss.backward()

                if (step + 1) % conf.ACCUM_STEPS == 0:
                    optimizer.step()
                    optimizer.zero_grad()

            # torch.cuda.synchronize()
            compute_time = time.time() - t2

            end = time.time()

            samples_per_sec = conf.BATCH_SIZE / compute_time

            print(f"\nsamples/sec= {samples_per_sec:.3f}, data={data_time:.3f}, transfer={transfer_time:.3f}, compute={compute_time:.3f}")

            # prof.step()

            # TODO: Optional testing metrics
        # TODO: Checkpoint
        # TODO: Eval + Record

#     prof.stop()
#     prof.export_chrome_trace("out/trace.json")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Training interrupted by user")
