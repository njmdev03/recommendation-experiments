import config as conf

import torch
from torch import nn
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
from data_wrappers.mind import MINDDataset, MINDNews

import utils
# from models.simple_transformer import SimpleTransformer


news = MINDNews()

# print("Loading dataset...")
# # Dataset
# raw = utils.get_dataset()

print("Training Tokenizers")
# Tokenizers
tok = utils.get_tokenizer(news.iter("abstract").to_numpy())

print(f"{tok.encode("Top Halloween parties this weekend")}")

print("Build Embeddings")
emb = utils.get_embedding(tok.__len__(), conf.EMBEDDING_SIZE)

# Training dataset
data = MINDDataset(tok)

# Dataloader
# from torch.nn.utils.rnn import pad_sequence

def collate_fn(batch, pad_id=0):
    print(f"Pre-collate Batch sample {batch}")
    print()
#     src_batch = [torch.tensor(x[0]) for x in batch]
#     tgt_batch = [torch.tensor(x[1]) for x in batch]

#     src_batch = pad_sequence(src_batch, batch_first=True, padding_value=pad_id)
#     tgt_batch = pad_sequence(tgt_batch, batch_first=True, padding_value=pad_id)

#     return src_batch, tgt_batch
    return batch

dataloader = DataLoader(
    data,
    batch_size=conf.BATCH_SIZE,
    # shuffle=True,
    collate_fn=collate_fn
)

for batch in dataloader:
    print(f"Batch sample {batch}")
    print()

    quit()

# Model
model = SimpleTransformer(emb, tgt_emb, tgt_tok.__len__())
model.to(conf.DEVICE)

# Training
criterion = nn.CrossEntropyLoss(ignore_index=tgt_tok.stoi["<pad>"])
optimizer = torch.optim.Adam(model.parameters(), lr=conf.LEARNING_RATE)

for epoch in range(conf.EPOCHS):
    print(f"Starting epoch {epoch}")
    model.train()

    for src_batch, tgt_batch in tqdm(dataloader):
        src_batch = src_batch.to(conf.DEVICE)
        tgt_batch = tgt_batch.to(conf.DEVICE)

        tgt_input = tgt_batch[:, :-1]
        tgt_output = tgt_batch[:, 1:]

        with torch.amp.autocast(str(conf.DEVICE)):
            logits = model(src_batch, tgt_input)

            loss = criterion(
                logits.reshape(-1, logits.size(-1)),
                tgt_output.reshape(-1)
            )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
