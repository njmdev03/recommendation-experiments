import torch
from torch import nn
from torch.utils.data.dataloader import DataLoader
import argparse
import os
import sys
from functools import partial
import torch.nn.functional as F

import config as conf
import config_utils
import utils

from data_wrappers.mind_split import MINDNews, MINDBehaviors
from training import checkpoint as checkpoints


def get_latest_checkpoint(ckpt_dir):
    """Find latest checkpoint by epoch number."""
    import glob
    ckpt_paths = sorted(glob.glob(os.path.join(ckpt_dir, "*.pt")))
    if not ckpt_paths:
        return None
    return ckpt_paths[-1]


def format_news(news_id, news_dataset, max_len=100):
    """Format news article for display."""
    if news_id == "<no_his>":
        return "[No history]"
    try:
        text = news_dataset[news_id]
        if len(text) > max_len:
            return f"[{news_id}] {text[:max_len]}..."
        else:
            return f"[{news_id}] {text}"
    except:
        return f"[{news_id}] [Error loading]"


def run_inference(user_idx, news_dataset, behaviors_dataset, model, tokenizer, device):
    """Run inference on single user behavior."""
    from data_wrappers.pipeline import NewsPipeline, BehaviorPipeline
    from data_wrappers.news_transforms import TokenizerTransform, PadTransform
    from data_wrappers.beh_transforms import (
        PadBehavior, NewsLookup, ToTensor
    )

    beh = behaviors_dataset[user_idx]

    hist_ids = beh["history_ids"]
    cand_ids_raw = beh["candidate_ids"]
    labels_raw = beh["labels"]

    print("\n" + "="*80)
    print(f"User behavior #{user_idx}")
    print("="*80)

    # History
    print("\n[User History]")
    for i, nid in enumerate(hist_ids):
        if nid == "<no_his>":
            continue
        print(f"  {i+1}. {format_news(nid, news_dataset)}")

    if len([n for n in hist_ids if n != "<no_his>"]) == 0:
        print("  (empty)")

    # Candidates + labels
    print(f"\n[Candidate Impressions] (ground truth label → 0=skip / 1=click)")
    for i, (cid, label) in enumerate(zip(cand_ids_raw, labels_raw)):
        status = "✓ CLICKED" if label == 1 else "  skipped"
        print(f"  {i}. {status} {format_news(cid, news_dataset)}")

    # Model prediction
    print("\n[Model Prediction]")

    # Build pipeline for preprocessing
    news_pipeline = NewsPipeline(
        news_dataset,
        transforms=[
            TokenizerTransform(tokenizer),
            PadTransform(max_len=conf.MAX_LEN),
        ]
    )

    # Preprocess WITHOUT losing label info
    sample = {
        "history_ids": hist_ids,
        "candidate_ids": cand_ids_raw,
        "labels": labels_raw
    }

    # Apply transforms manually
    pad_transform = PadBehavior(max_hist=conf.MAX_HIST, max_cand=conf.MAX_CAND)
    sample = pad_transform(sample)

    if sample is None:
        print("  Error: No positive label in candidates")
        return

    news_lookup = NewsLookup(news_pipeline)
    sample = news_lookup(sample)

    to_tensor = ToTensor()
    sample = to_tensor(sample)

    # Add batch dimension
    batch = {k: v.unsqueeze(0) for k, v in sample.items()}

    # Move to device
    for k in batch:
        batch[k] = batch[k].to(device)

    # Forward pass
    with torch.no_grad():
        scores = model(batch)  # [1, max_cand]

    # Get relative probabilities (softmax over valid candidates)
    scores_valid = scores[0, :len(cand_ids_raw)]
    probs = F.softmax(scores_valid, dim=0)  # [num_actual_candidates]

    # Show predictions for actual candidates
    print(f"  Top predictions (over {len(cand_ids_raw)} candidates):")
    top_k = min(5, len(cand_ids_raw))
    top_probs, top_indices = torch.topk(probs, top_k)

    for rank, (idx, prob) in enumerate(zip(top_indices, top_probs)):
        idx = idx.item()
        prob = prob.item()
        cid = cand_ids_raw[idx]
        is_correct = " ← CORRECT" if labels_raw[idx] == 1 else ""
        score_val = scores_valid[idx].item()
        print(f"    {rank+1}. [{idx:2d}] {prob*100:6.2f}% (score: {score_val:8.4f})  {format_news(cid, news_dataset)}{is_correct}")

    # Ground truth
    true_idx = labels_raw.index(1) if 1 in labels_raw else -1
    if true_idx >= 0:
        true_prob = probs[true_idx].item()
        true_score = scores_valid[true_idx].item()
        true_rank = (probs >= probs[true_idx]).sum().item()
        print(f"\n  Ground truth: position {true_idx}, {true_prob*100:.2f}% prob, score {true_score:.4f}, ranked #{true_rank}")


def main():
    parser = argparse.ArgumentParser(description="Run inference on MIND dataset")
    parser.add_argument("--dataset", type=str, choices=["small", "large"], default="small",
                        help="Dataset size (small/large)")
    parser.add_argument("--split", type=str, choices=["train", "dev"], default="dev",
                        help="Dataset split (train/dev)")
    parser.add_argument("--user_id", type=int, default=None,
                        help="Behavior index to show. If not set, enter interactive mode.")
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Checkpoint path. If not set, uses latest from config.")

    args = parser.parse_args()

    # Load data
    print("Loading dataset...")
    news_dataset = MINDNews(split=args.split, size=args.dataset)
    behaviors_dataset = MINDBehaviors(split=args.split, size=args.dataset)

    print(f"Loaded {len(news_dataset)} articles, {len(behaviors_dataset)} behaviors")

    # Load tokenizer
    print("Loading tokenizer...")
    tok = utils.get_tokenizer([
        news_dataset[nid] for nid in news_dataset.get_ids()
    ])

    # Load model
    print("Loading model...")
    embedding, emb_dim = utils.get_embedding(
        tok,
        embedding_size_hint=conf.EMBEDDING_SIZE,
        freeze=conf.FREEZE
    )
    model = utils.get_model(embedding_size=emb_dim, embedding=embedding)

    # Load checkpoint
    ckpt_path = args.ckpt or get_latest_checkpoint(conf.CKPT_DIR)
    if not ckpt_path:
        print(f"Error: No checkpoint found in {conf.CKPT_DIR}")
        sys.exit(1)

    print(f"Loading checkpoint: {ckpt_path}")
    optimizer = torch.optim.Adam(model.parameters(), lr=conf.LEARNING_RATE)
    scaler = torch.amp.GradScaler(enabled=conf.USE_MIX_PRE)
    checkpoints.load(ckpt_path, model, optimizer, scaler)

    model = model.to(conf.DEVICE)
    model.eval()

    # Interactive or batch mode
    if args.user_id is not None:
        # Batch mode
        try:
            run_inference(args.user_id, news_dataset, behaviors_dataset, model, tok, conf.DEVICE)
        except Exception as e:
            print(f"Error: {e}")
            sys.exit(1)
    else:
        # Interactive mode
        print(f"\nInteractive mode. Enter behavior index (0-{len(behaviors_dataset)-1}), or 'q' to quit:")

        while True:
            try:
                user_input = input("> ").strip()
                if user_input.lower() in ["q", "quit", "exit"]:
                    print("Goodbye.")
                    break

                user_idx = int(user_input)
                if user_idx < 0 or user_idx >= len(behaviors_dataset):
                    print(f"Invalid index. Range: 0-{len(behaviors_dataset)-1}")
                    print(f"Enter behavior index (0-{len(behaviors_dataset)-1}), or 'q' to quit:")
                    continue

                run_inference(user_idx, news_dataset, behaviors_dataset, model, tok, conf.DEVICE)
                print(f"Enter behavior index (0-{len(behaviors_dataset)-1}), or 'q' to quit:")

            except ValueError:
                print("Enter a valid integer index or 'q' to quit.")
                print(f"Enter behavior index (0-{len(behaviors_dataset)-1}), or 'q' to quit:")
            except KeyboardInterrupt:
                print("\nGoodbye.")
                break
            except Exception as e:
                print(f"Error: {e}")
                print(f"Enter behavior index (0-{len(behaviors_dataset)-1}), or 'q' to quit:")


if __name__ == "__main__":
    main()
