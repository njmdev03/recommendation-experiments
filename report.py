import os
import json
import glob
import torch
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Optional


class RunReport:
    """
    Loads checkpoints and metrics from a training run and provides
    utilities for inspection, aggregation, and export.
    """

    def __init__(self, run_dir: str, device: str = "cpu"):
        self.run_dir = run_dir
        self.device = device

        self.checkpoint_dir = os.path.join(run_dir, "checkpoints")
        self.eval_dir = os.path.join(run_dir, "eval")

        self.checkpoints = []
        self.metrics_table = None

        print("RAW checkpoint_dir:", self.checkpoint_dir)
        print("ABS checkpoint_dir:", os.path.abspath(self.checkpoint_dir))
        print("Exists:", os.path.exists(self.checkpoint_dir))
        print("Files:", os.listdir(self.checkpoint_dir) if os.path.exists(self.checkpoint_dir) else None)


    # -------------------------
    # Loading
    # -------------------------
    def load_checkpoints(self) -> List[Dict]:
        """
        Loads all checkpoints in chronological order.
        """
        ckpt_paths = sorted(glob.glob(os.path.join(self.checkpoint_dir, "*.pt")))

        checkpoints = []
        for path in ckpt_paths:
            ckpt = torch.load(path, map_location=self.device, weights_only=False)
            ckpt["path"] = path
            checkpoints.append(ckpt)

        self.checkpoints = checkpoints
        return checkpoints

    def load_eval_metrics(self, filename: str = "metrics.json") -> Optional[Dict]:
        """
        Loads evaluation metrics if available.
        """
        path = os.path.join(self.eval_dir, filename)
        if not os.path.exists(path):
            return None

        with open(path, "r") as f:
            return json.load(f)

    # -------------------------
    # Metrics extraction
    # -------------------------
    def build_metrics_table(self) -> pd.DataFrame:
        if not self.checkpoints:
            self.load_checkpoints()

        rows = []
        for ckpt in self.checkpoints:
            metrics = ckpt.get("metrics") or {}

            row = {
                "epoch": ckpt.get("epoch", -1),
                "path": ckpt.get("path"),
            }

            # Flatten train/eval metrics
            for split in ["train", "eval"]:
                split_metrics = metrics.get(split, {})
                for k, v in split_metrics.items():
                    row[f"{split}_{k}"] = v

            rows.append(row)

        df = pd.DataFrame(rows)

        if "epoch" in df.columns and df["epoch"].notna().any():
            df = df.sort_values("epoch")

        self.metrics_table = df
        return df

    def get_run_info(self) -> Dict:
        """
        Returns basic info about the model architecture and embeddings from config.
        """
        import config as conf

        info = {
            "model": conf.MODEL.value if hasattr(conf.MODEL, "value") else str(conf.MODEL),
            "embedding": conf.EMBEDDING.value if hasattr(conf.EMBEDDING, "value") else str(conf.EMBEDDING),
            "lr": conf.LEARNING_RATE,
            "batch_size": conf.BATCH_SIZE,
            "optimizer": "Adam", # Hardcoded in train.py
            "vocab_path": conf.VOCAB
        }

        # Try to get extra info
        try:
            if os.path.exists(conf.VOCAB):
                with open(conf.VOCAB, "r") as f:
                    vocab = json.load(f)
                info["vocab_size"] = len(vocab)

                if conf.EMBEDDING == conf.Embeddings.GLOVE:
                    # Logic similar to utils.py
                    glove_path = None
                    if conf.GLOVE_TYPE == conf.Glove.GLOVE_6B_50: glove_path = "./data/glove/glove.6B/glove.6B.50d.txt"
                    elif conf.GLOVE_TYPE == conf.Glove.GLOVE_6B_100: glove_path = "./data/glove/glove.6B/glove.6B.100d.txt"
                    elif conf.GLOVE_TYPE == conf.Glove.GLOVE_6B_200: glove_path = "./data/glove/glove.6B/glove.6B.200d.txt"
                    elif conf.GLOVE_TYPE == conf.Glove.GLOVE_6B_300: glove_path = "./data/glove/glove.6B/glove.6B.300d.txt"
                    elif conf.GLOVE_TYPE == conf.Glove.GLOVE_42B_300: glove_path = "./data/glove/glove.42B.300d.txt"

                    if glove_path and os.path.exists(glove_path):
                        words = set()
                        with open(glove_path, "r", encoding="utf-8") as f:
                            for line in f:
                                words.add(line.split()[0])

                        hits = sum(1 for w in vocab.keys() if w in words or w.lower() in words)
                        info["glove_coverage"] = hits / len(vocab)
                        info["glove_hits"] = hits
        except Exception as e:
            print(f"Warning: Could not calculate extra stats: {e}")

        return info

    # -------------------------
    # Inspection utilities
    # -------------------------
    def print_summary(self):
        """
        Prints a quick overview of the run.
        """
        df = self.build_metrics_table()
        info = self.get_run_info()

        print("\n" + "="*40)
        print("         RUN SUMMARY")
        print("="*40)
        print(f"Run directory: {self.run_dir}")
        print(f"Model:         {info['model']}")
        print(f"Embedding:     {info['embedding']}")
        print(f"LR:            {info['lr']}")
        print(f"Batch Size:    {info['batch_size']}")

        if "vocab_size" in info:
            print(f"Vocab Size:    {info['vocab_size']}")
        if "glove_coverage" in info:
            print(f"GloVe Coverage: {info['glove_coverage']:.2%} ({info['glove_hits']}/{info['vocab_size']})")

        print(f"Total Epochs:  {len(self.checkpoints)}")
        print("-" * 40)

        if not df.empty:
            last_row = df.iloc[-1]
            print("\nLatest Metrics (Epoch {}):".format(int(last_row['epoch'])))

            # Print train vs eval side by side if both exist
            metrics = set()
            for col in df.columns:
                if col.startswith("train_") or col.startswith("eval_"):
                    metrics.add(col.split("_", 1)[1])

            for m in sorted(list(metrics)):
                t_val = last_row.get(f"train_{m}", "N/A")
                e_val = last_row.get(f"eval_{m}", "N/A")

                t_str = f"{t_val:.4f}" if isinstance(t_val, (float, int)) else str(t_val)
                e_str = f"{e_val:.4f}" if isinstance(e_val, (float, int)) else str(e_val)

                print(f"{m:15} | Train: {t_str:8} | Eval: {e_str:8}")

        print("="*40)

    def best_epoch(self, metric: str, maximize: bool = True):
        """
        Returns best epoch according to a metric.
        """
        df = self.build_metrics_table()
        if metric not in df.columns:
            raise ValueError(f"Metric '{metric}' not found in logs.")

        idx = df[metric].idxmax() if maximize else df[metric].idxmin()
        return df.loc[idx]

    # -------------------------
    # Export utilities
    # -------------------------
    def export_csv(self, out_path: Optional[str] = None):
        df = self.build_metrics_table()

        out_path = out_path or os.path.join(self.run_dir, "metrics.csv")
        df.to_csv(out_path, index=False)
        print(f"Saved CSV -> {out_path}")

    def export_json(self, out_path: Optional[str] = None):
        df = self.build_metrics_table()

        out_path = out_path or os.path.join(self.run_dir, "metrics.json")
        df.to_json(out_path, orient="records", indent=2)
        print(f"Saved JSON -> {out_path}")

    # -------------------------
    # Plotting
    # -------------------------
    def plot_metric(self, metric: str, save_path: Optional[str] = None):
        df = self.build_metrics_table()

        if metric not in df.columns:
            raise ValueError(f"Metric '{metric}' not found.")

        plt.figure()
        plt.plot(df["epoch"], df[metric], marker="o")
        plt.title(f"{metric} over epochs")
        plt.xlabel("Epoch")
        plt.ylabel(metric)
        plt.grid(True)

        if save_path is None:
            save_path = os.path.join(self.run_dir, f"{metric}_curve.png")

        plt.savefig(save_path)
        plt.close()

        print(f"Saved plot -> {save_path}")

    def plot_curves(self):
        """
        Plots training and validation curves for loss and available metrics.
        """
        df = self.build_metrics_table()
        if df.empty:
            return

        metrics = set()
        for col in df.columns:
            if col.startswith("train_") or col.startswith("eval_"):
                metrics.add(col.split("_", 1)[1])

        for m in metrics:
            plt.figure(figsize=(10, 6))

            has_train = f"train_{m}" in df.columns
            has_eval = f"eval_{m}" in df.columns

            if has_train:
                plt.plot(df["epoch"], df[f"train_{m}"], label=f"Train {m}", marker='o')
            if has_eval:
                plt.plot(df["epoch"], df[f"eval_{m}"], label=f"Eval {m}", marker='s')

            plt.title(f"{m} over Epochs")
            plt.xlabel("Epoch")
            plt.ylabel(m)
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.7)

            save_path = os.path.join(self.run_dir, f"{m}_curve.png")
            plt.savefig(save_path)
            plt.close()
            print(f"Saved {m} curve -> {save_path}")

    def generate_markdown_report(self):
        """
        Generates a markdown report for academic purposes.
        """
        df = self.build_metrics_table()
        info = self.get_run_info()

        report = []
        report.append(f"# Experiment Report: {os.path.basename(self.run_dir)}")
        report.append(f"\n## Configuration")
        report.append(f"- **Model Architecture:** {info['model']}")
        report.append(f"- **Embeddings:** {info['embedding']}")
        report.append(f"- **Learning Rate:** {info['lr']}")
        report.append(f"- **Batch Size:** {info['batch_size']}")

        if "vocab_size" in info:
            report.append(f"- **Vocab Size:** {info['vocab_size']}")
        if "glove_coverage" in info:
            report.append(f"- **GloVe Coverage:** {info['glove_coverage']:.2%} ({info['glove_hits']}/{info['vocab_size']})")

        report.append(f"\n## Training Progress")

        # Filter columns for display
        cols_to_show = ["epoch"] + [c for c in df.columns if c.startswith("train_") or c.startswith("eval_")]
        cols_to_show = [c for c in cols_to_show if c in df.columns]

        try:
            report.append(df[cols_to_show].to_markdown(index=False))
        except ImportError:
            report.append(df[cols_to_show].to_string(index=False))

        report_path = os.path.join(self.run_dir, "report.md")
        with open(report_path, "w") as f:
            f.write("\n".join(report))

        print(f"Saved Markdown report -> {report_path}")


# -------------------------
# CLI usage example
# -------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=str, required=True)
    parser.add_argument("--plot", type=str, default=None, help="Metric to plot individually")
    parser.add_argument("--plot_all", action="store_true", help="Plot all metrics (train vs eval)")
    parser.add_argument("--export_csv", action="store_true")
    parser.add_argument("--export_json", action="store_true")
    parser.add_argument("--report", action="store_true", help="Generate Markdown report")

    args = parser.parse_args()

    report = RunReport(args.run_dir)
    report.load_checkpoints()
    report.print_summary()

    if args.export_csv:
        report.export_csv()

    if args.export_json:
        report.export_json()

    if args.plot:
        report.plot_metric(args.plot)

    if args.plot_all:
        report.plot_curves()

    if args.report:
        report.generate_markdown_report()