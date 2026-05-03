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
            ckpt = torch.load(path, map_location=self.device)
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

            rows.append({
                "epoch": ckpt.get("epoch", -1),  # SAFE DEFAULT
                "path": ckpt.get("path"),
                **metrics
            })

        df = pd.DataFrame(rows)

        # IMPORTANT: only sort if column exists and is valid
        if "epoch" in df.columns and df["epoch"].notna().any():
            df = df.sort_values("epoch")

        self.metrics_table = df
        return df

    # -------------------------
    # Inspection utilities
    # -------------------------
    def print_summary(self):
        """
        Prints a quick overview of the run.
        """
        df = self.build_metrics_table()

        print("\n=== RUN SUMMARY ===")
        print(f"Run directory: {self.run_dir}")
        print(f"Total checkpoints: {len(self.checkpoints)}")

        if not df.empty:
            print("\nLatest metrics:")
            print(df.iloc[-1].to_string(index=False))

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


# -------------------------
# CLI usage example
# -------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=str, required=True)
    parser.add_argument("--plot", type=str, default=None)
    parser.add_argument("--export_csv", action="store_true")
    parser.add_argument("--export_json", action="store_true")

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