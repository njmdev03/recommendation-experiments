# MIND Recommendation Models

News recommendation systems using attention and transformer architectures applied to the Microsoft MIND dataset. This project compares two model classes—NRMS-style attention models and dual-encoder transformers—across different embedding strategies to evaluate trade-offs between accuracy and computational efficiency.

## Overview

News recommendation presents unique challenges: handling sparse user interaction data, capturing temporal relevance, and processing high-dimensional text efficiently. This project implements two contrasting architectures to address these challenges. NRMS (Neural News Recommendation with Multi-Head Self-Attention) uses multi-head attention with additive pooling for news encoding and user behavior modeling. The transformer variant employs stacked transformer blocks with positional encoding, offering more expressive power at increased computational cost.

Experiments test three embedding strategies: simple trained embeddings from scratch, frozen pre-trained GloVe embeddings, and fine-tuned GloVe embeddings. Results demonstrate that embedding strategy significantly affects model performance, with GloVe embeddings outperforming simple embeddings on smaller datasets.

## Setup

### Environment

Install PyTorch with CUDA support (optional but recommended):

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

Install project dependencies:

```bash
pip install -r requirements.txt
```

### Dataset

Download MIND (Microsoft News Dataset) from the [official source](https://msnews.github.io/). Extract into `data/` directory:

```
data/
  MINDsmall_train/
  MINDsmall_dev/
  MINDlarge_train/      # optional
  MINDlarge_dev/        # optional
```

The dataset contains user reading histories, news articles with metadata, and implicit feedback signals. MIND small contains approximately 1M impressions from 100k users with 65k news articles.

### Embeddings (Optional)

Pre-trained GloVe embeddings accelerate training and often improve results. Download from the [GloVe project](https://nlp.stanford.edu/projects/glove/) and organize in `data/glove/`:

```
data/glove/
  glove.6B/
    glove.6B.100d.txt
    glove.6B.300d.txt
    # ... other dimensions
```

## Usage

### Dataset Statistics

View dataset characteristics:

```bash
python -m stats
```

### Training

Training is controlled via `config.py`. Select a configuration file from `configs/` and copy it to `config.py`:

```bash
cp configs/nrms_simple_trained.py config.py
python -m train
```

Configuration parameters include model architecture, embedding type, learning rate, batch size, and number of epochs. Checkpoints save to directories specified in config.

### Evaluation

Separate evaluation script loads saved checkpoints and computes metrics on test data:

```bash
python -m eval
```

Uses same configuration as training to locate model checkpoints.

### Visualization

Generate training curves and experiment reports:

```bash
python -m report --run_dir "out/nrms_simple_trained" --report --plot_all
```

Produces markdown reports with performance tables and PNG visualizations.

### Inference

Interactive inference on trained models:

```bash
python -m inference
```

Enter a user ID to inspect the model's reading history, see predicted recommendations, and compare against ground truth.

## Model Architectures

### NRMS: Attention-Based Recommendation

News encoder processes article text using multi-head self-attention followed by additive pooling. Architecture projects embeddings to d_model dimension, applies 16 parallel attention heads, and pools attended representations into fixed-size vectors.

User encoder models reading history as a sequence of encoded news articles. Applies multi-head attention over history and uses additive pooling to generate user representation. Candidate news ranked by dot product similarity with user vector.

Configuration: 16 heads, head_dim=16 (total d_model=256), additive attention for both news and user encoding.

### Transformer: Deep Transformer Model

News encoder stacks transformer blocks (self-attention plus feed-forward layers) with positional encoding. Multiple layers (4 used in experiments) allow deeper feature interaction. Additive pooling aggregates sequence into news vector.

User encoder applies transformer self-attention over history sequence, then pools to user representation. Otherwise identical ranking mechanism to NRMS.

Configuration: 8 heads, 4 transformer layers, 256 dimensions, positional encoding enabled.

Both models share a candidate ranking mechanism: score(user, candidate) = user_vector · candidate_vectors^T. During training, positive article receives label 1 and randomly sampled negatives receive 0. Binary cross-entropy loss optimizes ranking.

## Experiments

### Dataset and Metrics

All experiments use MIND small dataset (100k users, 65k articles, ~1M impressions). Evaluation metrics follow MIND challenge standards:

- **AUC**: Area under ROC curve, measures ranking discrimination ability
- **MRR**: Mean Reciprocal Rank, rewards correct positive ranking
- **NDCG@5**: Normalized Discounted Cumulative Gain at rank 5
- **NDCG@10**: Normalized Discounted Cumulative Gain at rank 10

Training measured loss and metrics. Inference time tracked per batch for efficiency analysis.

### Configurations

Eight model configurations tested:

| Model                       | Embeddings    | Details                                       |
|-----------------------------|---------------|-----------------------------------------------|
| NRMS Simple                 | Trained       | Embeddings learned from random initialization |
| NRMS GloVe Frozen           | 100d GloVe 6B | Pre-trained embeddings, no gradient updates   |
| NRMS GloVe Fine-tuned       | 100d GloVe 6B | Pre-trained embeddings, trainable             |
| Transformer Simple          | Trained       | Simple embeddings from scratch                |
| Transformer GloVe Frozen    | 100d GloVe 6B | Frozen pre-trained embeddings                 |
| Transformer GloVe Trainable | 100d GloVe 6B | Fine-tuned pre-trained embeddings             |

Hyperparameters: learning rate 1e-4, 3 epochs, batch sizes 12-16 (transformer/NRMS respectively).

## Results

### Performance Comparison

**NRMS Models** (trained on NVIDIA RTX 4060 mobile):

| Configuration    | Epoch 3 AUC | Epoch 3 MRR | NDCG@5    | NDCG@10   | Inference Time/Batch |
|------------------|-------------|-------------|-----------|-----------|----------------------|
| Simple           | 0.629       | 0.352       | 0.348     | 0.422     | 2.4ms                |
| GloVe Frozen     | **0.642**   | **0.369**   | **0.388** | **0.462** | 2.2ms                |
| GloVe Fine-tuned | 0.638       | **0.369**   | 0.387     | 0.461     | **2.1ms**            |

**Transformer Models** (trained on NVIDIA RTX 2070 Super):

| Configuration   | Epoch 3 AUC | Epoch 3 MRR | NDCG@5    | NDCG@10   | Inference Time/Batch |
|-----------------|-------------|-------------|-----------|-----------|----------------------|
| Simple          | 0.588       | 0.315       | 0.335     | 0.413     | 4.2-5.7ms            |
| GloVe Frozen    | **0.610**   | 0.321       | **0.345** | 0.418     | 5.4-5.7ms            |
| GloVe Trainable | 0.604       | **0.322**   | 0.344     | **0.419** | **4.2ms**            |

### Key Findings

**Embedding Impact**: GloVe embeddings improved NRMS performance by 2.1% AUC and 4.8% MRR compared to simple embeddings. Improvement smaller for transformers (3.7% AUC, 1.9% MRR), suggesting transformers learn representations less efficiently from random initialization.

**Architecture Comparison**: NRMS significantly outperforms transformers on this dataset: 7.3% higher AUC and 11.7% higher MRR at best configuration. Performance gap suggests NRMS better suited to smaller datasets with limited training data.

**Computational Efficiency**: NRMS achieves ~2.5x faster inference (2.2ms vs 5.4ms per batch). Training time per epoch comparable within architecture class (NRMS ~1300s, Transformer ~2500s per epoch on their respective GPUs). Cannot directly compare across GPU types.

**Training Dynamics**: All models converged within 3 epochs. NRMS loss decreased more consistently. Transformer loss showed higher variance, particularly with simple embeddings, indicating optimization challenges with deeper architecture.

**Fine-tuning Effect**: Freezing vs fine-tuning GloVe embeddings showed minimal difference (< 0.5% gap), suggesting pre-trained embeddings well-calibrated for this task and retraining offers limited benefit.

## Architecture Details

### Project Structure

```
configs/           # Experiment configurations
data/              # Dataset storage (MIND, GloVe)
data_wrappers/     # Dataset loading and preprocessing
models/            # PyTorch model implementations
tokenizing/        # Text tokenization
training/          # Training engine and evaluation
train.py           # Training entry point
eval.py            # Evaluation script
inference.py       # Interactive inference
report.py          # Report generation
stats.py           # Dataset statistics
```

### Key Components

**Data Pipeline** (`data_wrappers/`): Loads MIND dataset, tokenizes text, creates user-item interaction matrices. Handles variable-length sequences with padding masks.

**Models** (`models/`): NRMS and Transformer implementations with configurable dimensions, attention heads, and layers.

**Training** (`training/`): Distributed training engine with checkpointing, metric computation, and validation loops.

**Tokenization** (`tokenizing/`): Word-level tokenizer builds vocabulary from training data, handles unknown words gracefully.

## Conclusions

NRMS demonstrates superior performance and efficiency on the MIND small dataset. Attention mechanisms effectively capture user preferences with fewer parameters and faster inference than transformers. Pre-trained embeddings provide consistent but modest improvements, suggesting this dataset benefits from general word semantics.

Transformers, while less effective here, offer opportunities for future work: deeper architectures might improve with larger datasets and longer training, positional encoding could be enhanced for temporal news dynamics, and different pooling strategies might better aggregate history.

For production recommendation systems on resource-constrained devices, NRMS offers compelling trade-off between accuracy and latency. For research exploring language modeling and deeper architectures, transformer implementation provides extensible foundation.

The codebase supports easy configuration changes and new experiments. Adding new embedding types, architectures, or datasets requires minimal modifications to config and data wrapper classes.
