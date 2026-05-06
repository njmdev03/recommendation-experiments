# MIND Recommendation Models

Recommendation models based on the Microsoft _ News Dataset.

## Setup

(optional) install pytorch with support for your accelerator (CUDA) according to their [Getting Started]() page.

```bash

```

Install requirements

```bash
pip install -r requirements.txt
```

### Get the dataset

Download MIND from []() and extract it into a folder named `data/`. This should result in the following structure:

```
data
    MINDsmall_dev
    MINDsmall_train
    MINDlarge_dev # (optional)
    MINDlarge_train # (optional)
```

Optionally add glove models in `data/glove/` to use glove for embeddings.

## Run

The project comes with five scripts. stats.py gives basic dataset information. Run it with

```bash
python -m stats
```

train.py sets up and trains actual models. The runs are configured according to config.py. Example configs are provided in the `configs/` directory.

```bash
python -m train
```

Training can be configured to run evaluation during each epoch, however a separate evaluation script is also provided. This script uses the same config.py as training, but will load model checkpoints from the directory training is configured to save them too.

```bash
python -m eval
```

A report script is provided to generate charts and markdown reports of your runs.

```bash
python -m report --run_dir "path/to/training/output" --report --plot_all
```

Finally, an inference script allows you to see the models predictions from a given output.

```bash
python -m inference
```

## Results

The scripts were used to run six experiments, split between an attention model similar in architecture to NRMS and a full dual encoder transformer model. For each model, random embeddings and GloVe 6B 100d frozen and fine tuned embeddings were tested.

Evaluation metrics AUC, MRR, NDk@5, and NDk@10 were collected since they are standard for the MIND challenge. Top models at the time of testing achieved and AUC of ___, MRR of ___, NDk@5 of ___, and NDK@10 ___.

NRMS models were trained on a mobile 4060, transformer models were run on a 2070 Super. Runtimes within a model class can be compared, but NRMS cannot be compared to transformer, although NRMS runs much faster than transformer.

### Transformers
