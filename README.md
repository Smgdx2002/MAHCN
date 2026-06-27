# MAHCN

This is the PyTorch implementation for the paper: “MAHCN: Multiple Adaptive Hypergraphs Collaboration Network for Next Point-of-Interest Recommendation”.

## Requirements

The code has been checked with Python 3.9. Install the main dependencies with:

```bash
pip install torch numpy scipy pyyaml tqdm
```

Install the PyTorch build that matches your CUDA or CPU environment.

## Raw Data

https://sites.google.com/site/yangdingqi/home/foursquare-dataset

https://snap.stanford.edu/data/loc-gowalla.html


## Run

From the project root:

```bash
python run.py
```

Run with explicit common options:

```bash
python run.py --num_epochs 100 --batch_size 64 --deviceID 0
```

## Outputs

Each run creates a timestamped directory under `logs/` by default:

```text
logs/YYYYMMDD_HHMMSS/
```