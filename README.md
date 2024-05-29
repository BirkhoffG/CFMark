# Watermarking Counterfactual Explanations

This codebase contains code to reproduce our paper submitted to NeurIPS 2024.

## Install

This project uses 
[jax-relax](https://github.com/BirkhoffG/jax-relax/) (a fast and scalable recourse explanation library).

```sh
pip install -e ".[dev]" --upgrade
```

## Run Experiments

```sh
python -m scripts.run_all
```

Edit the `scripts/run_all.py` file to specify the datasets, CF methods, and attack methods to run.
