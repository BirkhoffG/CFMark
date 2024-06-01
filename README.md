# Watermarking Counterfactual Explanations

[![Arxiv](https://img.shields.io/badge/Arxiv-2405.18671-orange)](https://arxiv.org/pdf/2405.18671.pdf)

This codebase contains code to reproduce the paper "[Watermarking Counterfactual Explanations](http://arxiv.org/abs/2405.18671)".

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
