# KADC

Code for the KADC method for class-incremental learning with pre-trained Vision Transformers.

## Requirements

`torch==1.12.0`, `torchvision==0.13.0`, `timm==0.5.4`, `tqdm`, `numpy`, `scipy`, `quadprog`, `POT`

## Data

Place ImageNet-R, Cars-196 and CUB-200 under `./data/`. CIFAR-100 is downloaded automatically.

## Running

The `run_kadc*.sh` scripts contain the experiment configurations.

## Acknowledgement

This code builds on [SLDC](https://github.com/raoxuan98-hash/sldc), "Compensating Distribution Drifts in Continual Learning with Pre-trained Vision Transformers".
