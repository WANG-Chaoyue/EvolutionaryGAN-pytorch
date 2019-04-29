# EvolutionaryGAN-pytorch
The author's officially unofficial Pytorch [EvolutionaryGAN](https://arxiv.org/abs/1803.00657) implementation. The original theano code can be found [here](https://github.com/WANG-Chaoyue/EvolutionaryGAN).

![framework](imgs/EGAN_framework.jpg?raw=true "framework")

The author still working on improving the pytorch version and attempting to add more related functions to achieve better performance. Besides the proposed EGAN farmework, we also provide the [two_player_gan_model](https://github.com/WANG-Chaoyue/EvolutionaryGAN-pytorch/blob/master/models/two_player_gan_model.py) framework that contributes to integrating some existing GAN models together.

## Prerequisites

- Linux or macOS
- Python 3
- CPU or NVIDIA GPU + CUDA CuDNN

## Getting Started
### Installation

- Clone this repo:
```bash
git clone https://github.com/WANG-Chaoyue/EvolutionaryGAN-pytorch.git
cd EvolutionaryGAN-pytorch
```

- Install [PyTorch](https://pytorch.org/get-started/locally/) and other dependencies [requirements.txt]() (e.g., torchvision, [visdom](https://github.com/facebookresearch/visdom) and [dominate](https://github.com/Knio/dominate)).

### Two-player GANs Training

An example of [LSGAN](https://arxiv.org/abs/1611.04076) training command was saved in [./scripts/train_lsgan_cifar10.sh](). Train a model: 
```bash
bash ./scripts/train_lsgan_cifar10.sh
```
Through configuring args `--g_loss_mode`, `--d_loss_mode` and `--which_D`, different training strategies can be utilized to training the two-player GAN game. **Note** that more explanations of loss settings can be found below. 

### EvolutionaryGAN Training

An example of E-GAN training command was saved in [./scripts/train_egan_cifar10.sh](). Train a model:
```bash
bash ./scripts/train_egan_cifar10.sh
```
Different from Two-player GANs, here the arg `--g_loss_mode` should be set as a list of 'losses' (*e.g.,* `--g_loss_mode vanilla nsgan lsgan`), which are corresponding to different mutations (or variations). 


## Settings
### Datasets loading

