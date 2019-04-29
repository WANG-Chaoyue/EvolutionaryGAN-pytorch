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


## Functions

This code borrows heavily from [pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix), since it provided a flexible and efficient framework for pytorch deep networks training. In this part, we briefly introduce the functions of this code. The author is working on implementing more GAN related functions. 

### Datasets loading

### Model selecting

### Network architecture

### Adversarial losses (mutations)

### Inception metrics

## Citation
If you use this code for your research, please cite our paper.
```
@article{wang2019evolutionary,
  title={Evolutionary generative adversarial networks},
  author={Wang, Chaoyue and Xu, Chang and Yao, Xin and Tao, Dacheng},
  journal={IEEE Transactions on Evolutionary Computation},
  year={2019},
  publisher={IEEE}
}
```

## Related links
[Evolving Generative Adversarial Networks | Two Minute Papers #242](https://www.youtube.com/watch?v=ni6P5KU3SDU&vl=en)

## Acknowledgments
Pytorch framework from [pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix).

Pytorch Inception metrics code from [BigGAN-PyTorch](https://github.com/ajbrock/BigGAN-PyTorch).

TensorFlow Inception Score code from [OpenAI's Improved-GAN.](https://github.com/openai/improved-gan).

TensorFlow FID code from [TTUR](https://github.com/bioinf-jku/TTUR).

