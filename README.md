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

- Install [PyTorch](https://pytorch.org/get-started/locally/) and other dependencies [requirements.txt](https://github.com/WANG-Chaoyue/EvolutionaryGAN-pytorch/blob/master/requirements.txt) (e.g., torchvision, [visdom](https://github.com/facebookresearch/visdom) and [dominate](https://github.com/Knio/dominate)).

- Preparing *.npz* files for Pytorch Inception metrics evaluation (cifar10 as an example):
```
python inception_pytorch/calculate_inception_moments.py --dataset C10 --data_root datasets
```

### Two-player GANs Training

An example of [LSGAN](https://arxiv.org/abs/1611.04076) training command was saved in [./scripts/train_lsgan_cifar10.sh](). Train a model (cifar10 as an example): 
```bash
bash ./scripts/train_lsgan_cifar10.sh
```
Through configuring args `--g_loss_mode`, `--d_loss_mode` and `--which_D`, different training strategies can be utilized to training the two-player GAN game. **Note** that more explanations of loss settings can be found below. 

### EvolutionaryGAN Training

An example of E-GAN training command was saved in [./scripts/train_egan_cifar10.sh](). Train a model (cifar10 as an example):
```bash
bash ./scripts/train_egan_cifar10.sh
```
Different from Two-player GANs, here the arg `--g_loss_mode` should be set as a list of 'losses' (*e.g.,* `--g_loss_mode vanilla nsgan lsgan`), which are corresponding to different mutations (or variations). 


## Functions

This code borrows heavily from [pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix), since it provided a flexible and efficient framework for pytorch deep networks training. In this part, we briefly introduce the functions of this code. The author is working on implementing more GAN related functions. 

### Datasets loading

- Loading from image folder: [./data/single_dataset.py](https://github.com/WANG-Chaoyue/EvolutionaryGAN-pytorch/blob/master/data/single_dataset.py) `--dataset_mode single`

- Loading from HDF5 file: [./data/hdf5_dataset.py](https://github.com/WANG-Chaoyue/EvolutionaryGAN-pytorch/blob/master/data/hdf5_dataset.py) `--dataset_mode hdf5`

- Loading from [./data/torchvision](https://pytorch.org/docs/stable/torchvision/index.html): [torchvision_dataset.py](https://github.com/WANG-Chaoyue/EvolutionaryGAN-pytorch/blob/master/data/torchvision_dataset.py) `--dataset_mode torchvision` 

### Model selecting

- Two-player GANs: [./models/two_player_gan_model.py](https://github.com/WANG-Chaoyue/EvolutionaryGAN-pytorch/blob/master/models/two_player_gan_model.py) `--model two_player_gan`

- EvolutionaryGAN: [./models/egan_model.py](https://github.com/WANG-Chaoyue/EvolutionaryGAN-pytorch/blob/master/models/egan_model.py) `--model egan`

### Network architecture

- DCGAN-based networks architecture: [./models/DCGAN_nets.py](https://github.com/WANG-Chaoyue/EvolutionaryGAN-pytorch/blob/master/models/DCGAN_nets.py) `--netD DCGAN_cifar10 --netG DCGAN_cifar10`

More architectures will be added.

### Adversarial losses (mutations)

**Standard GANs**: `--which_D S`
After the GAN was first proposed by Goodfellow et al., many different adversarial losses have been devised. Generally, they can be described in the following way:

![General_form](imgs/General_form.gif?raw=true "General_form")

Through defining different functions *f(.)* and *g(.)*, different Standard GAN losses are delivered. 

**Relativistic Average GANs**: `--which_D Ra`
Recently, Alexia proposed the relativistic average GANs ([RaGANs](https://arxiv.org/pdf/1807.00734.pdf)) for GAN training. Its general loss function can be formulated as bellow:

![General_Ra_form](imgs/General_Ra_form.gif?raw=true "General_form")

Note that functions *f(.)* and *g(.)* are defined similarly with Standard GANs, yet the average term of both real and fake images are further considered. 

Through setting `--which_D`, we can basically select the general form of GAN losses. Then, through configuring `--d_loss_mode` and `--g_loss_mode`, the specific losses of discriminator and generator can be determined. Specifically, 

- The [original GAN](https://arxiv.org/abs/1406.2661) (or minimax GAN) losses: `--d_loss_mode vanilla` or `--g_loss_mode vanilla`.

- The [non-saturating GAN](https://arxiv.org/abs/1406.2661) losses: `--d_loss_mode nsgan` or `--g_loss_mode nsgan`.

- The [Least-Squares GAN](https://arxiv.org/abs/1611.04076) losses: `--d_loss_mode lsgan` or `--g_loss_mode lsgan`.

- The [Wasserstein GAN](https://arxiv.org/abs/1704.00028) losses: `--d_loss_mode wgan` or `--g_loss_mode wgan`. Note that Gradients Penalty term should be added `--use_gp`.

- The [Higne GAN](https://arxiv.org/abs/1802.05957) losses: `--d_loss_mode hinge` or `--g_loss_mode hinge`.

- The [Relativistic Standard GAN](https://arxiv.org/abs/1807.00734) losses: `--d_loss_mode rsgan` or `--g_loss_mode rsgan`.

**Note that**, in practice, different kinds of g_loss and d_loss can be combined, and the GP term can also be added into all Discriminators' training.

### Inception metrics

Although many Inception metrics have been proposed to measure generation performance, [Inception Score (IS)](https://arxiv.org/abs/1511.06434) and [Fréchet Inception Distance (FID)](https://github.com/bioinf-jku/TTUR) are two most used. Since both of them are firstly calculated by tensorflow codes, we adopted related codes: TensorFlow Inception Score code from [OpenAI's Improved-GAN](https://github.com/openai/improved-gan) and TensorFlow FID code from [TTUR](https://github.com/bioinf-jku/TTUR). Through setting `--score_name IS`, related scores will be measured during the training process. But, **note that** you will need to have TensorFlow 1.3 or earlier installed, as TF1.4+ breaks the original IS code.

**PyTorch version inception metrics** were adopted from [BigGAN-PyTorch](https://github.com/ajbrock/BigGAN-PyTorch). If you want to use it, simply set `--use_pytorch_scores`. **However**, these scores are different from the scores you would get using the official TF inception code, and are only for monitoring purposes.

- Preparing *.npz* files for Pytorch Inception metrics evaluation (cifar10 as an example):
```
python inception_pytorch/calculate_inception_moments.py --dataset C10 --data_root datasets
```

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

[The best of GAN papers in the year 2018](https://dtransposed.github.io/blog/Best-of-GANs-2018-(Part-1-out-of-2).html)

## Acknowledgments
Pytorch framework from [pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix).

Pytorch Inception metrics code from [BigGAN-PyTorch](https://github.com/ajbrock/BigGAN-PyTorch).

TensorFlow Inception Score code from [OpenAI's Improved-GAN.](https://github.com/openai/improved-gan).

TensorFlow FID code from [TTUR](https://github.com/bioinf-jku/TTUR).

