set -ex
python train.py --dataroot None --name two_player_gan_cifar10 \
       --dataset_mode torchvision --batch_size 64 \
       --model two_player_gan \
       --download_root ./datasets/cifar10 --dataset_name CIFAR10 \
       --crop_size 32 --load_size 32 \
       --d_loss_mode lsgan --g_loss_mode lsgan --which_D S \
       --netD DCGAN_cifar10 --netG DCGAN_cifar10 \
       --no_dropout --no_flip \
       --D_iters 1 \
       --print_freq 1000 --display_freq 1000 --save_giters_freq 100000
