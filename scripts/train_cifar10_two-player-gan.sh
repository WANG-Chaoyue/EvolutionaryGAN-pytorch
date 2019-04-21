set -ex
python train.py --dataroot None --name two_player_gan_cifar10 \
       --dataset_mode torchvision --batch_size 128 \
       --model two_player_gan \
       --download_root ./datasets/cifar10 --dataset_name CIFAR10 \
       --crop_size 32 --load_size 32 \
       --d_loss_mode lsgan --g_loss_mode lsgan --which_D Ra \
       --netD DCGAN_cifar10 --netG DCGAN_cifar10 \
       --ngf 128 --ndf 128 --norm batch \
       --init_type normal --init_gain 0.2 \
       --no_dropout --no_flip \
       --D_iters 1 \
       --score_name FID IS --evaluation_size 50000 --fid_batch_size 500 --fid_stat_file ./TTUR/stats/fid_stats_cifar10_train.npz \
       --print_freq 2000 --display_freq 2000 --score_freq 5000 --display_id -1 --save_giters_freq 100000
