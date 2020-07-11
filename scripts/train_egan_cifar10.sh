set -ex
python train.py --name egan_cifar10 \
       --dataset_mode torchvision --batch_size 32 --eval_size 256 --dataroot None \
       --model egan \
       --gpu_ids 0 \
       --download_root ./datasets/cifar10 --dataset_name CIFAR10 \
       --crop_size 32 --load_size 32 \
       --d_loss_mode vanilla --g_loss_mode nsgan vanilla lsgan --which_D S \
       --lambda_f 0.01 --candi_num 1 --z_type Uniform --z_dim 100 \
       --netD DCGAN_cifar10 --netG DCGAN_cifar10 --ngf 128 --ndf 128 --g_norm none --d_norm batch \
       --init_type normal --init_gain 0.02 \
       --no_dropout \
       --D_iters 3 \
       --score_name IS --evaluation_size 50000 --fid_batch_size 500 \
       --print_freq 1000 --display_freq 1000 --score_freq 5000 --display_id -1 --save_giters_freq 100000
