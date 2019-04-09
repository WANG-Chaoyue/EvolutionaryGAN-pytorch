set -ex
python train.py --dataroot None --name egam_cifar10 --dataset_mode torchvision --batch_size 64 \
       --download_root ./datasets/cifar10 --dataset_name CIFAR10
