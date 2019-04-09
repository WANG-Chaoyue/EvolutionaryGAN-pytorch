set -ex
python train.py --dataroot ~/Datasets/CelebA/ --name egan_celeba --dataset_mode hdf5 --batch_size 64\
       --hdf5_filename img_align_celeba_128.hdf5 --num_threads 0 
