"""Dataset class template

This module provides a template for users to implement custom datasets.
You can specify '--dataset_mode template' to use this dataset.
The class name should be consistent with both the filename and its dataset_mode option.
The filename should be <dataset_mode>_dataset.py
The class name should be <Dataset_mode>Dataset.py
You need to implement the following functions:
    -- <modify_commandline_options>:　Add dataset-specific options and rewrite default values for existing options.
    -- <__init__>: Initialize this dataset class.
    -- <__getitem__>: Return a data point and its metadata information.
    -- <__len__>: Return the number of images.
"""
from data.base_dataset import BaseDataset, get_transform
# from data.image_folder import make_dataset
# from PIL import Image

class TorchvisionDataset(BaseDataset):
    """A template dataset class for you to implement custom datasets."""
    @staticmethod
    def modify_commandline_options(parser, is_train):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        parser.add_argument('--download_root', type=str, default='./datasets', help='root directory of dataset exist or will be saved')
        parser.add_argument('--dataset_name', type=str, default='CIFAR10', help='name of imported dataset. CIFAR10 | CIFAR100')
        #parser.set_defaults(max_dataset_size=10, new_dataset_option=2.0)  # specify dataset-specific default values
        return parser

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions

        A few things can be done here.
        - save the options (have been done in BaseDataset)
        - get image paths and meta information of the dataset.
        - define the image transformation.
        """
        # save the option and dataset root
        BaseDataset.__init__(self, opt)
        # define the default transform function. You can use <base_dataset.get_transform>; You can also define your custom transform function
        self.transform = get_transform(opt)
        
        # import torchvision dataset
        if opt.dataset_name == 'CIFAR10':
            from torchvision.datasets import CIFAR10 as torchvisionlib
        elif opt.dataset_name == 'CIFAR100':
            from torchvision.datasets import CIFAR100 as torchvisionlib
        else:
            raise ValueError('torchvision_dataset import fault.')

        self.dataload = torchvisionlib(root = opt.download_root,
                                       transform = self.transform,
                                       download = True)
    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index -- a random integer for data indexing

        Returns:
            a dictionary of data with their names. It usually contains the data itself and its metadata information.

        """
        item = self.dataload.__getitem__(index)
        img = item[0]
        label = item[1]

        return {'image': img, 'target': label}

    def __len__(self):
        """Return the total number of images."""
        return len(self.dataload)
