import os
import torch
import numpy as np
from collections import OrderedDict
from abc import ABC, abstractmethod
from . import networks

from collections import OrderedDict
from torch.distributions import Categorical
from util.util import prepare_z_y, one_hot, visualize_imgs 
from TTUR import fid
from util.inception import get_inception_score
from inception_pytorch import inception_utils

class BaseModel(ABC):
    """This class is an abstract base class (ABC) for models.
    To create a subclass, you need to implement the following five functions:
        -- <__init__>:                      initialize the class; first call BaseModel.__init__(self, opt).
        -- <set_input>:                     unpack data from dataset and apply preprocessing.
        -- <forward>:                       produce intermediate results.
        -- <optimize_parameters>:           calculate losses, gradients, and update network weights.
        -- <modify_commandline_options>:    (optionally) add model-specific options and set default options.
    """

    def __init__(self, opt):
        """Initialize the BaseModel class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions

        When creating your custom class, you need to implement your own initialization.
        In this fucntion, you should first call <BaseModel.__init__(self, opt)>
        Then, you need to define four lists:
            -- self.loss_names (str list):          specify the training losses that you want to plot and save.
            -- self.model_names (str list):         specify the images that you want to display and save.
            -- self.visual_names (str list):        define networks used in our training.
            -- self.optimizers (optimizer list):    define and initialize optimizers. You can define one optimizer for each network. If two networks are updated at the same time, you can use itertools.chain to group them. See cycle_gan_model.py for an example.
        """
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.isTrain = opt.isTrain
        self.device = torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')  # get device name: CPU or GPU
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)  # save all the checkpoints to save_dir
        if opt.preprocess != 'scale_width':  # with [scale_width], input images might have different sizes, which hurts the performance of cudnn.benchmark.
            torch.backends.cudnn.benchmark = True
        self.loss_names = []
        self.model_names = []
        self.visual_names = []
        self.optimizers = []
        self.image_paths = []
        self.metric = 0  # used for learning rate policy 'plateau'

        # scores init
        if self.opt.use_pytorch_scores and self.opt.score_name is not None:
            no_FID = True
            no_IS = True
            parallel = len(opt.gpu_ids) > 1 
            for name in self.opt.score_name:
                if name == 'FID':
                    no_FID = False 
                if name == 'IS':
                    no_IS = False 
            self.get_inception_metrics = inception_utils.prepare_inception_metrics(opt.dataset_name, parallel, no_IS, no_FID) 
        else:
            for name in self.opt.score_name:
                if name == 'FID':
                    STAT_FILE = self.opt.fid_stat_file
                    INCEPTION_PATH = "./inception_v3/"

                    print("load train stats.. ")
                    # load precalculated training set statistics
                    f = np.load(STAT_FILE)
                    self.mu_real, self.sigma_real = f['mu'][:], f['sigma'][:]
                    f.close()
                    print("ok")

                    inception_path = fid.check_or_download_inception(INCEPTION_PATH) # download inception network
                    fid.create_inception_graph(inception_path)  # load the graph into the current TF graph

                    config = tf.ConfigProto()
                    config.gpu_options.allow_growth = True
                    self.sess = tf.Session(config = config)
                    self.sess.run(tf.global_variables_initializer())

    @staticmethod
    def modify_commandline_options(parser, is_train):
        """Add new model-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        return parser

    @abstractmethod
    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): includes the data itself and its metadata information.
        """
        pass

    @abstractmethod
    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        pass

    @abstractmethod
    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        pass

    def setup(self, opt):
        """Load and print networks; create schedulers

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        if self.isTrain:
            self.schedulers = [networks.get_scheduler(optimizer, opt) for optimizer in self.optimizers]
        if not self.isTrain or opt.continue_train:
            load_suffix = 'iter_%d' % opt.load_iter if opt.load_iter > 0 else opt.epoch
            self.load_networks(load_suffix)
        self.print_networks(opt.verbose)

    def eval(self):
        """Make models eval mode during test time"""
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                net.eval()

    def test(self):
        """Forward function used in test time.

        This function wraps <forward> function in no_grad() so we don't save intermediate steps for backprop
        It also calls <compute_visuals> to produce additional visualization results
        """
        with torch.no_grad():
            self.forward()
            self.compute_visuals()

    def compute_visuals(self):
        """Calculate additional output images for visdom and HTML visualization"""
        pass

    def get_image_paths(self):
        """ Return image paths that are used to load current data"""
        return self.image_paths

    def update_learning_rate(self):
        """Update learning rates for all the networks; called at the end of every epoch"""
        for scheduler in self.schedulers:
            if self.opt.lr_policy == 'plateau':
                scheduler.step(self.metric)
            else:
                scheduler.step()

        lr = self.optimizers[0].param_groups[0]['lr']
        print('learning rate = %.7f' % lr)

    def get_current_visuals(self):
        """Return visualization images. train.py will display these images with visdom, and save the images to a HTML"""
        visual_ret = OrderedDict()
        for name in self.visual_names:
            if isinstance(name, str):
                visual_ret[name] = getattr(self, name)
        return visual_ret

    def get_current_losses(self):
        """Return traning losses / errors. train.py will print out these errors on console, and save them to a file"""
        errors_ret = OrderedDict()
        for name in self.loss_names:
            if isinstance(name, str):
                errors_ret[name] = float(getattr(self, 'loss_' + name))  # float(...) works for both scalar tensor and float number
        return errors_ret

    def save_networks(self, epoch):
        """Save all the networks to the disk.

        Parameters:
            epoch (int) -- current epoch; used in the file name '%s_net_%s.pth' % (epoch, name)
        """
        for name in self.model_names:
            if isinstance(name, str):
                save_filename = '%s_net_%s.pth' % (epoch, name)
                save_path = os.path.join(self.save_dir, save_filename)
                net = getattr(self, 'net' + name)

                if len(self.gpu_ids) > 0 and torch.cuda.is_available():
                    torch.save(net.module.cpu().state_dict(), save_path)
                    net.cuda(self.gpu_ids[0])
                else:
                    torch.save(net.cpu().state_dict(), save_path)

    def __patch_instance_norm_state_dict(self, state_dict, module, keys, i=0):
        """Fix InstanceNorm checkpoints incompatibility (prior to 0.4)"""
        key = keys[i]
        if i + 1 == len(keys):  # at the end, pointing to a parameter/buffer
            if module.__class__.__name__.startswith('InstanceNorm') and \
                    (key == 'running_mean' or key == 'running_var'):
                if getattr(module, key) is None:
                    state_dict.pop('.'.join(keys))
            if module.__class__.__name__.startswith('InstanceNorm') and \
               (key == 'num_batches_tracked'):
                state_dict.pop('.'.join(keys))
        else:
            self.__patch_instance_norm_state_dict(state_dict, getattr(module, key), keys, i + 1)

    def load_networks(self, epoch):
        """Load all the networks from the disk.

        Parameters:
            epoch (int) -- current epoch; used in the file name '%s_net_%s.pth' % (epoch, name)
        """
        for name in self.model_names:
            if isinstance(name, str):
                load_filename = '%s_net_%s.pth' % (epoch, name)
                load_path = os.path.join(self.save_dir, load_filename)
                net = getattr(self, 'net' + name)
                if isinstance(net, torch.nn.DataParallel):
                    net = net.module
                print('loading the model from %s' % load_path)
                # if you are using PyTorch newer than 0.4 (e.g., built from
                # GitHub source), you can remove str() on self.device
                state_dict = torch.load(load_path, map_location=str(self.device))
                if hasattr(state_dict, '_metadata'):
                    del state_dict._metadata

                # patch InstanceNorm checkpoints prior to 0.4
                for key in list(state_dict.keys()):  # need to copy keys here because we mutate in loop
                    self.__patch_instance_norm_state_dict(state_dict, net, key.split('.'))
                net.load_state_dict(state_dict)

    def print_networks(self, verbose):
        """Print the total number of parameters in the network and (if verbose) network architecture

        Parameters:
            verbose (bool) -- if verbose: print the network architecture
        """
        print('---------- Networks initialized -------------')
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                num_params = 0
                for param in net.parameters():
                    num_params += param.numel()
                if verbose:
                    print(net)
                print('[Network %s] Total number of parameters : %.3f M' % (name, num_params / 1e6))
        print('-----------------------------------------------')

    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    # return visualization images. train.py will display these images, and save the images to a html
    def get_current_visuals(self):
        if self.opt.model == 'egan':
            # load current best G
            F = self.Fitness[:,2]
            idx = np.where(F==max(F))[0][0]
            self.netG.load_state_dict(self.G_candis[idx])

        visual_ret = OrderedDict()
        # gen_visual
        if not self.opt.cgan:
            gen_visual = self.netG(self.z_fixed).detach()
        else:
            gen_visual = self.netG(self.z_fixed, self.y_fixed).detach()
        self.gen_visual = visualize_imgs(gen_visual, self.N, self.opt.crop_size, self.opt.input_nc)

        # real_visual
        self.real_visual = visualize_imgs(self.real_imgs, self.N, self.opt.crop_size, self.opt.input_nc)

        for name in self.visual_names:
            if isinstance(name, str):
                visual_ret[name] = getattr(self, name)
        return visual_ret

    def get_current_scores(self):
        if self.opt.model == 'egan':
            # load current best G
            F = self.Fitness[:,2]
            idx = np.where(F==max(F))[0][0]
            self.netG.load_state_dict(self.G_candis[idx])

        # load current best G
        scores_ret = OrderedDict()

        samples = torch.zeros((self.opt.evaluation_size, 3, self.opt.crop_size, self.opt.crop_size), device=self.device)
        n_fid_batches = self.opt.evaluation_size // self.opt.fid_batch_size

        for i in range(n_fid_batches):
            frm = i * self.opt.fid_batch_size
            to = frm + self.opt.fid_batch_size

            if self.opt.z_type == 'Gaussian': 
                z = torch.randn(self.opt.fid_batch_size, self.opt.z_dim, 1, 1, device=self.device)
            elif self.opt.z_type == 'Uniform': 
                z = torch.rand(self.opt.fid_batch_size, self.opt.z_dim, 1, 1, device=self.device) *2. - 1.

            if self.opt.cgan:
                y = self.CatDis.sample([self.opt.fid_batch_size])
                y = one_hot(y, [self.opt.fid_batch_size])

            if not self.opt.cgan:
                gen_s = self.netG(z).detach()
            else:
                gen_s = self.netG(z, y).detach()
            samples[frm:to] = gen_s
            print("\rgenerate fid sample batch %d/%d " % (i + 1, n_fid_batches))

        print("%d samples generating done"%self.opt.evaluation_size)

        if self.opt.use_pytorch_scores:
            self.IS_mean, self.IS_var, self.FID = self.get_inception_metrics(samples, self.opt.evaluation_size, num_splits=10)
            if 'FID' in self.opt.score_name:
                print(self.FID)
                scores_ret['FID'] = float(self.FID) 
            if 'IS' in self.opt.score_name:
                print(self.IS_mean, self.IS_var)
                scores_ret['IS_mean'] = float(self.IS_mean)
                scores_ret['IS_var'] = float(self.IS_var)

        else:
            # Cast, reshape and transpose (BCHW -> BHWC)
            samples = samples.cpu().numpy()
            samples = ((samples + 1.0) * 127.5).astype('uint8')
            samples = samples.reshape(self.opt.evaluation_size, 3, self.opt.crop_size, self.opt.crop_size)
            samples = samples.transpose(0,2,3,1)
            for name in self.opt.score_name:
                if name == 'FID':
                    mu_gen, sigma_gen = fid.calculate_activation_statistics(samples,
                                          self.sess,
                                          batch_size=self.opt.fid_batch_size,
                                          verbose=True)
                    print("calculate FID:")
                    try:
                        self.FID = fid.calculate_frechet_distance(mu_gen, sigma_gen, self.mu_real, self.sigma_real)
                    except Exception as e:
                        print(e)
                        self.FID=500
                    print(self.FID)
                    scores_ret[name] = float(self.FID)
                if name == 'IS':
                    Imlist = []
                    for i in range(len(samples)):
                        im = samples[i,:,:,:]
                        Imlist.append(im)
                    print(np.array(Imlist).shape)
                    self.IS_mean, self.IS_var = get_inception_score(Imlist)

                    scores_ret['IS_mean'] = float(self.IS_mean)
                    scores_ret['IS_var'] = float(self.IS_var)
                    print(self.IS_mean, self.IS_var)

        return scores_ret
