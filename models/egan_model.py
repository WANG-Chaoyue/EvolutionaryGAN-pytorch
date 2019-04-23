"""Model class template

This module provides a template for users to implement custom models.
You can specify '--model template' to use this model.
The class name should be consistent with both the filename and its model option.
The filename should be <model>_dataset.py
The class name should be <Model>Dataset.py
It implements a simple image-to-image translation baseline based on regression loss.
Given input-output pairs (data_A, data_B), it learns a network netG that can minimize the following L1 loss:
    min_<netG> ||netG(data_A) - data_B||_1
You need to implement the following functions:
    <modify_commandline_options>:　Add model-specific options and rewrite default values for existing options.
    <__init__>: Initialize this model class.
    <set_input>: Unpack input data and perform data pre-processing.
    <forward>: Run forward pass. This will be called by both <optimize_parameters> and <test>.
    <optimize_parameters>: Update network weights; it will be called in every training iteration.
"""
import torch
import numpy as np 
import tensorflow as tf 
from .base_model import BaseModel
from . import networks
from util.util import prepare_z_y, one_hot, visualize_imgs 
from torch.distributions import Categorical
from collections import OrderedDict
from TTUR import fid
from util.inception import get_inception_score

import copy 
import pdb 

class EGANModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new model-specific options and rewrite default values for existing options.

        Parameters:
            parser -- the option parser
            is_train -- if it is training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        #parser.set_defaults(dataset_mode='aligned')  # You can rewrite default values for this model. For example, this model usually uses aligned dataset as its dataset
        if is_train:
            parser.add_argument('--g_loss_mode', nargs='*', default=['nsgan','lsgan','vanilla'], help='lsgan | nsgan | vanilla | wgan | hinge | rsgan')
            parser.add_argument('--d_loss_mode', type=str, default='lsgan', help='lsgan | nsgan | vanilla | wgan | hinge | rsgan') 
            parser.add_argument('--which_D', type=str, default='S', help='Standard(S) | Relativistic_average (Ra)') 
            parser.add_argument('--use_gp', action='store_true', default=False, help='if usei gradients penalty')

            parser.add_argument('--lambda_f', type=float, default=0.1, help='the hyperparameter that balance Fq and Fd')
            parser.add_argument('--candi_num', type=int, default=2, help='# of survived candidatures in each evolutinary iteration.')
            parser.add_argument('--eval_size', type=int, default=None, help='evaluation size')
        return parser

    def __init__(self, opt):
        """Initialize this model class.

        Parameters:
            opt -- training/test options

        A few things can be done here.
        - (required) call the initialization function of BaseModel
        - define loss function, visualization images, model names, and optimizers
        """
        BaseModel.__init__(self, opt)  # call the initialization method of BaseModel

        self.opt = opt
        if opt.d_loss_mode == 'wgan' and not opt.use_gp:
            raise NotImplementedError('using wgan on D must be with use_gp = True.')

        self.loss_names = ['G_real', 'G_fake', 'D_real', 'D_fake', 'D_gp', 'G', 'D']
        self.visual_names = ['real_visual', 'gen_visual']

        if self.isTrain:  # only defined during training time
            self.model_names = ['G', 'D']
        else:
            self.model_names = ['G']

        if self.opt.cgan:
            probs = np.ones(self.opt.cat_num)/self.opt.cat_num 
            self.CatDis = Categorical(torch.tensor(probs))

        # define networks 
        self.netG = networks.define_G(opt.z_dim, opt.output_nc, opt.ngf, opt.netG,
                opt.norm, opt.cgan, opt.cat_num, not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:  # define a discriminator; conditional GANs need to take both input and output images; Therefore, #channels for D is input_nc + output_nc
            self.netD = networks.define_D(opt.input_nc, opt.ndf, opt.netD,
                                          opt.norm, opt.cgan, opt.cat_num, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:  # only defined during training time
            # define G mutations 
            self.G_mutations = []
            for g_loss in opt.g_loss_mode: 
                self.G_mutations.append(networks.GANLoss(g_loss, 'G', opt.which_D).to(self.device))
            # define loss functions
            self.criterionD = networks.GANLoss(opt.d_loss_mode, 'D', opt.which_D).to(self.device)
            # initialize optimizers
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr_g, betas=(opt.beta1, opt.beta2))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr_d, betas=(opt.beta1, opt.beta2))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
        
        # Evolutinoary candidatures setting (init) 
        self.G_candis = [] 
        self.optG_candis = [] 
        for i in range(opt.candi_num): 
            self.G_candis.append(copy.deepcopy(self.netG.state_dict()))
            self.optG_candis.append(copy.deepcopy(self.optimizer_G.state_dict()))
        
        # visulize settings 
        self.N =int(np.trunc(np.sqrt(min(opt.batch_size, 64))))
        self.z_fixed = torch.randn(self.N*self.N, opt.z_dim, 1, 1, device=self.device) 
        if self.opt.cgan:
            yf = self.CatDis.sample([self.N*self.N])
            self.y_fixed = one_hot(yf, [self.N*self.N, self.opt.cat_num])

        # scores init
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

        if self.opt.eval_size is None:
            self.opt.eval_size = self.opt.batch_size

    def set_input(self, input):
        """input: a dictionary that contains the data itself and its metadata information."""
        self.real_imgs = input['image'].to(self.device)  
        if self.opt.cgan:
            self.targets = input['target'].to(self.device) 

    def forward(self, batch_size=None):
        bs = self.real_imgs.shape[0] if batch_size is None else batch_size
        z = torch.randn(bs, self.opt.z_dim, 1, 1, device=self.device) 
        # Fake images
        if not self.opt.cgan:
            gen_imgs = self.netG(z)
            y_ = None 
        else:
            y = self.CatDis.sample([bs])
            y_ = one_hot(y, [bs, self.opt.cat_num])
            gen_imgs = self.netG(z, self.y_)
        return gen_imgs, y_

    def backward_G(self, criterionG):
        # pass D 
        if not self.opt.cgan:
            self.fake_out = self.netD(self.gen_imgs)
        else:
            self.fake_out = self.netD(self.gen_imgs, self.y_)

        self.loss_G_fake, self.loss_G_real = criterionG(self.fake_out, self.real_out) 
        self.loss_G = self.loss_G_fake + self.loss_G_real
        self.loss_G.backward() 

    def backward_D(self):
        self.gen_imgs = self.gen_imgs.detach()
        # pass D 
        if not self.opt.cgan:
            self.fake_out = self.netD(self.gen_imgs)
            self.real_out = self.netD(self.real_imgs)
        else:
            self.fake_out = self.netD(self.gen_imgs, self.y_)
            self.real_out = self.netD(self.real_imgs, self.targets)

        self.loss_D_fake, self.loss_D_real = self.criterionD(self.fake_out, self.real_out) 
        if self.opt.use_gp is True: 
            self.loss_D_gp = networks.cal_gradient_penalty(self.netD, self.real_imgs, self.gen_imgs, self.device, type='mixed', constant=1.0, lambda_gp=10.0)[0]
        else:
            self.loss_D_gp = 0.

        self.loss_D = self.loss_D_fake + self.loss_D_real + self.loss_D_gp
        self.loss_D.backward() 

    def optimize_parameters(self, total_steps):
        #self.forward()               
        # update G
        if total_steps % (self.opt.D_iters + 1) == 0:
            self.gen_imgs, self.y_ = self.Evo_G()
        # update D
        else: 
            self.set_requires_grad(self.netD, True)
            self.optimizer_D.zero_grad()
            self.backward_D()
            self.optimizer_D.step()

    def Evo_G(self):
        # define real images pass D
        self.real_out = self.netD(self.real_imgs) if not self.opt.cgan else self.netD(self.real_imgs, self.targets)

        F_list = np.zeros(self.opt.candi_num) - 1e5
        G_list = [] 
        optG_list = [] 
        # variation-evluation-selection
        for i in range(self.opt.candi_num):
            for criterionG in self.G_mutations: 
                # Variation 
                self.netG.load_state_dict(self.G_candis[i])
                self.optimizer_G.load_state_dict(self.optG_candis[i])
                self.optimizer_G.zero_grad()
                self.gen_imgs, self.y_ = self.forward() 
                self.set_requires_grad(self.netD, False)
                self.backward_G(criterionG)
                self.optimizer_G.step()
                # Evaluation 
                with torch.no_grad(): 
                    eval_imgs, eval_y = self.forward(batch_size=self.opt.eval_size) 
                Fq, Fd = self.fitness_score(eval_imgs, eval_y) 
                F = Fq + self.opt.lambda_f * Fd 
                # Selection 
                
        return F_list, G_list, optG_list  

    def fitness_score(self, eval_imgs, eval_y):
        #eval_imgs.requires_grad_(True)
        #self.real_imgs.requires_grad_(True)
        self.set_requires_grad(self.netD, True)
        eval_fake = self.netD(eval_imgs) if not self.opt.cgan else self.netD(eval_imgs, self.y_)
        eval_real = self.netD(self.real_imgs) if not self.opt.cgan else self.netD(self.real_imgs, self.targets)

        # Quality fitness score
        Fq = eval_fake.data.mean().cpu().numpy()

        # Diversity fitness score
        eval_D_fake, eval_D_real = self.criterionD(eval_fake, eval_real) 
        eval_D = eval_D_fake + eval_D_real
        gradients = torch.autograd.grad(outputs=eval_D, inputs=self.netD.parameters(),
                                        grad_outputs=torch.ones(eval_D.size()).to(self.device),
                                        create_graph=True, retain_graph=True, only_inputs=True)
        with torch.no_grad():
            for i, grad in enumerate(gradients):
                grad = grad.view(-1)
                allgrad = grad if i == 0 else torch.cat([allgrad,grad]) 
        Fd = torch.log(torch.norm(allgrad)).data.cpu().numpy()
        pdb.set_trace()
        #gradients = torch.autograd.grad(outputs=eval_D, inputs=torch.cat([eval_imgs, self.real_imgs], dim = 0),
        #                                grad_outputs=torch.ones(eval_D.size()).to(self.device),
        #                                create_graph=True, retain_graph=True, only_inputs=True)
        return Fq, Fd 

    # return visualization images. train.py will display these images, and save the images to a html
    def get_current_visuals(self):
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
        scores_ret = OrderedDict()

        self.z_fixed = torch.randn(self.N*self.N, self.opt.z_dim, 1, 1, device=self.device) 
        if self.opt.cgan:
            yf = self.CatDis.sample([self.N*self.N])
            self.y_fixed = one_hot(yf, [self.N*self.N, self.opt.cat_num])

        sample_generated = False
        for name in self.opt.score_name:

            if sample_generated is False: 
                samples = np.zeros((self.opt.evaluation_size, 3, self.opt.crop_size, self.opt.crop_size))
                n_fid_batches = self.opt.evaluation_size // self.opt.fid_batch_size

                for i in range(n_fid_batches):
                    frm = i * self.opt.fid_batch_size
                    to = frm + self.opt.fid_batch_size

                    z = torch.randn(self.opt.fid_batch_size, self.opt.z_dim, 1, 1, device=self.device)
                    if self.opt.cgan:
                        y = self.CatDis.sample([self.opt.fid_batch_size])
                        y = one_hot(y, [self.opt.fid_batch_size])

                    if not self.opt.cgan:
                        gen_s = self.netG(z).detach()
                    else:
                        gen_s = self.netG(z, y).detach()
                    samples[frm:to] = gen_s.cpu().numpy()
                    print("\rgenerate fid sample batch %d/%d " % (i + 1, n_fid_batches))

                # Cast, reshape and transpose (BCHW -> BHWC)
                samples = ((samples + 1.0) * 127.5).astype('uint8')
                samples = samples.reshape(self.opt.evaluation_size, 3, self.opt.crop_size, self.opt.crop_size)
                samples = samples.transpose(0,2,3,1)

                print("%d samples generating done"%self.opt.evaluation_size)
                sample_generated = True

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
