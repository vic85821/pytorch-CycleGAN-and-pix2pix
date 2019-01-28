import torch
import itertools
from skimage.exposure import cumulative_distribution
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
from models.loss_network import LossNetwork 
import numpy as np
from torch.autograd import Variable

class CycleGANModel(BaseModel):
    def name(self):
        return 'CycleGANModel'

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        # default CycleGAN did not use dropout
        parser.set_defaults(no_dropout=True)
        if is_train:
            parser.add_argument('--lambda_A', type=float, default=10.0, help='weight for cycle loss (A -> B -> A)')
            parser.add_argument('--lambda_B', type=float, default=10.0,
                                help='weight for cycle loss (B -> A -> B)')
            parser.add_argument('--lambda_identity', type=float, default=0.5, help='use identity mapping. Setting lambda_identity other than 0 has an effect of scaling the weight of the identity mapping loss. For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, please set lambda_identity = 0.1')
            parser.add_argument('--lambda_perceptual', type=float, default=0.005, help='weight for perceptual loss between realA/fakeA and realB/fakeB')
            parser.add_argument('--use_makeup_loss', type=bool, default=True, help='use the makeup-loss for improving the generator training when training beautyGAN')
            parser.add_argument('--lambda_lips', type=float, default=1.0, help='weight for historgram matching of the lips')
            parser.add_argument('--lambda_eye_shadow', type=float, default=1.0, help='weight for historgram matching of the eye shadow')
            parser.add_argument('--lambda_shadow_margin', type=int, default=13, help='margin between the mask of the eye and the bbox')
            parser.add_argument('--lambda_face', type=float, default=0.1, help='weight for historgram matching of the face')
            
        return parser

    def initialize(self, opt):
        BaseModel.initialize(self, opt)

        # specify the training losses you want to print out. The program will call base_model.get_current_losses
        self.loss_names = ['G', 'D_A', 'G_A', 'cycle_A', 'idt_A', 'percept_A', 'D_B', 'G_B', 'cycle_B', 'idt_B', 'percept_B', 'makeup']
        # specify the images you want to save/display. The program will call base_model.get_current_visuals
        visual_names_A = ['real_A', 'fake_B', 'rec_A']
        visual_names_B = ['real_B', 'fake_A', 'rec_B']
        if self.isTrain and self.opt.lambda_identity > 0.0:
            visual_names_A.append('idt_A')
            visual_names_B.append('idt_B')
        if self.isTrain and self.opt.use_makeup_loss == True:
            visual_names_makeup = ['lips_src', 'face_src', 'eye_shadow_left_src', 'eye_shadow_right_src', 
                                   'lips_ref', 'face_ref', 'eye_shadow_left_ref', 'eye_shadow_right_ref',
                                   'lips_real_B', 'face_real_B', 'eye_shadow_left_real_B', 'eye_shadow_right_real_B']
            self.visual_names = visual_names_A + visual_names_B + visual_names_makeup
        else:
            self.visual_names = visual_names_A + visual_names_B
        # specify the models you want to save to the disk. The program will call base_model.save_networks and base_model.load_networks
        if self.isTrain:
            self.model_names = ['G', 'D_A', 'D_B']
        else:  # during test time, only load Gs
            self.model_names = ['G']

        # load/define networks
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            use_sigmoid = opt.no_lsgan
            self.netD_A = networks.define_D(opt.output_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, opt.init_gain, self.gpu_ids)
            self.netD_B = networks.define_D(opt.input_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            self.fake_A_pool = ImagePool(opt.pool_size)
            self.fake_B_pool = ImagePool(opt.pool_size)
            # define loss functions
            self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan).to(self.device)
            self.criterionCycle = torch.nn.L1Loss()
            self.criterionIdt = torch.nn.L1Loss()
            self.criterionPer = torch.nn.MSELoss()
            self.criterionMakeup = torch.nn.MSELoss()
            
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_A.parameters(), self.netD_B.parameters()),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers = []
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    def set_input(self, input):
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']
        if self.opt.phase == 'train':
            self.makeup = input['makeup_mask' if AtoB else 'nonmakeup_mask'].to(self.device)
            self.nonmakeup = input['nonmakeup_mask' if AtoB else 'makeup_mask'].to(self.device)
            self.mask_paths = input['nonmakeup_paths' if AtoB else 'makeup_paths']

    def cdf(self, im, mask):
        #computes the CDF of an image im as 2D numpy ndarray
        tmp = im[mask.reshape((self.imgSize, self.imgSize))]
        c, b = cumulative_distribution(tmp) 
        # pad the beginning and ending pixels and their CDF values
        c = np.insert(c, 0, [0]*b[0])
        c = np.append(c, [1]*(255-b[-1]))
        return c

    def hist_matching(self, c, c_t, im):
        #c: CDF of input image computed with the function cdf()
        #c_t: CDF of template image computed with the function cdf()
        #im: input image as 2D numpy ndarray
        #returns the modified pixel values
        pixels = np.arange(256)
        # find closest pixel-matches corresponding to the CDF of the input image, given the value of the CDF H of   
        # the template image at the corresponding pixels, s.t. c_t = H(pixels) <=> pixels = H-1(c_t)
        new_pixels = np.interp(c, c_t, pixels) 
        im = (np.reshape(new_pixels[im.ravel()], im.shape)).astype(np.uint8)
        return im
        
    def cast_image_to_uint8(self, img):
        return np.around(((img * 0.5) + 0.5) * 255.0).astype(np.uint8)
    
    def normalize(self, img):
        return ((img.astype(np.float32) / 255.0) - 0.5) / 0.5
    
    def from_numpy_to_var(self, arr):
        return Variable(torch.from_numpy(self.normalize(arr).transpose((2, 0, 1)))).type(torch.FloatTensor).cuda().unsqueeze(0)
    
    def get_histogram_matching(self, image_src, image_ref, target, mask_src, mask_ref):
        # get the face, lips, eye shadow region of image_src and image_ref
        # apply histogram matching to image_src to get the similar color distrbution as image_ref       
        mask_src = mask_src.detach().squeeze(0).cpu().numpy().transpose((1, 2, 0))
        mask_ref = mask_ref.detach().squeeze(0).cpu().numpy().transpose((1, 2, 0))
        image_src = image_src.detach().squeeze(0).cpu().numpy().transpose((1, 2, 0))
        image_ref = image_ref.detach().squeeze(0).cpu().numpy().transpose((1, 2, 0))
        target = target.detach().squeeze(0).cpu().numpy().transpose((1, 2, 0))

        mask_src = self.cast_image_to_uint8(mask_src)
        mask_ref = self.cast_image_to_uint8(mask_ref)
        image_src = self.cast_image_to_uint8(image_src)
        image_ref = self.cast_image_to_uint8(image_ref)
        target = self.cast_image_to_uint8(target)
        eye_shadow_margin = self.opt.lambda_shadow_margin        
        
        # Face
        face_src = ((mask_src == 1) | (mask_src == 6)) * image_src
        face_ref = ((mask_ref == 1) | (mask_ref == 6)) * image_ref
        face_target = ((mask_src == 1) | (mask_src == 6)) * target
        self.face_ref = self.from_numpy_to_var(self.hist_matching(self.cdf(face_src, ((mask_src == 1) | (mask_src == 6))), self.cdf(face_ref, ((mask_ref == 1) | (mask_ref == 6))), face_src))
        self.face_src = self.from_numpy_to_var(face_target)
        self.face_real_B = self.from_numpy_to_var(face_ref)
        
        # Lips
        lips_src = ((mask_src == 7) | (mask_src == 9)) * image_src
        lips_ref = ((mask_ref == 7) | (mask_ref == 9)) * image_ref
        lips_target = ((mask_src == 7) | (mask_src == 9)) * target
        self.lips_ref = self.from_numpy_to_var(self.hist_matching(self.cdf(lips_src, ((mask_src == 7) | (mask_src == 9))), self.cdf(lips_ref, ((mask_ref == 7) | (mask_ref == 9))), lips_src))
        self.lips_src = self.from_numpy_to_var(lips_target)
        self.lips_real_B = self.from_numpy_to_var(lips_ref)
        
        # Eye shadow
        cor_right_src = np.where(mask_src == [4])
        cor_right_ref = np.where(mask_ref == [4])
        if (cor_right_src[0].size != 0) and (cor_right_ref[0].size != 0):
            cor_src = np.zeros((self.imgSize, self.imgSize, 1), dtype=bool)
            cor_ref = np.zeros((self.imgSize, self.imgSize, 1), dtype=bool)
            cor_target = np.zeros((self.imgSize, self.imgSize, 1), dtype=bool)
            cor_src[cor_right_src[0].min()-eye_shadow_margin: cor_right_src[0].max()+eye_shadow_margin, cor_right_src[1].min()-eye_shadow_margin: cor_right_src[1].max()+eye_shadow_margin] = True
            cor_ref[cor_right_ref[0].min()-eye_shadow_margin: cor_right_ref[0].max()+eye_shadow_margin, cor_right_ref[1].min()-eye_shadow_margin: cor_right_ref[1].max()+eye_shadow_margin] = True
            cor_target[cor_right_src[0].min()-eye_shadow_margin: cor_right_src[0].max()+eye_shadow_margin, cor_right_src[1].min()-eye_shadow_margin: cor_right_src[1].max()+eye_shadow_margin] = True
            
            eye_shadow_right_src = (image_src * cor_src * (mask_src != 4))
            eye_shadow_right_ref = (image_ref * cor_ref * (mask_ref != 4))
            eye_shadow_right_target = (target * cor_target * (mask_src != 4))
            self.eye_shadow_right_ref = self.from_numpy_to_var(self.hist_matching(self.cdf(eye_shadow_right_src, cor_src * (mask_src != 4)), self.cdf(eye_shadow_right_ref, cor_ref * (mask_ref != 4)), eye_shadow_right_src))
            self.eye_shadow_right_src = self.from_numpy_to_var(eye_shadow_right_target)
            self.eye_shadow_right_real_B = self.from_numpy_to_var(eye_shadow_right_ref)
        else:
            self.eye_shadow_right_src = self.from_numpy_to_var(image_src * 0)
            self.eye_shadow_right_ref = self.from_numpy_to_var(image_ref * 0)
            self.eye_shadow_right_real_B = self.from_numpy_to_var(image_ref * 0)

        cor_left_src = np.where(mask_src == [5])
        cor_left_ref = np.where(mask_ref == [5])
        if (cor_left_src[0].size != 0) and (cor_left_ref[0].size != 0):
            cor_src = np.zeros((self.imgSize, self.imgSize, 1), dtype=bool)
            cor_ref = np.zeros((self.imgSize, self.imgSize, 1), dtype=bool)
            cor_target = np.zeros((self.imgSize, self.imgSize, 1), dtype=bool)
            cor_src[cor_left_src[0].min()-eye_shadow_margin: cor_left_src[0].max()+eye_shadow_margin, cor_left_src[1].min()-eye_shadow_margin: cor_left_src[1].max()+eye_shadow_margin] = True
            cor_ref[cor_left_ref[0].min()-eye_shadow_margin: cor_left_ref[0].max()+eye_shadow_margin, cor_left_ref[1].min()-eye_shadow_margin: cor_left_ref[1].max()+eye_shadow_margin] = True
            cor_target[cor_left_src[0].min()-eye_shadow_margin: cor_left_src[0].max()+eye_shadow_margin, cor_left_src[1].min()-eye_shadow_margin: cor_left_src[1].max()+eye_shadow_margin] = True
            
            eye_shadow_left_src = (image_src * cor_src * (mask_src != 5))
            eye_shadow_left_ref = (image_ref * cor_ref * (mask_ref != 5))
            eye_shadow_left_target = (target * cor_target * (mask_src != 5))
            self.eye_shadow_left_ref = self.from_numpy_to_var(self.hist_matching(self.cdf(eye_shadow_left_src, cor_src * (mask_src != 5)), self.cdf(eye_shadow_left_ref, cor_ref * (mask_ref != 5)), eye_shadow_left_src))
            self.eye_shadow_left_src = self.from_numpy_to_var(eye_shadow_left_target)
            self.eye_shadow_left_real_B = self.from_numpy_to_var(eye_shadow_left_ref)
        else:
            self.eye_shadow_left_src = self.from_numpy_to_var(image_src * 0)
            self.eye_shadow_left_ref = self.from_numpy_to_var(image_ref * 0)
            self.eye_shadow_left_real_B = self.from_numpy_to_var(image_ref * 0)
        
    def forward(self):
        self.fake_B, self.fake_A = self.netG(self.real_A, self.real_B)
        self.rec_B, self.rec_A = self.netG(self.fake_A, self.fake_B)
        
    def backward_D_basic(self, netD, real, fake):
        # Real
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        # backward
        loss_D.backward()
        return loss_D

    def backward_D_A(self):
        fake_B = self.fake_B_pool.query(self.fake_B)
        self.loss_D_A = self.backward_D_basic(self.netD_A, self.real_B, fake_B)

    def backward_D_B(self):
        fake_A = self.fake_A_pool.query(self.fake_A)
        self.loss_D_B = self.backward_D_basic(self.netD_B, self.real_A, fake_A)

    def backward_G(self):
        lambda_A = self.opt.lambda_A
        lambda_B = self.opt.lambda_B
        lambda_idt = self.opt.lambda_identity
        lambda_percept = self.opt.lambda_perceptual
        lambda_face = self.opt.lambda_face
        lambda_lips = self.opt.lambda_lips
        lambda_eye_shadow = self.opt.lambda_eye_shadow
        
        # Identity loss
        if lambda_idt > 0:
            self.idt_A, self.idt_B = self.netG(self.real_B, self.real_A)
            self.loss_idt_A = self.criterionIdt(self.idt_A, self.real_B) * lambda_B * lambda_idt
            self.loss_idt_B = self.criterionIdt(self.idt_B, self.real_A) * lambda_A * lambda_idt
        else:
            self.loss_idt_A = 0
            self.loss_idt_B = 0

        # vgg16 perceptual loss
        if lambda_percept > 0:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            with torch.no_grad():
                self.loss_network = LossNetwork()
                self.loss_network.to(self.device)
            self.loss_network.eval()

            with torch.no_grad():
                self.vgg_feature_real_A = self.loss_network(self.real_A)[0].detach()
                self.vgg_feature_real_B = self.loss_network(self.real_B)[0].detach()
                self.vgg_feature_fake_A = self.loss_network(self.fake_A)[0].detach()
                self.vgg_feature_fake_B = self.loss_network(self.fake_B)[0].detach()

            # perceptual loss (G_A(A), A)
            self.loss_percept_A = self.criterionPer(self.vgg_feature_real_A, self.vgg_feature_fake_B) * lambda_percept
            # perceptual loss (G_B(B), B)
            self.loss_percept_B = self.criterionPer(self.vgg_feature_real_B, self.vgg_feature_fake_A) * lambda_percept
        else:
            self.loss_percept_A = 0
            self.loss_percept_B = 0
        
        # makeup loss
        if self.opt.netG == 'beautyGAN' and self.opt.use_makeup_loss == True and self.opt.phase == 'train':
            self.get_histogram_matching(self.real_A, self.real_B, self.fake_B, self.nonmakeup, self.makeup)
            self.loss_face = self.criterionMakeup(self.face_src, self.face_ref) * lambda_face
            self.loss_lips = self.criterionMakeup(self.lips_src, self.lips_ref) * lambda_lips
            self.loss_eye_shadow = 0.5 * (self.criterionMakeup(self.eye_shadow_left_src, self.eye_shadow_left_ref) * lambda_eye_shadow + 
                                          self.criterionMakeup(self.eye_shadow_right_src, self.eye_shadow_right_ref) * lambda_eye_shadow)
            self.loss_makeup = (self.loss_face + self.loss_lips + self.loss_eye_shadow).type(torch.FloatTensor).cuda() 
        else:
            self.loss_makeup = 0
        
        
        # GAN loss D_A(G_A(A))
        self.loss_G_A = self.criterionGAN(self.netD_A(self.fake_B), True)
        # GAN loss D_B(G_B(B))
        self.loss_G_B = self.criterionGAN(self.netD_B(self.fake_A), True)
        # Forward cycle loss
        self.loss_cycle_A = self.criterionCycle(self.rec_A, self.real_A) * lambda_A
        # Backward cycle loss
        self.loss_cycle_B = self.criterionCycle(self.rec_B, self.real_B) * lambda_B
        # combined loss
        if self.opt.phase == 'train':
            self.loss_G = self.loss_G_A + self.loss_G_B + self.loss_cycle_A + self.loss_cycle_B + self.loss_idt_A + self.loss_idt_B + self.loss_percept_A + self.loss_percept_B + self.loss_makeup
        else:
            self.loss_G = self.loss_G_A + self.loss_G_B + self.loss_cycle_A + self.loss_cycle_B + self.loss_idt_A + self.loss_idt_B + self.loss_percept_A + self.loss_percept_B
 
        self.loss_G.backward()

    def optimize_parameters(self):
        # forward
        self.forward()
        # G_A and G_B
        self.set_requires_grad([self.netD_A, self.netD_B], False)
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()
        # D_A and D_B
        self.set_requires_grad([self.netD_A, self.netD_B], True)
        self.optimizer_D.zero_grad()
        self.backward_D_A()
        self.backward_D_B()
        self.optimizer_D.step()
