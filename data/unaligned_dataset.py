import os.path
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import random


class UnalignedDataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A')
        self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B')
        self.A_paths = make_dataset(self.dir_A)
        self.B_paths = make_dataset(self.dir_B)
        self.A_paths = sorted(self.A_paths)
        self.B_paths = sorted(self.B_paths)
        self.A_size = len(self.A_paths)
        self.B_size = len(self.B_paths)
        
        if opt.phase == 'train':
            self.dir_makeup = os.path.join(opt.maskroot, 'makeup')
            self.dir_nonmakeup = os.path.join(opt.maskroot, 'non-makeup')
            self.makeup_paths = make_dataset(self.dir_makeup)
            self.nonmakeup_paths = make_dataset(self.dir_nonmakeup)
            self.makeup_paths = sorted(self.makeup_paths)
            self.nonmakeup_paths = sorted(self.nonmakeup_paths)
            self.makeup_size = len(self.makeup_paths)
            self.nonmakeup_size = len(self.nonmakeup_paths)
            
        self.transform, self.transform_flip = get_transform(opt)

    def __getitem__(self, index):
        A_path = self.A_paths[index % self.A_size]
        nonmakeup_path = self.nonmakeup_paths[index % self.nonmakeup_size] if self.opt.phase == 'train' else None
        if self.opt.serial_batches:
            index_B = index % self.B_size
            index_makeup = index % self.makeup_size if self.opt.phase == 'train' else None
        else:
            index_B = random.randint(0, self.B_size - 1)
        B_path = self.B_paths[index_B]
        makeup_path = self.makeup_paths[index_B] if self.opt.phase == 'train' else None
        A_img = Image.open(A_path).convert('RGB')
        B_img = Image.open(B_path).convert('RGB')
        makeup_img = Image.open(makeup_path) if self.opt.phase == 'train' else None
        nonmakeup_img = Image.open(nonmakeup_path) if self.opt.phase == 'train' else None
        
        if random.random() > 0.5:
            A = self.transform_flip(A_img)
            nonmakeup_mask = self.transform_flip(nonmakeup_img) if self.opt.phase == 'train' else None
        else:
            A = self.transform(A_img)
            nonmakeup_mask = self.transform(nonmakeup_img) if self.opt.phase == 'train' else None
        
        if random.random() > 0.5:
            B = self.transform_flip(B_img)
            makeup_mask = self.transform_flip(makeup_img) if self.opt.phase == 'train' else None
        else:
            B = self.transform(B_img)
            makeup_mask = self.transform(makeup_img) if self.opt.phase == 'train' else None
        
        if self.opt.direction == 'BtoA':
            input_nc = self.opt.output_nc
            output_nc = self.opt.input_nc
        else:
            input_nc = self.opt.input_nc
            output_nc = self.opt.output_nc

        if input_nc == 1:  # RGB to gray
            tmp = A[0, ...] * 0.299 + A[1, ...] * 0.587 + A[2, ...] * 0.114
            A = tmp.unsqueeze(0)

        if output_nc == 1:  # RGB to gray
            tmp = B[0, ...] * 0.299 + B[1, ...] * 0.587 + B[2, ...] * 0.114
            B = tmp.unsqueeze(0)
        
        if self.opt.phase == 'train':
            return {'A': A, 'B': B, 'makeup_mask': makeup_mask, 'nonmakeup_mask': nonmakeup_mask,
                    'A_paths': A_path, 'B_paths': B_path, 'makeup_paths': makeup_path, 'nonmakeup_paths': nonmakeup_path}
        else:
            return {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        return max(self.A_size, self.B_size)

    def name(self):
        return 'UnalignedDataset'
