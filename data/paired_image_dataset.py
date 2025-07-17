import random
import numpy as np
import torch
import cv2
import os
from torch.utils import data as data
from torchvision.transforms.functional import normalize
from PIL import Image
import torch.nn.functional as F
from data.data_util import (paired_paths_from_folder,
                            paired_paths_from_folder_ppr10k,
                            paired_paths_from_folder_ppr10k_val,
                            paired_paths_from_lmdb,
                            paired_paths_from_txt)
from data.transforms import augment, paired_random_crop,paired_random_crop2
from utils import FileClient, imfrombytes, img2tensor
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from .torchvision_x_functional import resized_crop, hflip, to_tensor, crop, vflip, resize


class PairedImageDataset(data.Dataset):
    
    def __init__(self, opt):
        super(PairedImageDataset, self).__init__()
        self.opt = opt
        # file client (io backend)
     
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None
        if 'filename_tmpl' in opt:
            self.filename_tmpl = opt['filename_tmpl']
        else:
            self.filename_tmpl = '{}'

        if self.io_backend_opt['type'] == 'lmdb':
            self.gt_folder, self.lq_folder = opt['dataroot_gt'], opt['dataroot_lq']
            self.io_backend_opt['db_paths'] = [self.lq_folder, self.gt_folder]
            self.io_backend_opt['client_keys'] = ['lq', 'gt']
            self.paths = paired_paths_from_lmdb(
                [self.lq_folder, self.gt_folder], ['lq', 'gt'])
        elif self.io_backend_opt['type'] == 'disk':
            self.gt_folder, self.lq_folder = opt['dataroot_gt'], opt['dataroot_lq']
            self.paths = paired_paths_from_folder(
                [self.lq_folder, self.gt_folder], ['lq', 'gt'],
                self.filename_tmpl)
        elif self.io_backend_opt['type'] == 'txt':
            txts = [opt['datatxt_lq'], opt['datatxt_gt']]
            names = ['lq', 'gt']
            if opt.get('datatxt_mask'):
                txts.append(opt['datatxt_mask'])
                names.append('mask')
                self.mask = True
            else:
                self.mask = False
            self.paths = paired_paths_from_txt(txts, names, self.filename_tmpl)

        else:
            raise ValueError(
                f'io_backend not supported')

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(
                self.io_backend_opt.pop('type'), **self.io_backend_opt)

        # Load gt and lq images. Dimension order: HWC; channel order: BGR;
        # image range: [0, 1], float32.
        lq_path = self.paths[index]['lq_path']
        img_bytes = self.file_client.get(lq_path, 'lq')
        img_lq = imfrombytes(img_bytes, float32=True)  
        
        if img_lq is None or img_lq.size == 0:
            raise ValueError(f"Failed to load image: {lq_path}")
        
        gt_path = self.paths[index]['gt_path']
        img_bytes = self.file_client.get(gt_path, 'gt')
        img_gt = imfrombytes(img_bytes, float32=True)
        
        # augmentation for training
        if self.opt['phase'] == 'train':
            if self.opt.get('gt_size'):
                if_fix = self.opt['if_fix_size']
                gt_size = self.opt['gt_size']
                ratio = np.random.uniform(0.6,1.0)
                gt_size = round(gt_size * ratio)
                # H,W = img_lq.shape[:2]
                # crop_h = round(H*ratio)
                # crop_w = round(W*ratio)
                num_gpu = self.opt.get('num_gpu', 1) 
                if not if_fix and self.opt['batch_size_per_gpu'] * num_gpu != 1:
                    raise ValueError(
                        f'Param mismatch. Only support fix data shape if batchsize > 1 or num_gpu > 1.')
                # random crop
                img_gt, img_lq = paired_random_crop(img_gt, img_lq, gt_size, gt_size, 1, gt_path)
                

            img_gt, img_lq = augment([img_gt, img_lq], self.opt['use_flip'], self.opt['use_rot'])
            

        # BGR to RGB, HWC to CHW, numpy to tensor
        
        img_gt = img2tensor(img_gt, bgr2rgb=True, float32=True)
        img_lq = img2tensor(img_lq, bgr2rgb=True, float32=True)
    
        # img_lq = self.color_augment(img_lq)
                
        # normalize
        if self.mean is not None or self.std is not None:
            normalize(img_lq, self.mean, self.std, inplace=True)
            normalize(img_gt, self.mean, self.std, inplace=True)
        
        data_dict = {
            'lq': img_lq,
            'gt': img_gt,
            'lq_path': lq_path,
            'gt_path': gt_path
        }
        
        return data_dict
    
        
    def __len__(self):
        return len(self.paths)
    
class PairedImageDataset_ToneMapping(data.Dataset):
    
    def __init__(self, opt):
        super(PairedImageDataset_ToneMapping, self).__init__()
        self.opt = opt
        # file client (io backend)
     
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None
        if 'filename_tmpl' in opt:
            self.filename_tmpl = opt['filename_tmpl']
        else:
            self.filename_tmpl = '{}'

        if self.io_backend_opt['type'] == 'lmdb':
            self.gt_folder, self.lq_folder = opt['dataroot_gt'], opt['dataroot_lq']
            self.io_backend_opt['db_paths'] = [self.lq_folder, self.gt_folder]
            self.io_backend_opt['client_keys'] = ['lq', 'gt']
            self.paths = paired_paths_from_lmdb(
                [self.lq_folder, self.gt_folder], ['lq', 'gt'])
        elif self.io_backend_opt['type'] == 'disk':
            self.gt_folder, self.lq_folder = opt['dataroot_gt'], opt['dataroot_lq']
            self.paths = paired_paths_from_folder(
                [self.lq_folder, self.gt_folder], ['lq', 'gt'],
                self.filename_tmpl)
        elif self.io_backend_opt['type'] == 'txt':
            txts = [opt['datatxt_lq'], opt['datatxt_gt']]
            names = ['lq', 'gt']
            if opt.get('datatxt_mask'):
                txts.append(opt['datatxt_mask'])
                names.append('mask')
                self.mask = True
            else:
                self.mask = False
            self.paths = paired_paths_from_txt(txts, names, self.filename_tmpl)

        else:
            raise ValueError(
                f'io_backend not supported')

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(
                self.io_backend_opt.pop('type'), **self.io_backend_opt)

        # Load gt and lq images. Dimension order: HWC; channel order: BGR;
        # image range: [0, 1], float32.
        lq_path = self.paths[index]['lq_path']
        img_bytes = self.file_client.get(lq_path, 'lq')
        img_lq = imfrombytes(img_bytes, flag='unchanged', float32=True)  #  ,flag='unchanged'
        if img_lq is None or img_lq.size == 0:
            raise ValueError(f"Failed to load image: {lq_path}")
        
        gt_path = self.paths[index]['gt_path']
        img_bytes = self.file_client.get(gt_path, 'gt')
        img_gt = imfrombytes(img_bytes, float32=True)

        # augmentation for training
        if self.opt['phase'] == 'train':
            if self.opt.get('gt_size'):
                if_fix = self.opt['if_fix_size']
                gt_size = self.opt['gt_size']
                ratio = np.random.uniform(0.6,1.0)
                # H,W = img_lq.shape[:2]
                gt_size = round(gt_size * ratio)
                # crop_h = round(H*ratio)
                # crop_w = round(W*ratio)
                num_gpu = self.opt.get('num_gpu', 1) 
                if not if_fix and self.opt['batch_size_per_gpu'] * num_gpu != 1:
                    raise ValueError(
                        f'Param mismatch. Only support fix data shape if batchsize > 1 or num_gpu > 1.')
                # random crop
                img_gt, img_lq = paired_random_crop(img_gt, img_lq, gt_size, gt_size, 1, gt_path)
                

            img_gt, img_lq = augment([img_gt, img_lq], self.opt['use_flip'], self.opt['use_rot'])
            

        # BGR to RGB, HWC to CHW, numpy to tensor
        img_gt = img2tensor(img_gt, bgr2rgb=True, float32=True)
        img_lq = img2tensor(img_lq, bgr2rgb=True, float32=True)

        # img_lq = self.color_augment(img_lq)
                
        # normalize
        if self.mean is not None or self.std is not None:
            normalize(img_lq, self.mean, self.std, inplace=True)
            normalize(img_gt, self.mean, self.std, inplace=True)
        
        data_dict = {
            'lq': img_lq,
            'gt': img_gt,
            'lq_path': lq_path,
            'gt_path': gt_path
        }
        
        return data_dict
    
        
    def __len__(self):
        return len(self.paths)
    
class PPRDataset(data.Dataset):
    
    def __init__(self, opt):
        super(PPRDataset, self).__init__()
        self.opt = opt
        # file client (io backend)
     
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None
        if 'filename_tmpl' in opt:
            self.filename_tmpl = opt['filename_tmpl']
        else:
            self.filename_tmpl = '{}'

        self.gt_folder, self.lq_folder = opt['dataroot_gt'], opt['dataroot_lq']
        self.mask_folder = opt['dataroot_mask']
        self.paths = paired_paths_from_folder_ppr10k(
            [self.lq_folder, self.gt_folder, self.mask_folder], ['lq', 'gt', 'mask'],
            self.filename_tmpl)
        

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(
                self.io_backend_opt.pop('type'), **self.io_backend_opt)

        # Load gt and lq images. Dimension order: HWC; channel order: BGR;
        # image range: [0, 1], float32.

        lq_path = self.paths[index]['lq_path']
        img_bytes = self.file_client.get(lq_path, 'lq')
        img_lq = imfrombytes(img_bytes, flag='unchanged', float32=True)  #  ,flag='unchanged'
        if img_lq is None or img_lq.size == 0:
            raise ValueError(f"Failed to load image: {lq_path}")
        
        gt_path = self.paths[index]['gt_path']
        img_bytes = self.file_client.get(gt_path, 'gt')
        img_gt = imfrombytes(img_bytes, float32=True)
        H,W = img_gt.shape[:2]
        
        mask_path = self.paths[index]['mask_path']
        img_bytes = self.file_client.get(mask_path, 'mask')
        img_mask = imfrombytes(img_bytes, flag='grayscale', resize=True, float32=True, img_size=(W,H))
        
        imgs = [img_gt, img_lq, img_mask]
        
        # augmentation for training
        if self.opt['phase'] == 'train':
            if self.opt.get('gt_size'):
                if_fix = self.opt['if_fix_size']
                gt_size = self.opt['gt_size']
                ratio = np.random.uniform(0.6,1.0)
                gt_size = round(min(H,W) * ratio)
                num_gpu = self.opt.get('num_gpu', 1) 
                if not if_fix and self.opt['batch_size_per_gpu'] * num_gpu != 1:
                    raise ValueError(
                        f'Param mismatch. Only support fix data shape if batchsize > 1 or num_gpu > 1.')
                # random crop
                imgs = paired_random_crop2(imgs, if_fix, gt_size, gt_path)

            imgs = augment(imgs, self.opt['use_flip'], self.opt['use_rot'])
            

        # BGR to RGB, HWC to CHW, numpy to tensor
        imgs = img2tensor(imgs, bgr2rgb=True, float32=True)
        img_gt, img_lq, img_mask = imgs
        img_lq = torch.cat([img_lq, img_mask], dim = 0)
                
        # normalize
        if self.mean is not None or self.std is not None:
            normalize(img_lq, self.mean, self.std, inplace=True)
            normalize(img_gt, self.mean, self.std, inplace=True)
        
        data_dict = {
            'lq': img_lq,
            'gt': img_gt,
            'lq_path': lq_path,
            'gt_path': gt_path
        }
        
        return data_dict
    
        
    def __len__(self):
        return len(self.paths)


class PPRDataset_Val(data.Dataset):
    
    def __init__(self, opt):
        super(PPRDataset_Val, self).__init__()
        self.opt = opt
        # file client (io backend)
     
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None
        if 'filename_tmpl' in opt:
            self.filename_tmpl = opt['filename_tmpl']
        else:
            self.filename_tmpl = '{}'
        if self.io_backend_opt['type'] == 'lmdb':
            self.gt_folder, self.lq_folder = opt['dataroot_gt'], opt['dataroot_lq']
            self.io_backend_opt['db_paths'] = [self.lq_folder, self.gt_folder]
            self.io_backend_opt['client_keys'] = ['lq', 'gt']
            self.paths = paired_paths_from_lmdb(
                [self.lq_folder, self.gt_folder], ['lq', 'gt'])
        elif self.io_backend_opt['type'] == 'disk':
            self.gt_folder, self.lq_folder = opt['dataroot_gt'], opt['dataroot_lq']
            self.mask_folder = opt['dataroot_mask']
            # self.paths = paired_paths_from_folder(
            #     [self.lq_folder, self.gt_folder], ['lq', 'gt'],
            #     self.filename_tmpl)
            self.paths = paired_paths_from_folder_ppr10k_val(
            [self.lq_folder, self.gt_folder,self.mask_folder], ['lq', 'gt', 'mask'],
            self.filename_tmpl)

        else:
            raise ValueError(
                f'io_backend not supported')
        # self.train_input_files = sorted(self.paths['lq_path'])
        # self.train_target_files = sorted(self.paths['gt_path'])


    def __getitem__(self, index):
        
        lq_path = self.paths[index]['lq_path']
        gt_path = self.paths[index]['gt_path']
        mask_path = self.paths[index]['mask_path']
        
        img_input = cv2.imread(lq_path,-1)
        img_exptC = Image.open(gt_path)
        img_mask = Image.open(mask_path)

        img_input = np.array(img_input)

        img_input = img_input[:, :, [2, 1, 0]]

        img_input = to_tensor(img_input)
        img_exptC = TF.to_tensor(img_exptC)
        img_mask = TF.to_tensor(img_mask)
        
        img_mask = img_mask.unsqueeze(0)
        img_input = img_input.unsqueeze(0)
        img_mask = F.interpolate(img_mask, size=img_input.shape[2:], mode='bilinear', align_corners=False)
        img_input = torch.cat([img_input, img_mask], dim = 1)
        img_input = img_input.squeeze(0)
        data_dict = {
            'lq': img_input,
            'gt': img_exptC,
            'lq_path': lq_path,
            'gt_path': gt_path,
        }
        
        return data_dict
    
        
    def __len__(self):
        return len(self.paths)
        