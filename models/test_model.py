import importlib
import torch
import cv2
from collections import OrderedDict
from copy import deepcopy
from os import path as osp
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
import numpy as np
from pyiqa import create_metric
from models.archs import define_network
from models.base_model import BaseModel
from utils import get_root_logger, imwrite, tensor2img
import os
loss_module = importlib.import_module('models.losses')
metric_module = importlib.import_module('metrics')
import torch.nn.functional as F

class TestModel(BaseModel):

    def __init__(self, opt):
        super().__init__(opt)

        # define network
        self.net_g = define_network(deepcopy(opt['network_g']))
        self.net_g = self.model_to_device(self.net_g)
        self.print_network(self.net_g)

        # load pretrained models
        load_path = self.opt['path'].get('pretrain_network_g', None)
        if load_path is not None:
            self.load_network(self.net_g, load_path,
                              self.opt['path'].get('strict_load_g', True))
        
        self.metrics = {}
        if 'metrics' in opt['val']:
            for name, metric_opt in opt['val']['metrics'].items():
                self.metrics[name] = create_metric(metric_opt['type'], device=self.device)

    def feed_data(self, data):
        self.lq = data['lq'].to(self.device)
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)

    def test(self):
        self.net_g.eval()
        with torch.no_grad():
            print('*****************input lq: ', self.lq.shape)
            self.output = self.net_g(self.lq)

    def test_speed(self, times_per_img=50, size=None):
        if size is not None:
            lq_img = self.lq.resize_(1, 3, size[0], size[1]).clone().to(self.lq.device)
        else:
            lq_img = self.lq
        
        self.net_g.eval()
        with torch.no_grad():
            torch.cuda.synchronize()
            start = time.time()
            for _ in range(times_per_img):
                self.output = self.net_g(lq_img)
                
                
            torch.cuda.synchronize()
            self.duration = (time.time() - start)
            
      
    def dist_validation(self, dataloader, current_iter, tb_logger, save_img):
        logger = get_root_logger()
        logger.info('Only support single GPU validation.')
        self.nondist_validation(dataloader, current_iter, tb_logger, save_img)

    def nondist_validation(self, dataloader, current_iter, tb_logger,
                           save_img):
        logger = get_root_logger()
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        if with_metrics:
            self.metric_results = {
                metric: 0
                for metric in self.opt['val']['metrics'].keys()
            }
        pbar = tqdm(total=len(dataloader), unit='image')

        for idx, val_data in enumerate(dataloader):
            img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]
            self.feed_data(val_data)
            self.test()

            visuals = self.get_current_visuals()
            visual_imgs = {}
            for item in visuals:
                visual_imgs[item] = tensor2img(visuals[item])
                # tentative for out of GPU memory
                if hasattr(self, item):
                    delattr(self, item)
            torch.cuda.empty_cache()

            # if with_metrics:
            #     # calculate metrics
            #     opt_metric = deepcopy(self.opt['val']['metrics'])
            #     for name, opt_ in opt_metric.items():
            #         metric_type = opt_.pop('type')
            #         result = getattr(
            #             metric_module, metric_type)(
            #                 visual_imgs['output'], visual_imgs['gt'], **opt_)
            #         logger.info(f'{name}_{img_name}: {result}')
            #         self.metric_results[name] += result
            if with_metrics:
                for name, metric in self.metrics.items():
                    output_tensor = torch.from_numpy(visual_imgs['output']).permute(2, 0, 1).unsqueeze(0).to(self.device).float() / 255.0
                    gt_tensor = torch.from_numpy(visual_imgs['gt']).permute(2, 0, 1).unsqueeze(0).to(self.device).float() / 255.0
                
                    # pyiqa 
                    result = metric(output_tensor, gt_tensor)
                    logger.info(f'{name}_{img_name}: {result.item()}')
                    self.metric_results[name] += result.item()


            if save_img:
                if self.opt['is_train']:
                    save_img_path = osp.join(self.opt['path']['visualization'],
                                             str(current_iter),
                                             f'{img_name}_{current_iter}.jpg')
                else:
                    if self.opt['val']['suffix']:
                        save_img_path = osp.join(
                            self.opt['path']['visualization'], dataset_name,
                            f'{img_name}_{self.opt["val"]["suffix"]}.jpg')
                    else:
                        save_img_path = osp.join(
                            self.opt['path']['visualization'], dataset_name,
                            f'{img_name}_{self.opt["name"]}.jpg')
                for item in visual_imgs:
                    imwrite(visual_imgs[item], save_img_path.replace('.jpg', f'_{item}.jpg'))

            pbar.update(1)
            pbar.set_description(f'Test {img_name}')
        pbar.close()

        if with_metrics:
            for metric in self.metric_results.keys():
                self.metric_results[metric] /= (idx + 1)

            self._log_validation_metric_values(current_iter, dataset_name,
                                               tb_logger)

    def nondist_validation_speed(self, dataloader, times_per_img, num_imgs, size):

        avg_duration = 0
        for idx, val_data in enumerate(dataloader):
            if idx > num_imgs:
                break
            img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]
            self.feed_data(val_data)
            self.test_speed(times_per_img, size=size)
            avg_duration += self.duration / num_imgs
            
            print(f'{idx} Testing {img_name} (shape: {self.lq.shape[2]} * {self.lq.shape[3]}) duration: {self.duration}')
            print(f'average duration is {avg_duration} seconds')
        

    def _log_validation_metric_values(self, current_iter, dataset_name,
                                      tb_logger):
        log_str = f'Validation {dataset_name}\n'
        for metric, value in self.metric_results.items():
            log_str += f'\t # {metric}: {value:.4f}\n'
        logger = get_root_logger()
        logger.info(log_str)
        if tb_logger:
            for metric, value in self.metric_results.items():
                tb_logger.add_scalar(f'metrics/{metric}', value, current_iter)

    def get_current_visuals(self):
        out_dict = OrderedDict()
        for item in self.opt['val']['visuals']:
            out_dict[item] = getattr(self, item).detach().cpu()
        return out_dict
