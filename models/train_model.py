import importlib
import random
import torch
from collections import OrderedDict
from copy import deepcopy
from os import path as osp
from tqdm import tqdm
import time
import numpy as np
from pyiqa import create_metric

from models.archs import define_network
from models.base_model import BaseModel
from models.losses import compute_gradient_penalty
from utils import get_root_logger, imwrite, tensor2img

loss_module = importlib.import_module('models.losses')
metric_module = importlib.import_module('metrics')

class TrainModel(BaseModel):

    def __init__(self, opt):
        super().__init__(opt)

        # define network
        self.net_g = define_network(deepcopy(opt['network_g']))
        self.net_g = self.model_to_device(self.net_g)
        self.print_network(self.net_g)
        
        if self.opt.get('network_d'):
            self.net_d = define_network(deepcopy(self.opt['network_d']))
            self.net_d = self.model_to_device(self.net_d)
            self.print_network(self.net_d)
        
        # load pretrained models
        load_path = self.opt['path'].get('pretrain_network_g', None)
        if load_path is not None:
            self.load_network(self.net_g, load_path,
                              self.opt['path'].get('strict_load_g', False))
        if self.opt.get('network_d'):
            load_path = self.opt['path'].get('pretrain_network_d', None)
            if load_path is not None:
                self.load_network(self.net_d, load_path,
                                self.opt['path'].get('strict_load_d', False))

        if self.is_train:
            self.init_training_settings()
            
        self.best_metric_results = None

    def init_training_settings(self):
        self.net_g.train()
        if self.opt.get('network_d'):
            self.net_d.train()
        train_opt = self.opt['train']

        # define losses
        # base pix loss
        if train_opt.get('base_opt'):
            base_type = train_opt['base_opt'].pop('type')
            cri_base_cls = getattr(loss_module, base_type)
            self.cri_base = cri_base_cls(**train_opt['base_opt']).to(
                self.device)
        if train_opt.get('hrp_opt'):
            hrp_type = train_opt['hrp_opt'].pop('type')
            cri_hrp_cls = getattr(loss_module, hrp_type)
            self.cri_hrp = cri_hrp_cls(**train_opt['hrp_opt']).to(
                self.device)
        if train_opt.get('perceptual_opt'):
            perceptual_type = train_opt['perceptual_opt'].pop('type')
            cri_perceptual_cls = getattr(loss_module, perceptual_type)
            self.cri_perceptual = cri_perceptual_cls(**train_opt['perceptual_opt']).to(
                self.device)
        if train_opt.get('ssim_opt'):
            ssim_type = train_opt['ssim_opt'].pop('type')
            cri_ssim_cls = getattr(loss_module, ssim_type)
            self.cri_ssim = cri_ssim_cls(**train_opt['ssim_opt']).to(
                self.device)
        # color loss
        if train_opt.get('color_opt'):
            color_type = train_opt['color_opt'].pop('type')
            cri_color_cls = getattr(loss_module, color_type)
            self.cri_color = cri_color_cls(**train_opt['color_opt']).to(
                self.device)
        # base color loss
        if train_opt.get('gan_opt'):
            gan_type = train_opt['gan_opt'].pop('type')
            cri_gan_cls = getattr(loss_module, gan_type)
            self.cri_gan = cri_gan_cls(**train_opt['gan_opt']).to(self.device)
            self.net_d_iters = train_opt.get('net_d_iters', 1)
            self.net_d_init_iters = train_opt.get('net_d_init_iters', 0)        
        if train_opt.get('gp_opt'):
            self.gp_weight = train_opt['gp_opt'].pop('loss_weight')

        # iqa losses
        if train_opt.get('iqa'):
            self.cri_iqa = []
            for iqa_method in train_opt['iqa']:
                self.cri_iqa.append(iqa_method)
                setattr(self, f'cri_iqa_{iqa_method}', create_metric(
                    iqa_method, as_loss=True, device=self.device, 
                    **train_opt['iqa'][iqa_method]))

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

    def setup_optimizers(self):
        train_opt = self.opt['train']
        optim_params = []
        logger = get_root_logger()
        for k, v in self.net_g.named_parameters():
            # print(k)
            if v.requires_grad:
                # logger.info(f'add {k} to update list')
                optim_params.append(v)
            else:
                logger.warning(f'Params {k} will not be optimized.')

        # optimizer g
        optim_type = train_opt['optim_g'].pop('type')
        if optim_type == 'Adam':
            self.optimizer_g = torch.optim.Adam(optim_params,
                                                **train_opt['optim_g'])
        else:
            raise NotImplementedError(
                f'optimizer {optim_type} is not supperted yet.')
        self.optimizers.append(self.optimizer_g)

        # optimizer d
        if train_opt.get('optim_d'):
            optim_type = train_opt['optim_d'].pop('type')
            if optim_type == 'Adam':
                self.optimizer_d = torch.optim.Adam(self.net_d.parameters(),
                                                    **train_opt['optim_d'])
            else:
                raise NotImplementedError(
                    f'optimizer {optim_type} is not supperted yet.')
            self.optimizers.append(self.optimizer_d)

    def feed_data(self, data):
        self.lq = data['lq'].to(self.device)
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)
        

    def optimize_parameters(self, current_iter):
        # optimize net_g
        if self.opt.get('network_d'):
            for p in self.net_d.parameters():
                p.requires_grad = False

        self.optimizer_g.zero_grad()
        self.output = self.net_g(self.lq)
        
        l_g_total = 0
        loss_dict = OrderedDict()

        # pixel loss
        if hasattr(self, 'cri_base'):
            l_g_base = self.cri_base(self.output, self.gt)
            # l_g_base.requires_grad_(True)
            l_g_total += l_g_base
            loss_dict['l_g_base'] = l_g_base
        if hasattr(self, 'cri_hrp'):
            l_g_hrp = self.cri_hrp(self.output, self.base, self.gt)
            l_g_total += l_g_hrp
            loss_dict['l_g_hrp'] = l_g_hrp
        if hasattr(self, 'cri_perceptual'):
            l_g_perceptual = self.cri_perceptual(self.output, self.gt)
            l_g_total += l_g_perceptual
            loss_dict['l_g_perceptual'] = l_g_perceptual
        if hasattr(self, 'cri_ssim'):
            l_g_ssim = self.cri_ssim(self.output, self.gt)
            l_g_total += l_g_ssim
            loss_dict['l_g_ssim'] = l_g_ssim    
        if hasattr(self, 'cri_color'):
            l_g_color = self.cri_color(self.output, self.gt)
            l_g_total += l_g_color
            loss_dict['l_g_color'] = l_g_color

        if hasattr(self, 'cri_gan'):
            if (current_iter % self.net_d_iters == 0 and current_iter > self.net_d_init_iters):
            # gan loss
                fake_g_pred = self.net_d(self.output)
                l_g_gan = self.cri_gan(fake_g_pred, True, is_disc=False)
                l_g_total += l_g_gan
                loss_dict['l_g_gan'] = l_g_gan

        if hasattr(self, 'cri_iqa'):
            for iqa_method in self.cri_iqa:
                loss = getattr(self, f'cri_iqa_{iqa_method}')(self.output, self.gt)
                l_g_total += loss
                loss_dict[f'l_g_iqa_{iqa_method}'] = loss

        l_g_total.backward()
        self.optimizer_g.step()

        if hasattr(self, 'cri_gan'):
            # optimize net_d
            for p in self.net_d.parameters():
                p.requires_grad = True

            self.optimizer_d.zero_grad()
            self.output = self.net_g(self.lq)
            # real
            real_d_pred = self.net_d(self.ref)
            l_d_real = self.cri_gan(real_d_pred, True, is_disc=True)
            loss_dict['l_d_real'] = l_d_real
            loss_dict['out_d_real'] = torch.mean(real_d_pred.detach())
            # fake
            fake_d_pred = self.net_d(self.output)
            l_d_fake = self.cri_gan(fake_d_pred, False, is_disc=True)
            loss_dict['l_d_fake'] = l_d_fake
            loss_dict['out_d_fake'] = torch.mean(fake_d_pred.detach())
            gradient_penalty = compute_gradient_penalty(self.net_d, self.ref, self.output)
            l_d = l_d_real + l_d_fake + self.gp_weight * gradient_penalty

            l_d.backward()
            self.optimizer_d.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)
        # print("Loss Dictionary Keys and Values:")
        # for key, value in loss_dict.items():
        #     print(f"{key}: {value}")

    def test(self):
        self.net_g.eval()
        with torch.no_grad():
            self.output = self.net_g(self.lq)
        self.net_g.train()

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
            if self.best_metric_results is None:
                self.best_metric_results = {}
                for metric in self.opt['val']['metrics'].keys():
                    self.best_metric_results[metric] = dict(value=0, iter=-1)
            
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
                if hasattr(self, item):
                    delattr(self, item)
            torch.cuda.empty_cache()

            if with_metrics:
                opt_metric = deepcopy(self.opt['val']['metrics'])
                for name, opt_ in opt_metric.items():
                    metric_type = opt_.pop('type')
                    result = getattr(
                        metric_module, metric_type)(
                            visual_imgs['output'], visual_imgs['gt'], **opt_)
                    logger.info(f'{name}_{img_name}: {result}')
                    self.metric_results[name] += result

            if save_img:
                if self.opt['is_train']:
                    save_img_path = osp.join(self.opt['path']['visualization'],
                                             str(current_iter),
                                             f'{img_name}_{current_iter}.png')
                else:
                    if self.opt['val']['suffix']:
                        save_img_path = osp.join(
                            self.opt['path']['visualization'], dataset_name,
                            f'{img_name}_{self.opt["val"]["suffix"]}.png')
                    else:
                        save_img_path = osp.join(
                            self.opt['path']['visualization'], dataset_name,
                            f'{img_name}_{self.opt["name"]}.png')
                for item in visual_imgs:
                    imwrite(visual_imgs[item], save_img_path.replace('.png', f'_{item}.png'))

            pbar.update(1)
            pbar.set_description(f'Test {img_name}')
        pbar.close()

        if with_metrics:
            for metric in self.metric_results.keys():
                self.metric_results[metric] /= (idx + 1)

            self._log_validation_metric_values(current_iter, dataset_name,
                                               tb_logger)

    def _log_validation_metric_values(self, current_iter, dataset_name,
                                      tb_logger):
        log_str = f'Validation {dataset_name}\n'
        for metric, value in self.metric_results.items():
            if value > self.best_metric_results[metric]['value']:
                self.best_metric_results[metric]['value'] = value
                self.best_metric_results[metric]['iter'] = current_iter
            
            best_val = self.best_metric_results[metric]['value']
            best_iter = self.best_metric_results[metric]['iter']

            log_str += (f'\t # {metric}: {value:.4f}. '
                        f'Best: {best_val:.4f} @ iter {best_iter}\n')

        logger = get_root_logger()
        logger.info(log_str)
        if tb_logger:
            for metric, value in self.metric_results.items():
                tb_logger.add_scalar(f'metrics/{dataset_name}/{metric}', value, current_iter)
                tb_logger.add_scalar(f'metrics/{dataset_name}/best_{metric}', self.best_metric_results[metric]['value'], current_iter)

    def get_current_visuals(self):
        out_dict = OrderedDict()
        for item in self.opt['val']['visuals']:
            out_dict[item] = getattr(self, item).detach().cpu()
        return out_dict

    def save(self, epoch, current_iter):
        self.save_network(self.net_g, 'net_g', current_iter)
        if self.opt.get('network_d'):
            self.save_network(self.net_d, 'net_d', current_iter)
        self.save_training_state(epoch, current_iter)
