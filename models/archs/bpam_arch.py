import torch
import torch.nn as nn
import time
from models.archs.arch_util import ResidualBlockNoBN
from models.archs.NAFNet_arch import NAFNet
import torch.nn.functional as F

from models.archs.cpp_ext_interface import slice_function,apply_coefficent
        
class GuideNN(nn.Module):
    def __init__(self, ci=3, co=1, kernel=1, layer=2, res=False, channel=16, scale=1):
        super().__init__()
        assert scale == 1, f'Not support scale={scale} yet!'
        guide_net = [nn.Conv2d(ci, channel, kernel, stride=1, padding=kernel//2), nn.ReLU()]
        for _ in range(layer - 2):
            if res:
                guide_net += [ResidualBlockNoBN(channel)]
            else:
                guide_net += [nn.Conv2d(channel, channel, kernel, stride=1, padding=kernel//2), 
                              nn.ReLU()]
        guide_net += [nn.Conv2d(channel, co, kernel, stride=1, padding=kernel//2), nn.Tanh()]  # nn.Tanh()
        self.guide_net = nn.Sequential(*guide_net)
        
        
    def forward(self, x):
        return self.guide_net(x)

    
class Slicing_CUDA(nn.Module):
    def __init__(self):
        super().__init__()
                
    def forward(self, grid, guide):
        B, c, H, W = guide.shape
        
        pos = torch.tensor([[0,H,0,W]]).repeat(B,1).to(guide.device)
           
        pos_h, size_h, pos_w, size_w = pos[0]
        x = (torch.arange(W).to(guide) + pos_w - size_w / 2) / (size_w / 2)
        y = (torch.arange(H).to(guide) + pos_h - size_h / 2) / (size_h / 2)
        yy, xx = torch.meshgrid(y, x)  
        xx = xx.unsqueeze(0).unsqueeze(0)  
        yy = yy.unsqueeze(0).unsqueeze(0)  
            
        all_list = []
        for i in range(c):
            guide_i = guide[:, i : i+1, :, :]
            merged_3 = torch.cat([xx, yy, guide_i], dim=1)
            all_list.append(merged_3)

        guide_3c = torch.cat(all_list, dim=1)
        
        out = slice_function(grid, guide_3c)

        return out


class Slicing(nn.Module):
    def __init__(self):
        super().__init__()
                
    def forward(self, grid, guide):
        gb, c, gh, gw = guide.shape    

        pos = torch.tensor([[0,gh,0,gw]]).repeat(gb,1).to(guide.device)
        xxx, yyy = [], []     
        for b in range(gb):   
            pos_h, size_h, pos_w, size_w = pos[b]  
            x = (torch.arange(gw).to(guide) + pos_w - size_w / 2) / (size_w / 2) 
            y = (torch.arange(gh).to(guide) + pos_h - size_h / 2) / (size_h / 2)
            yy, xx = torch.meshgrid(y, x)  
            xx = xx.unsqueeze(0).unsqueeze(0)  
            yy = yy.unsqueeze(0).unsqueeze(0)  
            xxx.append(xx)  
            yyy.append(yy)
        xxx = torch.cat(xxx, dim=0)  
        yyy = torch.cat(yyy, dim=0)
        
        return nn.functional.grid_sample(grid,
                                         torch.stack((xxx, yyy, guide), dim=-1),      
                                         padding_mode='border').squeeze(2)
    
          
class apply_coeff(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, coeff_split):        
        B, c, H, W = x.shape

        weighted = coeff_split[:, :, :c, :, :] * x.unsqueeze(1)  # [B, out_channels, 3, H, W]

        out = weighted.sum(dim=2)  

        out += coeff_split[:, :, c, :, :]  

        return out    

class MLP(nn.Module):
    def __init__(self, co_1=1, co_2=1, kernel=1, layer=2):
        super().__init__()
        self.context1= GuideNN(ci=3, co=co_1, kernel=kernel, layer=layer, res=False, channel=16, scale=1)
        self.context2= GuideNN(ci=8, co=co_2, kernel=kernel, layer=layer, res=False, channel=16, scale=1)
        self.slicing = Slicing()
        self.apply_coeff = apply_coeff()
        self.activation = nn.ReLU()

    def forward(self, src, grid1, grid2):

        context_map = self.context1(src)
        
        coeff_map_r = self.slicing(grid1[:, 0:8, :, :, :], context_map[:, 0:1, :, :])
        coeff_map_g = self.slicing(grid1[:, 8:16, :, :, :], context_map[:, 1:2, :, :])
        coeff_map_b = self.slicing(grid1[:, 16:24, :, :, :], context_map[:, 2:3, :, :])
        coeff_map_c = self.slicing(grid1[:, 24:32, :, :, :], context_map[:, 3:4, :, :])
        coeff1 = torch.cat([coeff_map_r, coeff_map_g, coeff_map_b, coeff_map_c], dim = 1)
        
        coeff1_split = coeff1.view(coeff1.shape[0], 4, 8, coeff1.shape[2], coeff1.shape[3]).permute(0, 2, 1, 3, 4)

        hidden = self.apply_coeff(src, coeff1_split)
        hidden = self.activation(hidden)
        
        context_map2 = self.context2(hidden)
        
        coeff_map_1 = self.slicing(grid2[:, 0:3, :, :, :], context_map2[:, 0:1, :, :])
        coeff_map_2 = self.slicing(grid2[:, 3:6, :, :, :], context_map2[:, 1:2, :, :])
        coeff_map_3 = self.slicing(grid2[:, 6:9, :, :, :], context_map2[:, 2:3, :, :])
        coeff_map_4 = self.slicing(grid2[:, 9:12, :, :, :], context_map2[:, 3:4, :, :])
        coeff_map_5 = self.slicing(grid2[:, 12:15, :, :, :], context_map2[:, 4:5, :, :])
        coeff_map_6 = self.slicing(grid2[:, 15:18, :, :, :], context_map2[:, 5:6, :, :])
        coeff_map_7 = self.slicing(grid2[:, 18:21, :, :, :], context_map2[:, 6:7, :, :])
        coeff_map_8 = self.slicing(grid2[:, 21:24, :, :, :], context_map2[:, 7:8, :, :])
        coeff_map_9 = self.slicing(grid2[:, 24:27, :, :, :], context_map2[:, 8:9, :, :])

        coeff2 = torch.cat([coeff_map_1,coeff_map_2,coeff_map_3,coeff_map_4,coeff_map_5,coeff_map_6,coeff_map_7,coeff_map_8,coeff_map_9], dim = 1)
        
        coeff2_split = coeff2.view(coeff2.shape[0], 9, 3, coeff2.shape[2], coeff2.shape[3]).permute(0, 2, 1, 3, 4)   
        out = self.apply_coeff(hidden, coeff2_split)
        
        return out

class MLP_CUDA(nn.Module):
    def __init__(self, co_1=1, co_2=1, kernel=1, layer=2):
        super().__init__()
        self.context1 = GuideNN(ci=3, co=co_1, kernel=kernel, layer=layer, res=False, channel=16, scale=1)
        self.context2 = GuideNN(ci=8, co=co_2, kernel=kernel, layer=layer, res=False, channel=16, scale=1)
        
        self.slicing = Slicing_CUDA()
        self.activation = nn.ReLU()
        
    def forward(self, src, grid1, grid2):
        
        context_map = self.context1(src)
        
        coeff1 = self.slicing(grid1, context_map)
        
        coeff1_split = coeff1.view(coeff1.shape[0], 4, 8, coeff1.shape[2], coeff1.shape[3]).permute(0, 2, 1, 3, 4).contiguous()
        
        hidden = apply_coefficent(src, coeff1_split)
        hidden = self.activation(hidden)
        
        context_map2 = self.context2(hidden)
        
        coeff2 = self.slicing(grid2, context_map2)
        
        coeff2_split = coeff2.view(coeff2.shape[0], 9, 3, coeff2.shape[2], coeff2.shape[3]).permute(0, 2, 1, 3, 4).contiguous()
        
        out = apply_coefficent(hidden, coeff2_split)
        
        
        return out     

class BPAM(nn.Module):
    def __init__(self, ci=3,grid_range=8, scale_factor=0.5,co_1=1,co_2=1,kernel=1,layer=2,width = 16,enc_blks = [2, 2, 2, 2],dec_blks = [2, 2, 2, 2], middle_blk_num=2):
        super().__init__()
        self.relu = nn.ReLU()
        self.downsample = nn.Upsample(scale_factor = scale_factor)     # scale_factor = 0.5
        self.mlp = MLP(co_1=co_1, co_2=co_2, kernel=kernel, layer=layer)
        self.ci = ci
        self.grid_range = grid_range

        self.model = NAFNet(input_channel=self.ci, width=width, middle_blk_num=middle_blk_num,
                    enc_blk_nums=enc_blks, dec_blk_nums=dec_blks,final_dim_1=32*grid_range,final_dim_2=27*grid_range)

    def forward(self, x):
        
        x_down = self.downsample(x)
        grid1, grid2 = self.model(x_down)
        b, c, h, w = grid1.shape
        grid1 = grid1.reshape(b,c//self.grid_range,self.grid_range,h,w)
        b, c, h, w = grid2.shape
        grid2 = grid2.reshape(b,c//self.grid_range,self.grid_range,h,w)
        
        out = self.mlp(x, grid1, grid2)
        out = self.relu(out)
         
        return out

    
class BPAM_PPR(nn.Module):
    def __init__(self, ci=4, grid_range=8,scale_factor=1,co_1=4,co_2=9,kernel=3,layer=4,width=32,enc_blks = [2, 2, 2, 2],dec_blks = [2, 2, 2, 2], middle_blk_num=2):
        super().__init__()
        self.relu = nn.ReLU()
        self.downsample = nn.Upsample(scale_factor = scale_factor)     
        self.mlp = MLP(co_1=co_1, co_2=co_2, kernel=kernel, layer=layer)
        self.ci = ci
        self.grid_range = grid_range

        self.model = NAFNet(input_channel=self.ci, width=width, middle_blk_num=middle_blk_num,
                    enc_blk_nums=enc_blks, dec_blk_nums=dec_blks,final_dim_1=384,final_dim_2=324)
        
    def forward(self, x):
        
        x_down = self.downsample(x)
        grid1, grid2 = self.model(x_down)
        b, c, h, w = grid1.shape
        grid1 = grid1.reshape(b,c//self.grid_range,self.grid_range,h,w)
        b, c, h, w = grid2.shape
        grid2 = grid2.reshape(b,c//self.grid_range,self.grid_range,h,w)
        
        out = self.mlp(x[:,:3,:,:], grid1, grid2)
        out = self.relu(out)
        
        return out        