import torch

from torch.cuda.amp import custom_fwd, custom_bwd
from typing import Tuple

import bilateral_slicing
import apply_coeff

class SliceFunction(torch.autograd.Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, 
                grid: torch.Tensor, 
                x: torch.Tensor) -> torch.Tensor:
        x = x.contiguous()
        grid = grid.contiguous()

        assert x.ndimension() == 4, \
            "only support 2D image with batch and channel dimensions (4D tensor)"
        assert grid.ndimension() in [5], \
            "only support 3D lookup table with batch dimension (5D tensor)"
        
        output = x.new_zeros((x.size(0), grid.size(1), x.size(2), x.size(3)))
        output.contiguous()
               
        bilateral_slicing.tri_forward(grid, x, output)
        
        ctx.save_for_backward(grid, x)
        
        return output
    
    # @staticmethod
    # @custom_bwd
    # def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor]:
        
    #     grad_output = grad_output.contiguous()
        
    #     grid, x = ctx.saved_tensors
                
    #     grad_img = torch.zeros_like(x)
    #     grad_grid = torch.zeros_like(grid)          
        
    #     bilateral_slicing.tri_backward(grad_output, grid, x, grad_grid, grad_img)
        
    #     return grad_grid, grad_img


def slice_function(
    grid: torch.Tensor,
    img: torch.Tensor) -> torch.Tensor:
    r"""Trilinear Bilateral Grid Transform.

    Args:
        img (torch.Tensor): input image of shape (b, 3, h, w).
        grid (torch.Tensor): output values of the Bilateral grid, shape (b, N, d, h/8, w/8).
    Returns:
        torch.Tensor: transformed image of shape (b, N, h, w).
    """
    return SliceFunction.apply(grid, img)

class ApplyCoeffFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input: torch.Tensor, coeff: torch.Tensor) -> torch.Tensor:
        
        assert input.ndim == 4, "Input must be a 4D tensor (N, C, H, W)"
        assert coeff.ndim == 5, "Coeff must be a 5D tensor (N, groups, coeffs_per_group, H, W)"
        assert input.device == coeff.device, "Input and coeff must be on the same device"

        N, C, H, W = input.shape
        groups, coeffs_per_group = coeff.shape[1], coeff.shape[2]
        output = torch.empty((N, groups, H, W), device=input.device, dtype=input.dtype)

        apply_coeff.apply_coeff_forward(input, coeff, output)

        ctx.save_for_backward(input, coeff)
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        input, coeff = ctx.saved_tensors
        assert grad_output.shape == (input.shape[0], coeff.shape[1], input.shape[2], input.shape[3]), \
            "Grad output shape must match (N, groups, H, W)"

        grad_input = torch.zeros_like(input)
        grad_coeff = torch.zeros_like(coeff)

        apply_coeff.apply_coeff_backward(
            grad_output,
            input, coeff,
            grad_input, grad_coeff
        )
        return grad_input, grad_coeff


def apply_coefficent(img: torch.Tensor, coeff: torch.Tensor) -> torch.Tensor:
    """
    :param img: [N, C, H, W] 
    :param coeff: [N, groups, coeffs_per_group, H, W] 
    :return: [N, groups, H, W] 
    """
    assert img.ndim == 4, "Input must be a 4D tensor (N, C, H, W)"
    assert coeff.ndim == 5, "Coeff must be a 5D tensor (N, groups, coeffs_per_group, H, W)"
    return ApplyCoeffFunction.apply(img, coeff)
