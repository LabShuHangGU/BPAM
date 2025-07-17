#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

template <typename scalar_t>
__global__ void ApplyCoeffForward(
    const scalar_t* __restrict__ input,     // [N, C, H, W]
    const scalar_t* __restrict__ coeff,     // [N, groups, coeffs, H, W]
    scalar_t* __restrict__ output,          // [N, groups, H, W]
    const int channels,
    const int height,
    const int width,
    const int coeffs_per_group,
    const int groups 
) {
    const int pixel_idx = blockIdx.z * blockDim.x + threadIdx.x;
    if (pixel_idx >= height * width) return;

    const int H = pixel_idx / width;
    const int W = pixel_idx % width;

    const int batch_idx = blockIdx.x;       
    const int group_idx = blockIdx.y;       

    const int input_offset = batch_idx * channels * height * width;
    const int coeff_offset = batch_idx * groups * coeffs_per_group * height * width;

    scalar_t result = 0;

    
    for (int c = 0; c < coeffs_per_group - 1; ++c) {  
        const int input_idx = input_offset + c * height * width + H * width + W;
        const int coeff_idx = coeff_offset + group_idx * coeffs_per_group * height * width +
                              c * height * width + H * width + W;
        result += input[input_idx] * coeff[coeff_idx];
    }

    const int bias_idx = coeff_offset + group_idx * coeffs_per_group * height * width +
                         (coeffs_per_group - 1) * height * width + H * width + W;
    result += coeff[bias_idx];

    const int output_idx = batch_idx * groups * height * width + group_idx * height * width + H * width + W;
    output[output_idx] = result;
}


void ApplyCoeffForwardLauncher(
    const torch::Tensor& input,  // [N, C, H, W]
    const torch::Tensor& coeff,  // [N, groups, coeffs, H, W]
    torch::Tensor& output        // [N, groups, H, W]
) {
    const int batch_size = input.size(0);
    const int channels = input.size(1);
    const int height = input.size(2);
    const int width = input.size(3);

    const int groups = coeff.size(1);
    const int coeffs_per_group = coeff.size(2);
    
    if (batch_size <= 0 || groups <= 0 || height <= 0 || width <= 0) {
        throw std::runtime_error("Invalid input dimensions for ApplyCoeffForwardLauncher");
    }

    const int max_threads_per_block = 1024;  
    const int threads = std::min(256, max_threads_per_block);  
    const int pixels_per_block = std::max((height * width + threads - 1) / threads, 1);  
    
    const dim3 blocks(batch_size, groups, pixels_per_block);  
    const dim3 threads_per_block(threads);

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "ApplyCoeffForward", ([&] {
        ApplyCoeffForward<scalar_t>
            <<<blocks, threads_per_block>>>(
                input.data_ptr<scalar_t>(),
                coeff.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                channels,
                height,
                width,
                coeffs_per_group,
                groups
            );
    }));
    AT_CUDA_CHECK(cudaGetLastError());
}

template <typename scalar_t>
__global__ void ApplyCoeffBackward(
    const scalar_t* __restrict__ grad_out,    // [N, groups, H, W]
    const scalar_t* __restrict__ input,       // [N, C, H, W]
    const scalar_t* __restrict__ coeff,       // [N, groups, coeffs_per_group, H, W]

    scalar_t* __restrict__ grad_input,        // [N, C, H, W]
    scalar_t* __restrict__ grad_coeff,        // [N, groups, coeffs_per_group, H, W]

    const int batch_size,
    const int channels,
    const int height,
    const int width,
    const int groups,
    const int coeffs_per_group
)
{
    const int pixel_idx = blockIdx.z * blockDim.x + threadIdx.x;
    if (pixel_idx >= height * width) return;

    const int H = pixel_idx / width;
    const int W = pixel_idx % width;

    const int batch_idx = blockIdx.x;
    const int group_idx = blockIdx.y;

    const int input_offset  = batch_idx * channels * height * width;
    const int coeff_offset  = batch_idx * groups   * coeffs_per_group * height * width;
    const int grad_inp_off  = input_offset;   
    const int grad_coef_off = coeff_offset;   
    const int goff = batch_idx * groups * height * width;  // grad_out offset

    const int grad_out_idx = goff + group_idx * height * width + H * width + W;
    const scalar_t grad_o_val = grad_out[grad_out_idx];

    // grad_input[..., c, h, w] += grad_out[..., g, h, w] * coeff[..., g, c, h, w]
    for (int c = 0; c < coeffs_per_group - 1; ++c) {
        // input_idx
        const int inp_idx = input_offset + c * height * width + H * width + W;
        // coeff_idx
        const int cof_idx = coeff_offset + group_idx * coeffs_per_group * height * width 
                            + c * height * width + H * width + W;

        atomicAdd(
            grad_input + inp_idx, 
            grad_o_val * coeff[cof_idx]
        );
        // grad_coeff[..., c, h, w] += grad_out[..., g, h, w] * input[..., c, h, w]
        atomicAdd(
            grad_coeff + cof_idx,
            grad_o_val * input[inp_idx]
        );
    }

    const int bias_idx = coeff_offset + group_idx * coeffs_per_group * height * width +
                         (coeffs_per_group - 1) * height * width + H * width + W;
    // grad_coeff[..., bias, h, w] += grad_out[..., g, h, w]
    atomicAdd(grad_coeff + bias_idx, grad_o_val);
}

void ApplyCoeffBackwardLauncher(
    const torch::Tensor& grad_out,   // [N, groups, H, W]
    const torch::Tensor& input,      // [N, C, H, W]
    const torch::Tensor& coeff,      // [N, groups, coeffs_per_group, H, W]

    torch::Tensor& grad_input,       // [N, C, H, W]
    torch::Tensor& grad_coeff        // [N, groups, coeffs_per_group, H, W]
) {
    const int batch_size = input.size(0);
    const int channels   = input.size(1);
    const int height     = input.size(2);
    const int width      = input.size(3);

    const int groups            = coeff.size(1);
    const int coeffs_per_group  = coeff.size(2);

    const int threads = 256;
    const int pixels_per_block = (height * width + threads - 1) / threads;
    dim3 blocks(batch_size, groups, pixels_per_block);
    dim3 threads_per_block(threads);

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "ApplyCoeffBackward", ([&] {
        ApplyCoeffBackward<scalar_t>
            <<<blocks, threads_per_block>>>(
                grad_out.data_ptr<scalar_t>(),
                input.data_ptr<scalar_t>(),
                coeff.data_ptr<scalar_t>(),
                grad_input.data_ptr<scalar_t>(),
                grad_coeff.data_ptr<scalar_t>(),
                batch_size,
                channels,
                height,
                width,
                groups,
                coeffs_per_group
            );
    }));
    AT_CUDA_CHECK(cudaGetLastError());
}
