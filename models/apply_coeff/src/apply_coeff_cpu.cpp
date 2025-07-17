#include <torch/extension.h>
#include <vector>
#include <ATen/ATen.h>

template <typename scalar_t>
void ApplyCoeffForwardCPU(
    const scalar_t* input,      // [N, C, H, W]
    const scalar_t* coeff,      // [N, groups, coeffs, H, W]
    scalar_t*       output,     // [N, groups, H, W]
    const int N,
    const int C,
    const int H,
    const int W,
    const int groups,
    const int coeffs_per_group
)
{
    
    for (int n = 0; n < N; ++n) {
        
        const int input_offset_n = n * C * H * W;
        const int coeff_offset_n = n * groups * coeffs_per_group * H * W;
        const int output_offset_n= n * groups * H * W;

        for (int g = 0; g < groups; ++g) {
            const int coeff_g_off = coeff_offset_n + g * coeffs_per_group * H * W;
            const int output_g_off= output_offset_n + g * H * W;

            for (int h = 0; h < H; ++h) {
                for (int w_ = 0; w_ < W; ++w_) {
                    int pixel_idx = h * W + w_;
                    scalar_t result = 0;

                    for (int c = 0; c < coeffs_per_group - 1; ++c) {
                        // input index
                        int inp_idx   = input_offset_n + c * H * W + pixel_idx;
                        // coeff index
                        int coeff_idx = coeff_g_off + c * H * W + pixel_idx;
                        result += input[inp_idx] * coeff[coeff_idx];
                    }
                    // bias
                    int bias_idx = coeff_g_off + (coeffs_per_group - 1) * H * W + pixel_idx;
                    result += coeff[bias_idx];

                    int out_idx  = output_g_off + pixel_idx;
                    output[out_idx] = result;
                }
            }
        }
    }
}

void ApplyCoeffCPUForwardLauncher(
    const torch::Tensor &input,    // [N, C, H, W]
    const torch::Tensor &coeff,    // [N, groups, coeffs, H, W]
    torch::Tensor       &output    // [N, groups, H, W]
)
{
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
    TORCH_CHECK(coeff.is_contiguous(), "coeff must be contiguous");
    TORCH_CHECK(output.is_contiguous(), "output must be contiguous");

    int N = input.size(0);
    int C = input.size(1);
    int H = input.size(2);
    int W = input.size(3);

    int groups           = coeff.size(1);
    int coeffs_per_group = coeff.size(2);

    AT_DISPATCH_FLOATING_TYPES(
        input.scalar_type(), "ApplyCoeffForwardCPU", ([&] {
            const scalar_t* inp_data   = input.data_ptr<scalar_t>();
            const scalar_t* coeff_data = coeff.data_ptr<scalar_t>();
            scalar_t*       out_data   = output.data_ptr<scalar_t>();

            ApplyCoeffForwardCPU<scalar_t>(
                inp_data, coeff_data, out_data,
                N, C, H, W,
                groups, coeffs_per_group
            );
        })
    );
}

template <typename scalar_t>
void ApplyCoeffBackwardCPU(
    const scalar_t* grad_out,   // [N, groups, H, W]
    const scalar_t* input,      // [N, C, H, W]
    const scalar_t* coeff,      // [N, groups, coeffs_per_group, H, W]

    scalar_t* grad_input,       // [N, C, H, W]
    scalar_t* grad_coeff,       // [N, groups, coeffs_per_group, H, W]

    const int N,
    const int C,
    const int H,
    const int W,
    const int groups,
    const int coeffs_per_group
)
{
    for (int n = 0; n < N; ++n) {
        int inp_off   = n * C             * H * W;
        int coeff_off = n * groups        * coeffs_per_group * H * W;
        int goff      = n * groups        * H * W;

        for (int g = 0; g < groups; ++g) {
            int coeff_g_off    = coeff_off + g * coeffs_per_group * H * W;
            int grad_out_g_off = goff      + g * H * W;

            for (int h = 0; h < H; ++h) {
                for (int w_ = 0; w_ < W; ++w_) {
                    int pixel_idx = h * W + w_;
                    scalar_t grad_o_val = grad_out[grad_out_g_off + pixel_idx];

                    for (int c = 0; c < coeffs_per_group - 1; ++c) {
                        int inp_idx  = inp_off   + c * H * W + pixel_idx;
                        int cof_idx  = coeff_g_off + c * H * W + pixel_idx;

                        // grad_input[n, c, h, w] += grad_out[n, g, h, w] * coeff[n, g, c, h, w]
                        grad_input[inp_idx] += grad_o_val * coeff[cof_idx];

                        // grad_coeff[n, g, c, h, w] += grad_out[n, g, h, w] * input[n, c, h, w]
                        grad_coeff[cof_idx] += grad_o_val * input[inp_idx];
                    }
                    // bias
                    int bias_idx = coeff_g_off + (coeffs_per_group - 1) * H * W + pixel_idx;
                    grad_coeff[bias_idx] += grad_o_val;
                }
            }
        }
    }
}

void ApplyCoeffCPUBackwardLauncher(
    const torch::Tensor &grad_out,  // [N, groups, H, W]
    const torch::Tensor &input,     // [N, C, H, W]
    const torch::Tensor &coeff,     // [N, groups, coeffs_per_group, H, W]
    torch::Tensor &grad_input,      // [N, C, H, W]
    torch::Tensor &grad_coeff       // [N, groups, coeffs_per_group, H, W]
)
{
    TORCH_CHECK(grad_out.is_contiguous(), "grad_out must be contiguous");
    TORCH_CHECK(input.is_contiguous(),    "input must be contiguous");
    TORCH_CHECK(coeff.is_contiguous(),    "coeff must be contiguous");
    TORCH_CHECK(grad_input.is_contiguous(),"grad_input must be contiguous");
    TORCH_CHECK(grad_coeff.is_contiguous(),"grad_coeff must be contiguous");

    int N = input.size(0);
    int C = input.size(1);
    int H = input.size(2);
    int W = input.size(3);

    int groups            = coeff.size(1);
    int coeffs_per_group  = coeff.size(2);

    AT_DISPATCH_FLOATING_TYPES(
        input.scalar_type(), "ApplyCoeffBackwardCPU", ([&] {
            const scalar_t* g_o_data   = grad_out.data_ptr<scalar_t>();
            const scalar_t* inp_data   = input.data_ptr<scalar_t>();
            const scalar_t* cof_data   = coeff.data_ptr<scalar_t>();

            scalar_t* g_in_data  = grad_input.data_ptr<scalar_t>();
            scalar_t* g_cf_data  = grad_coeff.data_ptr<scalar_t>();

            ApplyCoeffBackwardCPU<scalar_t>(
                g_o_data, inp_data, cof_data,
                g_in_data, g_cf_data,
                N, C, H, W,
                groups, coeffs_per_group
            );
        })
    );
}
