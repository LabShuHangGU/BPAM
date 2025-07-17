#include <torch/extension.h>

#define CHECK_INPUT_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_INPUT_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT_CPU(x) TORCH_CHECK(!x.device().is_cuda(), #x " must be a CPU tensor")

void ApplyCoeffForwardLauncher(const torch::Tensor &input, const torch::Tensor &coeff, torch::Tensor &output);
void ApplyCoeffBackwardLauncher(const torch::Tensor &grad_out, const torch::Tensor &input, const torch::Tensor &coeff,
                                torch::Tensor &grad_input, torch::Tensor &grad_coeff);

void ApplyCoeffCPUForwardLauncher(const torch::Tensor &input, const torch::Tensor &coeff, torch::Tensor &output);
void ApplyCoeffCPUBackwardLauncher(const torch::Tensor &grad_out, const torch::Tensor &input, const torch::Tensor &coeff,
                                   torch::Tensor &grad_input, torch::Tensor &grad_coeff);

// Forward
void apply_coeff_forward(const torch::Tensor &input,
                         const torch::Tensor &coeff,
                         torch::Tensor &output) {
    if (input.device().is_cuda()) {
        CHECK_INPUT_CUDA(input);
        CHECK_INPUT_CONTIGUOUS(coeff);
        CHECK_INPUT_CONTIGUOUS(output);

        ApplyCoeffForwardLauncher(input, coeff, output);
    } else {
        CHECK_INPUT_CPU(input);
        CHECK_INPUT_CONTIGUOUS(coeff);
        CHECK_INPUT_CONTIGUOUS(output);

        ApplyCoeffCPUForwardLauncher(input, coeff, output);
    }
}

// Backward
void apply_coeff_backward(const torch::Tensor &grad_out,
                          const torch::Tensor &input,
                          const torch::Tensor &coeff,
                          torch::Tensor &grad_input,
                          torch::Tensor &grad_coeff) {
    if (input.device().is_cuda()) {
        CHECK_INPUT_CUDA(grad_out);
        CHECK_INPUT_CUDA(input);
        CHECK_INPUT_CUDA(coeff);
        CHECK_INPUT_CUDA(grad_input);
        CHECK_INPUT_CUDA(grad_coeff);

        ApplyCoeffBackwardLauncher(grad_out, input, coeff, grad_input, grad_coeff);
    } else {
        CHECK_INPUT_CPU(grad_out);
        CHECK_INPUT_CPU(input);
        CHECK_INPUT_CPU(coeff);
        CHECK_INPUT_CPU(grad_input);
        CHECK_INPUT_CPU(grad_coeff);

        ApplyCoeffCPUBackwardLauncher(grad_out, input, coeff, grad_input, grad_coeff);
    }
}

// PyBind
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("apply_coeff_forward", &apply_coeff_forward, "Apply coefficients (forward)");
    m.def("apply_coeff_backward", &apply_coeff_backward, "Apply coefficients (backward)");
}
