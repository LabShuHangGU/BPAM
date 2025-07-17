#include <torch/extension.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// Forward declarations for the implementation functions
void TriLinearSliceForwardLauncher(const torch::Tensor &grid, const torch::Tensor &input, torch::Tensor &output); 
// CORRECTED: 'output' is now passed by reference (&)
void TriLinearCPUSliceForwardLauncher(const torch::Tensor &grid, const torch::Tensor &input, torch::Tensor &output); 

// The main dispatch function called from Python
// RENAMED: for clarity, as it handles both CPU and CUDA
void trilinearslice_forward(const torch::Tensor &grid, const torch::Tensor &input, torch::Tensor output)
{
    // The device check determines which implementation to use
    if (input.device().is_cuda()) {
        CHECK_INPUT(input);
        CHECK_INPUT(grid);
        CHECK_INPUT(output);
        TriLinearSliceForwardLauncher(grid, input, output);
    }
    else
    {
        CHECK_CONTIGUOUS(input);
        CHECK_CONTIGUOUS(grid);
        CHECK_CONTIGUOUS(output);
        TriLinearCPUSliceForwardLauncher(grid, input, output);
    }
}

// pybind11 module definition
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  
  m.def("tri_forward", &trilinearslice_forward, "Trilinear Slice forward (dispatches to CUDA or CPU)");
}

