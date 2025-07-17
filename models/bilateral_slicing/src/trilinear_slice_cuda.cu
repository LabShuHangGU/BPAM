
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>
#define BLOCK_X 16
#define BLOCK_Y 16

// ==================================================
//  Forward Kernel
//    grid_channels = sub_ch * k
//    input.shape = [3*k, H, W]
//    output.shape= [grid_channels, H, W]
// ==================================================
template <typename scalar_t>
__global__ void TriLinearSliceForward(
    const scalar_t* __restrict__ grid,   // [grid_channels, D, H_g, W_g] 
    const scalar_t* __restrict__ input,  // [3*k, H, W]                  
    scalar_t*       __restrict__ output, // [grid_channels, H, W]        

    // 形状
    const int height,
    const int width,
    const int depth,
    const int grid_h,
    const int grid_w,
    const int grid_channels,   // = sub_ch * k

    // offset
    const int shift,           // = depth * grid_h * grid_w
    const scalar_t bin_d,      // = 1.f/(depth -1)
    const scalar_t bin_h,      // = 1.f/(grid_h-1)
    const scalar_t bin_w,      // = 1.f/(grid_w-1)

    const int pixel_count,     // = H*W

    const int k                
)
{
    // x=> pixel_id, y=> gc in [0..grid_channels-1]
    int pixel_id = blockIdx.x * blockDim.x + threadIdx.x;
    int gc       = blockIdx.y * blockDim.y + threadIdx.y;

    if (pixel_id >= pixel_count || gc >= grid_channels) {
        return;
    }
    
    int sub_ch   = grid_channels / k;
    int chunk_id = gc / sub_ch; // [0..k-1]
    int c        = gc % sub_ch; // [0..sub_ch-1]

    // input.shape=[3*k, H, W], pixel_id=[0..H*W-1]
    int c_off  = 3 * chunk_id;
    scalar_t x = input[(c_off + 0)*pixel_count + pixel_id];
    scalar_t y = input[(c_off + 1)*pixel_count + pixel_id];
    scalar_t s = input[(c_off + 2)*pixel_count + pixel_id];
    
    scalar_t ix = ((x + 1) * grid_w - 1) * 0.5f;
    scalar_t iy = ((y + 1) * grid_h - 1) * 0.5f;
    scalar_t is = ((s + 1) * depth - 1) * 0.5f;
    ix = fminf(fmaxf(ix, 0.0f), grid_w - 1.0f);
    iy = fminf(fmaxf(iy, 0.0f), grid_h - 1.0f);
    is = fminf(fmaxf(is, 0.0f), depth - 1.0f);

    int ix0 = static_cast<int>(ix);
    int iy0 = static_cast<int>(iy);
    int is0 = static_cast<int>(is);
    int ix1 = min(ix0 + 1, grid_w - 1);
    int iy1 = min(iy0 + 1, grid_h - 1);
    int is1 = min(is0 + 1, depth - 1);
    
    scalar_t x_d = ix - ix0;
    scalar_t y_d = iy - iy0;
    scalar_t s_d = is - is0;

    int id_00 = ix0 + iy0 * grid_w;
    int id_10 = ix1 + iy0 * grid_w;
    int id_01 = ix0 + iy1 * grid_w;
    int id_11 = ix1 + iy1 * grid_w;

    int id000_s = id_00 + is0 * (grid_w*grid_h);
    int id100_s = id_10 + is0 * (grid_w*grid_h);
    int id010_s = id_01 + is0 * (grid_w*grid_h);
    int id110_s = id_11 + is0 * (grid_w*grid_h);

    int id001_s = id_00 + is1 * (grid_w*grid_h);
    int id101_s = id_10 + is1 * (grid_w*grid_h);
    int id011_s = id_01 + is1 * (grid_w*grid_h);
    int id111_s = id_11 + is1 * (grid_w*grid_h);

    scalar_t w000 = (1 - x_d)*(1 - y_d)*(1 - s_d);
    scalar_t w100 = (    x_d)*(1 - y_d)*(1 - s_d);
    scalar_t w010 = (1 - x_d)*(    y_d)*(1 - s_d);
    scalar_t w110 = (    x_d)*(    y_d)*(1 - s_d);

    scalar_t w001 = (1 - x_d)*(1 - y_d)*(    s_d);
    scalar_t w101 = (    x_d)*(1 - y_d)*(    s_d);
    scalar_t w011 = (1 - x_d)*(    y_d)*(    s_d);
    scalar_t w111 = (    x_d)*(    y_d)*(    s_d);

    scalar_t val000 = grid[gc*shift + id000_s];
    scalar_t val100 = grid[gc*shift + id100_s];
    scalar_t val010 = grid[gc*shift + id010_s];
    scalar_t val110 = grid[gc*shift + id110_s];
    scalar_t val001 = grid[gc*shift + id001_s];
    scalar_t val101 = grid[gc*shift + id101_s];
    scalar_t val011 = grid[gc*shift + id011_s];
    scalar_t val111 = grid[gc*shift + id111_s];

    scalar_t interp_val =
        w000*val000 + w100*val100 + w010*val010 + w110*val110 +
        w001*val001 + w101*val101 + w011*val011 + w111*val111;

    output[gc*pixel_count + pixel_id] = interp_val;
}
     
void TriLinearSliceForwardLauncher(
    const torch::Tensor &grid,   // [N, grid_channels, D,H_g,W_g], grid_channels = sub_ch*k
    const torch::Tensor &input,  // [N, 3*k,           H,W]
    torch::Tensor       &output  // [N, grid_channels, H,W]
)
{
    c10::cuda::CUDAGuard device_guard(grid.device());

    int batch_size    = grid.size(0);
    int grid_channels = grid.size(1); // = sub_ch * k
    int depth         = grid.size(2);
    int grid_h        = grid.size(3);
    int grid_w        = grid.size(4);

    // input: [N, 3*k, H, W]
    int input_ch      = input.size(1);
    int height        = input.size(2);
    int width         = input.size(3);
    int pixel_count   = height * width;

    TORCH_CHECK(input_ch % 3 == 0,
        "Input channels must be multiple of 3 => each (x,y,s) group has 3 channels.");
    int k = input_ch / 3; 

    // output: [N, grid_channels, H, W]
    TORCH_CHECK(output.size(1) == grid_channels,
        "Output's channel dimension must match grid_channels");

    TORCH_CHECK(grid_channels % k == 0,
        "grid_channels must be divisible by k, so we can split it into k sub-chunks.");

    // bin
    float bin_d = 1.f / (depth  - 1);
    float bin_h = 1.f / (grid_h - 1);
    float bin_w = 1.f / (grid_w - 1);

    int shift = depth * grid_h * grid_w;

    // 2D grid: 
    //   x => pixel_id in [0..H*W-1]
    //   y => gc in [0..grid_channels-1]
    dim3 block(BLOCK_X, BLOCK_Y);
    dim3 grid_dim(
        (pixel_count    + BLOCK_X - 1) / BLOCK_X,
        (grid_channels + BLOCK_Y - 1) / BLOCK_Y
    );

    for (int b = 0; b < batch_size; ++b) {
        AT_DISPATCH_FLOATING_TYPES(
            input.scalar_type(), "TriLinearSliceForward", ([&] {
                const scalar_t* data_grid  = grid[b].data_ptr<scalar_t>();
                const scalar_t* data_input = input[b].data_ptr<scalar_t>();
                scalar_t*       data_out   = output[b].data_ptr<scalar_t>();

                TriLinearSliceForward<scalar_t>
                    <<<grid_dim, block, 0, at::cuda::getCurrentCUDAStream()>>>(
                        data_grid, 
                        data_input,
                        data_out,
                        // shape
                        height, width,
                        depth, grid_h, grid_w, grid_channels,
                        // offset
                        shift,
                        static_cast<scalar_t>(bin_d),
                        static_cast<scalar_t>(bin_h),
                        static_cast<scalar_t>(bin_w),
                        pixel_count,
                        k
                    );
            })
        );
        AT_CUDA_CHECK(cudaGetLastError());
    }
}