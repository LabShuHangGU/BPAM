#include <torch/extension.h>

#include <ATen/ATen.h>

// ==================================================
//  CPU Forward Implementation
// ==================================================
template <typename scalar_t>
void TriLinearCPUSliceForward(
    const torch::Tensor &grid,   // [N, grid_channels, D, H_g, W_g]
    const torch::Tensor &input,  // [N, 3*k, H, W]
    torch::Tensor &output        // [N, grid_channels, H, W]
) {
    // Get tensor dimensions
    const int batch_size = grid.size(0);
    const int grid_channels = grid.size(1);
    const int depth = grid.size(2);
    const int grid_h = grid.size(3);
    const int grid_w = grid.size(4);

    const int input_ch = input.size(1);
    const int height = input.size(2);
    const int width = input.size(3);
    const int pixel_count = height * width;

    // k is the number of (x,y,s) coordinate groups
    const int k = input_ch / 3;
    const int sub_ch = grid_channels / k;

    // Use accessors for safe and easy multi-dimensional access
    auto grid_acc = grid.accessor<scalar_t, 5>();
    auto input_acc = input.accessor<scalar_t, 4>();
    auto output_acc = output.accessor<scalar_t, 4>();

    // Loop over each item in the batch
    for (int b = 0; b < batch_size; ++b) {
        // The two inner loops replicate the CUDA grid
        // Loop over each output channel (equivalent to blockIdx.y * blockDim.y + threadIdx.y)
        for (int gc = 0; gc < grid_channels; ++gc) {
            // Loop over each pixel (equivalent to blockIdx.x * blockDim.x + threadIdx.x)
            for (int pixel_id = 0; pixel_id < pixel_count; ++pixel_id) {
                // ====================================================================
                // This section is a direct translation of the CUDA kernel's logic
                // ====================================================================
                
                // From pixel_id to (h, w) coordinates
                const int h = pixel_id / width;
                const int w = pixel_id % width;

                // Determine which (x,y,s) group to use based on the grid channel
                const int chunk_id = gc / sub_ch; // [0..k-1]

                // Get (x,y,s) coordinates from the input tensor
                // These are assumed to be in the range [-1, 1]
                const scalar_t x = input_acc[b][chunk_id * 3 + 0][h][w];
                const scalar_t y = input_acc[b][chunk_id * 3 + 1][h][w];
                const scalar_t s = input_acc[b][chunk_id * 3 + 2][h][w];

                // --- Trilinear Interpolation Logic ---

                // 1. Normalize coordinates from [-1, 1] to grid index space [0, dim-1]
                scalar_t ix = ((x + 1) * grid_w - 1) * 0.5f;
                scalar_t iy = ((y + 1) * grid_h - 1) * 0.5f;
                scalar_t is = ((s + 1) * depth - 1) * 0.5f;

                // 2. Clamp coordinates to be within valid grid boundaries
                ix = std::fmin(std::fmax(ix, static_cast<scalar_t>(0.0)), static_cast<scalar_t>(grid_w - 1.0));
                iy = std::fmin(std::fmax(iy, static_cast<scalar_t>(0.0)), static_cast<scalar_t>(grid_h - 1.0));
                is = std::fmin(std::fmax(is, static_cast<scalar_t>(0.0)), static_cast<scalar_t>(depth - 1.0));

                // 3. Get the integer corners of the surrounding 8-point cube in the grid
                const int ix0 = static_cast<int>(ix);
                const int iy0 = static_cast<int>(iy);
                const int is0 = static_cast<int>(is);
                const int ix1 = std::min(ix0 + 1, grid_w - 1);
                const int iy1 = std::min(iy0 + 1, grid_h - 1);
                const int is1 = std::min(is0 + 1, depth - 1);

                // 4. Calculate the fractional distances (deltas) for interpolation
                const scalar_t x_d = ix - ix0;
                const scalar_t y_d = iy - iy0;
                const scalar_t s_d = is - is0;

                // 5. Calculate the interpolation weights for the 8 corners
                const scalar_t w000 = (1 - x_d) * (1 - y_d) * (1 - s_d);
                const scalar_t w100 = (    x_d) * (1 - y_d) * (1 - s_d);
                const scalar_t w010 = (1 - x_d) * (    y_d) * (1 - s_d);
                const scalar_t w110 = (    x_d) * (    y_d) * (1 - s_d);
                const scalar_t w001 = (1 - x_d) * (1 - y_d) * (    s_d);
                const scalar_t w101 = (    x_d) * (1 - y_d) * (    s_d);
                const scalar_t w011 = (1 - x_d) * (    y_d) * (    s_d);
                const scalar_t w111 = (    x_d) * (    y_d) * (    s_d);

                // 6. Get the values from the grid at the 8 corner locations
                // Note the accessor makes this much cleaner than manual index calculation
                const scalar_t val000 = grid_acc[b][gc][is0][iy0][ix0];
                const scalar_t val100 = grid_acc[b][gc][is0][iy0][ix1];
                const scalar_t val010 = grid_acc[b][gc][is0][iy1][ix0];
                const scalar_t val110 = grid_acc[b][gc][is0][iy1][ix1];
                const scalar_t val001 = grid_acc[b][gc][is1][iy0][ix0];
                const scalar_t val101 = grid_acc[b][gc][is1][iy0][ix1];
                const scalar_t val011 = grid_acc[b][gc][is1][iy1][ix0];
                const scalar_t val111 = grid_acc[b][gc][is1][iy1][ix1];

                // 7. Compute the final interpolated value as the weighted sum
                const scalar_t interp_val =
                    w000 * val000 + w100 * val100 + w010 * val010 + w110 * val110 +
                    w001 * val001 + w101 * val101 + w011 * val011 + w111 * val111;
                
                // 8. Write the result to the output tensor
                output_acc[b][gc][h][w] = interp_val;
            }
        }
    }
}

// ==================================================
//  Launcher function for CPU
// ==================================================
void TriLinearCPUSliceForwardLauncher(
    const torch::Tensor &grid,   // [N, grid_channels, D,H_g,W_g]
    const torch::Tensor &input,  // [N, 3*k, H,W]
    torch::Tensor &output        // [N, grid_channels, H,W]
) {
    // Perform shape checks, similar to the CUDA launcher
    const int grid_channels = grid.size(1);
    const int input_ch = input.size(1);
    
    TORCH_CHECK(input_ch % 3 == 0,
        "Input channels must be multiple of 3 => each (x,y,s) group has 3 channels.");
    const int k = input_ch / 3;

    TORCH_CHECK(output.size(1) == grid_channels,
        "Output's channel dimension must match grid_channels");

    TORCH_CHECK(grid_channels % k == 0,
        "grid_channels must be divisible by k, so we can split it into k sub-chunks.");

    // Dispatch to the appropriate floating point type implementation
    AT_DISPATCH_FLOATING_TYPES(
        input.scalar_type(), "TriLinearCPUSliceForward", ([&] {
            TriLinearCPUSliceForward<scalar_t>(grid, input, output);
        })
    );
}
