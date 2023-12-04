#include "tensor.h"

tensor3d_t* conv2d(tensor3d_t *inp, tensor4d_t *kernel, tensor3d_t *bias) {
    int N_out = kernel->dim0;
    int N_in = kernel->dim1;
    int kernel_size = kernel->dim2;

    int H = inp->dim1;
    int W = inp->dim2;

    tensor3d_t *out = init_tensor3d(N_out, H, W);

    for (int out_channel = 0; out_channel < N_out; out_channel++) {
        for (int in_channel = 0; in_channel < N_in; in_channel++) {
            for (int y = 0; y < H; y++) {
                for (int x = 0; x < W; x++) {
                    for (int k1 = 0; k1 < kernel_size; k1++) {
                        for (int k2 = 0; k2 < kernel_size; k2++) {
                            out->data[idx_3d(out, out_channel, y, x)] += inp->data[idx_3d(inp, in_channel, y, x)] * kernel->data[idx_4d(kernel, out_channel, in_channel, k1, k2)];
                        }
                    }
                }
            }
        }
    }

    return out;
}