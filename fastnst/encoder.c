#include "encoder.h"

#include <stdlib.h>

encoder_weights_t* load_weights_encoder(const char* path) {
    encoder_weights_t *weights = malloc(sizeof(encoder_weights_t));

    int conv_dim[11] = {3, 3, 64, 64, 128, 128, 140, 153, 153, 153, 261};
    int conv_kernel_size[10] = {1, 3, 3, 3, 3, 3, 3, 3, 3, 3};
    
    for (int i = 0; i < 10; i++) {
        weights->conv2d_weight[i] = init_tensor4d(conv_dim[i+1], conv_dim[i], conv_kernel_size[i], conv_kernel_size[i]);
        for (int j = 0; j < conv_dim[i+1]*conv_dim[i]*conv_kernel_size[i]*conv_kernel_size[i]; j++) {
            scanf("%f", weights->conv2d_weight[i]->data + j);
        }
        weights->conv2d_weight[i] = init_tensor3d(conv_dim[i+1], conv_kernel_size[i], conv_kernel_size[i]);
        for (int j = 0; j < conv_dim[i+1]*conv_kernel_size[i]*conv_kernel_size[i]; j++) {
            scanf("%f", weights->conv2d_weight[i]->data + j);
        }
    }
    return weights;
};

tensor3d_t* encoder(tensor3d_t* inp, encoder_weights_t* weights) {
    inp = conv2d(inp, weights->conv2d_weight[0], weights->conv2d_bias[0]);
    inp = conv2d(inp, weights->conv2d_weight[1], weights->conv2d_bias[1]);
    inp = relu(inp);
    inp = conv2d(inp, weights->conv2d_weight[2], weights->conv2d_bias[2]);
    inp = relu(inp);
    inp = maxpool2d(inp);
    inp = conv2d(inp, weights->conv2d_weight[3], weights->conv2d_bias[3]);
    inp = relu(inp);
    inp = conv2d(inp, weights->conv2d_weight[4], weights->conv2d_bias[4]);
    inp = relu(inp);
    inp = maxpool2d(inp);
    inp = conv2d(inp, weights->conv2d_weight[5], weights->conv2d_bias[5]);
    inp = relu(inp);
    inp = conv2d(inp, weights->conv2d_weight[6], weights->conv2d_bias[6]);
    inp = relu(inp);
    inp = conv2d(inp, weights->conv2d_weight[7], weights->conv2d_bias[7]);
    inp = relu(inp);
    inp = conv2d(inp, weights->conv2d_weight[8], weights->conv2d_bias[8]);
    inp = relu(inp);
    inp = maxpool2d(inp);
    inp = conv2d(inp, weights->conv2d_weight[9], weights->conv2d_bias[9]);
    inp = relu(inp);
}