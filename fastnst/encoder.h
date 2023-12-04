#ifndef ENCODER_H
#define ENCODER_H

#include "tensor.h"

typedef struct encoder_weights {
    tensor4d_t *conv2d_weight[10];
    tensor3d_t *conv2d_bias[10];
} encoder_weights_t;

encoder_weights_t* load_weights_encoder(const char* path);
tensor3d_t* encoder(tensor3d_t* inp, encoder_weights_t* weights);

#endif`