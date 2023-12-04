#ifndef DECODER_H
#define DECODER_H

#include "tensor.h"

typedef struct decoder_weights {
    int hi;
} decoder_weights_t;

decoder_weights_t* load_weights_decoder(const char* path);
tensor3d_t* decoder(tensor3d_t* inp, decoder_weights_t* weights);

#endif`