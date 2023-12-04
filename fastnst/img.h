#ifndef IMG_H
#define IMG_H

#include "tensor.h"

#include <stdint.h>

#define NUM_CHANNELS (3)
#define INP_WIDTH (512)
#define INP_HEIGHT (512)

typedef struct img {
    int width, height;
    uint8_t *pixels;
} img_t;

img_t* import_img(const char* path);
void export_img(const img_t* img, const char* path);
img_t* resize_img(const img_t* img, const int width, const int height);
tensor3d_t* to_tensor(const img_t* img);

#endif