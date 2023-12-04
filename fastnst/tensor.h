#ifndef TENSOR_H
#define TENSOR_H

typedef struct tensor2d
{
    int dim0, dim1;
    float *data;
} tensor2d_t;

typedef struct tensor3d
{
    int dim0, dim1, dim2;
    float *data;
} tensor3d_t;

inline __attribute__((always_inline))
int idx_3d(tensor3d_t *tensor, int x0, int x1, int x2) {
    return x0 * tensor->dim1 * tensor->dim2 + x1 * tensor->dim2 + x2;
}

typedef struct tensor4d
{
    int dim0, dim1, dim2, dim3;
    float *data;
} tensor4d_t;

#endif