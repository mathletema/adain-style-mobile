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

typedef struct tensor4d
{
    int dim0, dim1, dim2, dim3;
    float *data;
} tensor4d_t;

inline __attribute__((always_inline))
int idx_2d(tensor3d_t *tensor, int x0, int x1) {
    return x0 * tensor->dim2 + x1;
}

inline __attribute__((always_inline))
int idx_3d(tensor3d_t *tensor, int x0, int x1, int x2) {
    return x0 * tensor->dim1 * tensor->dim2 + x1 * tensor->dim2 + x2;
}

inline __attribute__((always_inline))
int idx_4d(tensor4d_t *tensor, int x0, int x1, int x2, int x3) {
    return x0 * tensor->dim1 * tensor->dim2 * tensor->dim3 + x1 * tensor->dim2 * tensor->dim3 + x2 * tensor->dim3 + x3;
}

inline __attribute__((always_inline))
tensor2d_t* init_tensor2d(int dim0, int dim1) {
    tensor2d_t* res = malloc(sizeof(tensor2d_t));
    res->dim0 = dim0;
    res->dim1 = dim1;
    res->data = malloc(dim0 * dim1 * sizeof(float));
    return res;
}

inline __attribute__((always_inline))
tensor3d_t* init_tensor3d(int dim0, int dim1, int dim2) {
    tensor3d_t* res = malloc(sizeof(tensor3d_t));
    res->dim0 = dim0;
    res->dim1 = dim1;
    res->dim2 = dim2;
    res->data = malloc(dim0 * dim1 * dim2 * sizeof(float));
    return res;
}

inline __attribute__((always_inline))
tensor4d_t* init_tensor4d(int dim0, int dim1, int dim2, int dim3) {
    tensor4d_t* res = malloc(sizeof(tensor4d_t));
    res->dim0 = dim0;
    res->dim1 = dim1;
    res->dim2 = dim2;
    res->dim3 = dim3;
    res->data = malloc(dim0 * dim1 * dim2 * dim3 * sizeof(float));
    return res;
}

tensor3d_t* conv2d(tensor3d_t *inp, tensor4d_t *kernel, tensor3d_t *bias);

#endif