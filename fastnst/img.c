#include "img.h"
#include "tensor.h"

#include <stdio.h>
#include <jpeglib.h>

img_t* import_img(const char* path) {
    struct jpeg_decompress_struct info;
	struct jpeg_error_mgr err;

    FILE* fHandle;
    if ((fHandle = fopen(path, "rb")) == NULL) {
        fprintf(stderr, "could not open file!\n");
        return NULL;
    }

    info.err = jpeg_std_error(&err);
	jpeg_create_decompress(&info);

	jpeg_stdio_src(&info, fHandle);
	jpeg_read_header(&info, 1);

	jpeg_start_decompress(&info);
    if (info.num_components != NUM_CHANNELS) {
        fprintf(stderr, "img does not have 3 channels!\n");
        return NULL;
    }
    
    img_t *img = malloc(sizeof(img_t));
    img->width = info.output_width;
    img->height = info.output_height;
    img->pixels = malloc(NUM_CHANNELS * img->width * img->height * sizeof(uint8_t));

    JSAMPARRAY buffer;
    int row_stride = img->width * info.output_components;
    buffer = (*info.mem->alloc_sarray)((j_common_ptr)&info, JPOOL_IMAGE, row_stride, 1);

    uint8_t *current_row = img->pixels;
     while (info.output_scanline < img->height) {
        jpeg_read_scanlines(&info, buffer, 1);
        for (int i = 0; i < row_stride; ++i) {
            *current_row++ = buffer[0][i];
        }
    }

    jpeg_finish_decompress(&info);
    jpeg_destroy_decompress(&info);

    fclose(fHandle);

    return img;
}

void export_img(const img_t *img, const char *path) {
    printf("exporting..!\n");

    FILE *fHandle = fopen(path, "wb");

    if (!fHandle) {
        fprintf(stderr, "could not open file!\n");
        return;
    }

    struct jpeg_compress_struct info;
    struct jpeg_error_mgr err;

    info.err = jpeg_std_error(&err);
    jpeg_create_compress(&info);

    jpeg_stdio_dest(&info, fHandle);

    info.image_width = img->width;
    info.image_height = img->height;
    info.input_components = NUM_CHANNELS;
    info.in_color_space = JCS_RGB;

    jpeg_set_defaults(&info);
    jpeg_set_quality(&info, 100, TRUE); // Adjust quality as needed

    jpeg_start_compress(&info, TRUE);

    JSAMPROW buffer[1];
    int row_stride = img->width * NUM_CHANNELS;

    while (info.next_scanline < info.image_height) {
        buffer[0] = &img->pixels[info.next_scanline * row_stride];
        jpeg_write_scanlines(&info, buffer, 1);
    }

    jpeg_finish_compress(&info);
    jpeg_destroy_compress(&info);
    fclose(fHandle);
}

img_t* resize_img(const img_t* img, const int width, const int height) {
    img_t* res = malloc(sizeof(img_t));
    res->width = width;
    res->height = height;
    res->pixels = malloc(NUM_CHANNELS * width * height * sizeof(uint8_t));

    double x_ratio = (double) img->width / width;
    double y_ratio = (double) img->height / height;

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int nearest_x = (int)(x * x_ratio);
            int nearest_y = (int)(y * y_ratio);
            
            int img_index = nearest_y * img->width + nearest_x;
            int res_index = y * width + x;
            
            for (int c = 0; c < NUM_CHANNELS; c++) {
                res->pixels[NUM_CHANNELS * res_index + c] = img->pixels[NUM_CHANNELS * img_index + c];
            }
        }
    }

    return res;
}

tensor3d_t* to_tensor(const img_t* img) {
    if (img->width != INP_WIDTH || img->height != INP_HEIGHT) {
        fprintf(stderr, "img not resized yet!\n");
        return NULL;
    }
    
    tensor3d_t* res = malloc(sizeof(tensor3d_t));
    res->dim0 = NUM_CHANNELS;
    res->dim1 = img->height;
    res->dim2 = img->width;
    res->data = malloc(NUM_CHANNELS * img->width * img->height * sizeof(float));

    for (int c = 0; c < NUM_CHANNELS; c++) {
        for (int y = 0; y < img->height; y++) {
            for (int x = 0; x < img->width; x++) {
                int img_idx = y * img->width + x;
                res->data[idx_3d(res, c, y, x)] = img->pixels[NUM_CHANNELS * img_idx + c];
            }
        }
    }

    return res;
}