#include <onnxruntime_cxx_api.h>
#include <array>
#include <cmath>
#include <cstdio>
#include <algorithm>
#include <jpeglib.h>
#include <iostream>

#define NUM_CHANNELS 3
#define IMG_WIDTH 512
#define IMG_HEIGHT 512
#define RGB_MAX 256

typedef struct img {
    int width, height;
    uint8_t *pixels;
} img_t;

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
    
    img_t *img = (img_t*) malloc(sizeof(img_t));
    img->width = info.output_width;
    img->height = info.output_height;
    img->pixels = (uint8_t*) malloc(NUM_CHANNELS * img->width * img->height * sizeof(uint8_t));

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

img_t* resize_img(const img_t* img, const int width, const int height) {
    img_t* res = (img_t*) malloc(sizeof(img_t));
    res->width = width;
    res->height = height;
    res->pixels = (uint8_t*) malloc(NUM_CHANNELS * width * height * sizeof(uint8_t));

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

void free_img(img_t* img) {
    free(img->pixels);
    free(img);
}

std::vector<float> combine_images(img_t* content, img_t* style) {
    const int input_tensor_size = 2 * NUM_CHANNELS * IMG_WIDTH * IMG_HEIGHT;
    std::vector<float> input_tensor_values(input_tensor_size);

    for (int c = 0; c < NUM_CHANNELS; c++) {
        for (int y = 0; y < IMG_HEIGHT; y++) {
            for (int x = 0; x < IMG_WIDTH; x++) {
                int idx = IMG_WIDTH * y + x;
                input_tensor_values[c * IMG_WIDTH * IMG_HEIGHT + idx] = content->pixels[NUM_CHANNELS * idx + c] / RGB_MAX;
            }
        }
    }
    for (int c = 0; c < NUM_CHANNELS; c++) {
        for (int y = 0; y < IMG_HEIGHT; y++) {
            for (int x = 0; x < IMG_WIDTH; x++) {
                int idx = IMG_WIDTH * y + x;
                input_tensor_values[NUM_CHANNELS * IMG_WIDTH * IMG_HEIGHT + c * IMG_WIDTH * IMG_HEIGHT + idx] = style->pixels[NUM_CHANNELS * idx + c] / RGB_MAX;
            }
        }
    }

    return input_tensor_values;
}



int main(int argc, char** argv) {
    /** LOAD ENCODER **/
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "encoder");
    Ort::SessionOptions session_options;
    Ort::Session session = Ort::Session(env, "encoder.onnx", session_options); 

    /** LOAD IMAGES **/
    img_t* content_raw = import_img("cat.jpg");
    img_t* style_raw = import_img("cat.jpg");

    /** RESIZE IMAGES **/
    img_t* content = resize_img(content_raw, IMG_WIDTH, IMG_HEIGHT);
    img_t* style = resize_img(style_raw, IMG_WIDTH, IMG_HEIGHT);
    free_img(content_raw);
    free_img(style_raw);

    Ort::AllocatorWithDefaultOptions allocator;

    // print number of model input nodes
    const size_t num_input_nodes = session.GetInputCount();
    std::vector<Ort::AllocatedStringPtr> input_names_ptr;
    std::vector<const char*> input_node_names;
    input_names_ptr.reserve(num_input_nodes);
    input_node_names.reserve(num_input_nodes);
    std::vector<int64_t> input_node_dims;  // simplify... this model has only 1 input node {1, 3, 224, 224}.
                                            // Otherwise need vector<vector<>>

     std::cout << "Number of inputs = " << num_input_nodes << std::endl;

    // iterate over all input nodes
    for (size_t i = 0; i < num_input_nodes; i++) {
        // print input node names
        auto input_name = session.GetInputNameAllocated(i, allocator);
        std::cout << "Input " << i << " : name =" << input_name.get() << std::endl;
        input_node_names.push_back(input_name.get());
        input_names_ptr.push_back(std::move(input_name));

        // print input node types
        auto type_info = session.GetInputTypeInfo(i);
        auto tensor_info = type_info.GetTensorTypeAndShapeInfo();

        ONNXTensorElementDataType type = tensor_info.GetElementType();
        std::cout << "Input " << i << " : type = " << type << std::endl;

        // print input shapes/dims
        input_node_dims = tensor_info.GetShape();
        std::cout << "Input " << i << " : num_dims = " << input_node_dims.size() << '\n';
        for (size_t j = 0; j < input_node_dims.size(); j++) {
        std::cout << "Input " << i << " : dim[" << j << "] =" << input_node_dims[j] << '\n';
        }
        std::cout << std::flush;
    }

    const int input_tensor_size = 2 * NUM_CHANNELS * IMG_WIDTH * IMG_HEIGHT;
    auto input_tensor_values = combine_images(content, style);
    
    std::vector<const char*> output_node_names = {"self_30_1"};

    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    auto input_tensor = Ort::Value::CreateTensor<float>(memory_info, input_tensor_values.data(), input_tensor_size,
                                                                input_node_dims.data(), 4);
    printf("input is tensor: %s\n", input_tensor.IsTensor() ? "true" : "false");

    auto output_tensors = session.Run(Ort::RunOptions{nullptr}, input_node_names.data(), &input_tensor, 1, output_node_names.data(), 1);
    printf("number of outputs: %ld\n", output_tensors.size());
    printf("output is tensor: %s\n", output_tensors.front().IsTensor() ? "true" : "false");

    float* feats = output_tensors.front().GetTensorMutableData<float>();

    printf("content dims: %dx%d\n", content->width, content->height);
    printf("style dims: %dx%d\n", style->width, style->height);

    /** FREE MEMORY **/
    free_img(content);
    free_img(style);
}