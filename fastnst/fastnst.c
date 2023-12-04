#include "img.h"
#include "tensor.h"
#include "encoder.h"
#include "decoder.h"
#include "adain.h"

#include <stdio.h>

#ifndef ENCODER_WEIGHTS
#define ENCODER_WEIGHTS "weights/encoder.pt"
#endif

#ifndef DECODER_WEIGHTS
#define DECODER_WEIGHTS "weights/decoder.pt"
#endif

int main (int argc, char *argv[]) {
    if (argc < 4) {
        fprintf(stderr, "Usage: fastnst.exe [content.jpg] [style.jpg] [generated.jpg]");
        return 1;
    }

    encoder_weights_t *encoder_weights = load_weights_encoder(ENCODER_WEIGHTS);
    decoder_weights_t *decoder_weights = load_weights_decoder(DECODER_WEIGHTS);

    img_t *content = import_img(argv[1]);
    content = resize_img(content, 512, 512);
    
    img_t *style = import_img(argv[2]);
    style = resize_img(style, 512, 512);

    tensor3d_t *content_tensor = to_tensor(content);
    tensor3d_t *style_tensor = to_tensor(style);

    tensor3d_t *content_features = encoder(content_tensor, encoder_weights);
    tensor3d_t *style_features = encoder(style_tensor, encoder_weights);
    tensor3d_t *generated_features = adain(content_features, style_features);
    tensor3d_t *generated_img = decoder(generated_features, decoder_weights);

    img_t *generated = to_img(generated_img);
    export_img(generated, argv[3]);
}