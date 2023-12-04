#include "img.h"

#include <stdio.h>

int main (int argc, char *argv[]) {
    if (argc < 3) {
        fprintf(stderr, "Usage: fastnst.exe [content.jpg] [style.jpg]");
        return 1;
    }
    printf("content: %s\n", argv[1]);
    printf("style: %s\n", argv[2]);

    img_t *content = import_img(argv[1]);
    img_t *content_small = resize_img(content, 512, 512);
    export_img(content_small, "out.jpg");
    printf("width: %d, height: %d\n", content->width, content->height);
}