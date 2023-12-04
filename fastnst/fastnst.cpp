#include <cstdio>

int main (int argc, char *argv[]) {
    if (argc < 3) {
        fprintf(stderr, "Usage: fastnst.exe [content.jpg] [style.jpg]");
        return 1;
    }
    printf("content: %s\n", argv[1]);
    printf("style: %s\n", argv[2]);
}