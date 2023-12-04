CC = gcc
DEPS = img.o tensor.o fastnst.o
TARGET = fastnst.exe
LIBS = -ljpeg

all: $(DEPS)
	$(CC) $(DEPS) -o $(TARGET) $(LIBS)

img.o:
	$(CC) -c img.c -o img.o

tensor.o:
	$(CC) -c tensor.c -o tensor.o

fastnst.o:
	$(CC) -c fastnst.c -o fastnst.o

clean:
	rm fastnst.exe