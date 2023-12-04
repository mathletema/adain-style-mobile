CC = gcc

all: img.o fastnst.o
	$(CC) img.o fastnst.o -o fastnst.exe -ljpeg

img.o:
	$(CC) -c img.c -o img.o

fastnst.o:
	$(CC) -c fastnst.c -o fastnst.o

clean:
	rm fastnst.exe