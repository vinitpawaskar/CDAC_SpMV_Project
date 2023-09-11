CC=icx
CFLAGS=-qmkl
SIZE=32768
intelmake: intelmake.c intelfunc.c
	$(CC) -O0 -o intelmake.o intelmake.c intelfunc.c $(CFLAGS)

clean:
	rm *.o

run:
	./intelmake.o $(SIZE)
