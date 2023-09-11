CC=nvcc
CFLAGS=-lcublas -lcusparse
SIZE=32768
cudamake: cudamake.cu cudafunc.cu
	$(CC) -O0 -o cudamake.o cudamake.cu cudafunc.cu $(CFLAGS)

clean:
	rm *.o

run:
	./cudamake.o $(SIZE)
