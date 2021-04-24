CC = clang++
CFLAGS = -flto -O3 -m64 -mavx -mavx2 -Wall -march=native -pthread
#CFLAGS = -g -march=native -pthread
INCLUDE = -I ./include
SOURCES =
LIBS = -lssl -lcrypto -lpmem

all: benchmark.cc
	$(CC) $(CFLAGS) $(INCLUDE) $(SOURCES) benchmark.cc -o benchmark $(LIBS)
clean:
	rm benchmark
