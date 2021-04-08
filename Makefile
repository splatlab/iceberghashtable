CC = clang++
CFLAGS = -g -O3 -m64 -mavx -mavx2 -Wall -march=native -pthread
#CFLAGS = -g -march=native -pthread
INCLUDE = -I ./include
SOURCES = src/iceberg_table.c src/hashutil.c src/partitioned_counter.c
LIBS = -lssl -lcrypto -lpmem

all: src/iceberg_table.c main.cc
	$(CC) $(CFLAGS) $(INCLUDE) $(SOURCES) main.cc -o main $(LIBS)
clean:
	rm main
