all: iceberg_table.c main.cc
	g++ -O3 -m64 -mavx -mavx2 -Wall -frename-registers -march=native -pthread -I ./ iceberg_table.c hashutil.c partitioned_counter.c main.cc -o main -lssl -lcrypto
clean:
	rm main
