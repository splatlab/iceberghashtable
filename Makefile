all: iceberg_table.c main.cc
	g++ -O3 -std=c++11 -m64 -mavx -mavx2 -frename-registers -march=native -Wall -I ./ iceberg_table.c hashutil.c main.cc -o main -lssl -lcrypto
clean:
	rm main
