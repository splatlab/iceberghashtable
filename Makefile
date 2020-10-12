all: iceberg_table.c main.cc
	g++ -O3 -std=c++11 -m64 -frename-registers -march=native -I ./ iceberg_table.c main.cc -o main -lssl -lcrypto
clean:
	rm main
