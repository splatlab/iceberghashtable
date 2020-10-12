#include "iceberg_table.h"
#include <time.h>
#include <thread>
#include <immintrin.h>  // portable to all x86 compilers
#include <tmmintrin.h>
#include <openssl/rand.h>
#include <unistd.h>
#include <chrono>
#include <random>

using namespace std::chrono;

std::vector<uint64_t> in_keys, in_vals, not_in_keys, not_in_vals;

iceberg_table * restrict table;

double elapsed(high_resolution_clock::time_point t1, high_resolution_clock::time_point t2) {
	return (duration_cast<duration<double>>(t2 - t1)).count();
}

int main (int argc, char** argv) {
	if (argc != 2) {
		fprintf(stderr, "Specify the log of the number of slots in the table.\n");
		exit(1);
	}

	uint64_t tbits = atoi(argv[1]);
	uint64_t N = (1ULL << tbits) * 0.2;
	uint64_t nslots = 1ULL << tbits;

	if ((table = iceberg_init(tbits)) == NULL) {
		fprintf(stderr, "Can't allocate iceberg table.\n");
		exit(EXIT_FAILURE);
	}

	srand(time(NULL));

	//Generating vectors of size N for data contained and not contained in the tablea
	uint64_t *vals = (uint64_t*)malloc(N * 2 * sizeof(uint64_t));
	RAND_bytes((unsigned char *)vals, sizeof(*vals) * N * 2);
	uint64_t *other_vals = (uint64_t*)malloc(N * 2 * sizeof(uint64_t));
	RAND_bytes((unsigned char *)other_vals, sizeof(*other_vals) * N * 2);
	for (uint64_t i = 0; i < N * 2; i += 2) {
		in_keys.push_back(vals[i]);
		in_vals.push_back(vals[i + 1]);
		not_in_keys.push_back(other_vals[i]);
		not_in_vals.push_back(other_vals[i + 1]);
	}
	
	printf("INSERTIONS\n");

	high_resolution_clock::time_point t1 = high_resolution_clock::now();
	
	for(uint64_t i = 0; i < N; i++)
		iceberg_insert(table, in_keys[i], in_vals[i]);	
	
	high_resolution_clock::time_point t2 = high_resolution_clock::now();
	printf("Inserts: %f /sec\n", N / elapsed(t1, t2));
	
	printf("QUERIES\n");
	
	uint64_t val;
	t1 = high_resolution_clock::now();
		
	for(uint64_t i = 0; i < N; i++)
		if(iceberg_get_value(table, not_in_keys[i], val)) {
			printf("False positive query\n");
			exit(EXIT_FAILURE);
		}
	
	t2 = high_resolution_clock::now();
	printf("Negative queries: %f /sec\n", N / elapsed(t1, t2));
	
	t1 = high_resolution_clock::now();
		
	for(uint64_t i = 0; i < N; i++)
		if(!iceberg_get_value(table, in_keys[i], val) && val != in_vals[i]) {
			printf("False negative query\n");
			exit(EXIT_FAILURE);
		}

	t2 = high_resolution_clock::now();
	printf("Positive queries: %f /sec\n", N / elapsed(t1, t2));	
}
