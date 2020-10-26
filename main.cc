#include "iceberg_table.h"
#include <time.h>
#include <thread>
#include <immintrin.h>
#include <tmmintrin.h>
#include <openssl/rand.h>
#include <unistd.h>
#include <chrono>
#include <random>
#include <algorithm>

using namespace std::chrono;

//vectors of key/value pairs in the table and not in the table
std::vector<std::pair<uint64_t, uint64_t>> in_table, not_in_table;

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
	uint64_t N = (1ULL << tbits) * 0.95;
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
		in_table.push_back({vals[i], vals[i + 1]});
		not_in_table.push_back({other_vals[i], other_vals[i + 1]});
	}
	
	printf("\nINSERTIONS\n");

	high_resolution_clock::time_point t1 = high_resolution_clock::now();
	
	uint64_t ct = 0;

	for(uint64_t i = 0; i < N; i++)
		if(!iceberg_insert(table, in_table[i].first, in_table[i].second)) ct++;	
	
	high_resolution_clock::time_point t2 = high_resolution_clock::now();
	printf("Inserts: %f /sec\n", N / elapsed(t1, t2));
	printf("Percent of non-level 1 inserts: %f\n", ((double)ct) / N);

	printf("\nQUERIES\n");

	std::mt19937 g(__builtin_ia32_rdtsc());
	std::shuffle(in_table.begin(), in_table.end(), g);
	
	uint64_t val;
	t1 = high_resolution_clock::now();

	ct = 0;

	for(uint64_t i = 0; i < N; i++)
		if(iceberg_get_value(table, not_in_table[i].first, val)) ct++;
	
	t2 = high_resolution_clock::now();
	printf("Negative queries: %f /sec\n", N / elapsed(t1, t2));
	printf("Error rate: %f\n", ((double)ct) / N);
	
	t1 = high_resolution_clock::now();
		
	ct = 0;

	for(uint64_t i = 0; i < N; i++)
		if(!iceberg_get_value(table, in_table[i].first, val)  || val != in_table[i].second) ct++;

	t2 = high_resolution_clock::now();
	printf("Positive queries: %f /sec\n", N / elapsed(t1, t2));
	printf("Error rate: %f\n", ((double)ct) / N);

	printf("\nREMOVALS\n");

	std::vector<std::pair<uint64_t, uint64_t>> removed, non_removed;

	for(uint64_t i = 0; i < N / 2; ++i) removed.push_back(in_table[i]);
	for(uint64_t i = N / 2; i < N; ++i) non_removed.push_back(in_table[i]);

	shuffle(removed.begin(), removed.end(), g);
	shuffle(non_removed.begin(), non_removed.end(), g);

	t1 = high_resolution_clock::now();
	
	ct = 0;

	for(uint64_t i = 0; i < N / 2; i++)
		if(!iceberg_remove(table, removed[i].first, removed[i].second)) ct++;
	
	t2 = high_resolution_clock::now();
	printf("Removals: %f /sec\n", (N / 2) / elapsed(t1, t2));
	printf("Percent of failed removals: %f\n", ((double)ct) / (N / 2));

	shuffle(removed.begin(), removed.end(), g);

	t1 = high_resolution_clock::now();

	ct = 0;

	for(uint64_t i = 0; i < N / 2; i++)
		if(iceberg_get_value(table, removed[i].first, val) && val == removed[i].second) ct++;

	t2 = high_resolution_clock::now();
	printf("Negative queries after removals: %f /sec\n", N / elapsed(t1, t2));
	printf("Error rate: %f\n", ((double)ct) / (N / 2));

	t1 = high_resolution_clock::now();
		
	ct = 0;

	for(uint64_t i = 0; i < N / 2; i++)
		if(!iceberg_get_value(table, non_removed[i].first, val) || val != non_removed[i].second) ct++;

	t2 = high_resolution_clock::now();
	printf("Positive queries after removals: %f /sec\n", N / elapsed(t1, t2));
	printf("Error rate: %f\n", ct * 2 / (double)N);
}
