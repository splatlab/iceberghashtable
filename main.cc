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
	uint64_t N = (1ULL << tbits) * 1.09;

	
	high_resolution_clock::time_point t1 = high_resolution_clock::now();

	if ((table = iceberg_init(tbits)) == NULL) {
		fprintf(stderr, "Can't allocate iceberg table.\n");
		exit(EXIT_FAILURE);
	}


	high_resolution_clock::time_point t2 = high_resolution_clock::now();
	printf("Creation time: %f\n", elapsed(t1, t2));

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

	printf("INSERTIONS\n");

	uint64_t splits = 19;

	N -= N % splits;
	while(in_table.size() % splits) {
		in_table.pop_back();
		not_in_table.pop_back();
	}

	uint64_t size = N / splits;
	uint64_t ct = 0;

	for(uint64_t i = 0; i < splits; ++i) {
		
		t1 = high_resolution_clock::now();

		for(uint64_t j = 0; j < size; ++j)
			if(!iceberg_insert(table, in_table[i * size + j].first, in_table[i * size + j].second))
				ct++;

		t2 = high_resolution_clock::now();

		printf("%f\n", size / elapsed(t1, t2));
		//printf("Inserts at load factor %f: %f\n", iceberg_load_factor(table), size / elapsed(t1, t2));
		printf("%ld %ld\n", table->metadata->lv3_balls, table->metadata->lv2_balls);
	}
	
	printf("Percent of failed inserts: %f\n", ((double)ct) / N);
	printf("Number level 1 inserts: %ld\n", table->metadata->total_balls - table->metadata->lv2_balls - table->metadata->lv3_balls);
	printf("Number level 2 inserts: %ld\n", table->metadata->lv2_balls);
	printf("Number level 3 inserts: %ld\n", table->metadata->lv3_balls);
	printf("Total inserts: %ld\n", table->metadata->total_balls);

	uint64_t max_size = 0, sum_sizes = 0;
	for(uint64_t i = 0; i < table->metadata->nblocks; ++i) {
		max_size = std::max(max_size, table->metadata->lv3_sizes[i]);
		sum_sizes += table->metadata->lv3_sizes[i];
	}

	printf("Average list size: %f\n", sum_sizes / (double)table->metadata->nblocks);
	printf("Max list size: %ld\n", max_size);

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
	printf("Load factor: %f\n", iceberg_load_factor(table));

	printf("\nREMOVALS\n");

	std::vector<std::pair<uint64_t, uint64_t>> removed, non_removed;

	for(uint64_t i = 0; i < N / 2; ++i) removed.push_back(in_table[i]);
	for(uint64_t i = N / 2; i < N; ++i) non_removed.push_back(in_table[i]);

	shuffle(removed.begin(), removed.end(), g);
	shuffle(non_removed.begin(), non_removed.end(), g);

	t1 = high_resolution_clock::now();
	
	ct = 0;

	for(uint64_t i = 0; i < N / 2; i++)
		if(!iceberg_remove(table, removed[i].first)) { ct++; printf("%ld\n", i); if(!iceberg_get_value(table, removed[i].first, val)) ct += 1000; }
	
	t2 = high_resolution_clock::now();
	printf("Removals: %f /sec\n", (N / 2) / elapsed(t1, t2));
	printf("Number of failed removals: %ld\n", ct);
	printf("Percent of failed removals: %f\n", ((double)ct) / (N / 2));
	printf("Load factor: %f\n", iceberg_load_factor(table));

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
	printf("Load factor: %f\n", iceberg_load_factor(table));
}
