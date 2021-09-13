#include <time.h>
#include <thread>
#include <immintrin.h>
#include <tmmintrin.h>
#include <openssl/rand.h>
#include <unistd.h>
#include <chrono>
#include <random>
#include <algorithm>
#include <string.h>
#include <assert.h>

// Dash
#include "Hash.h"
#include "allocator.h"
#include "ex_finger.h"

// pool path and name
static const char *pool_name = "/mnt/pmem1/pmem_hash.data";
// pool size
static const size_t pool_size = 1024ul * 1024ul * 1024ul * 10ul;
// epoch size
#define EPOCH_SIZE 1024

using namespace std::chrono;

//vectors of key/value pairs in the table and not in the table
std::vector<std::pair<uint64_t, uint64_t>> in_table, not_in_table;

Hash<uint64_t> *ht;

double elapsed(high_resolution_clock::time_point t1, high_resolution_clock::time_point t2) {
  return (duration_cast<duration<double>>(t2 - t1)).count();
}

void do_inserts(uint8_t id, uint64_t *keys, uint64_t *values, uint64_t start, uint64_t n) {
  uint64_t num_epochs = (n - 1) / EPOCH_SIZE + 1;
  for (uint64_t epoch = 0; epoch < num_epochs; epoch++) {
    auto epoch_guard = Allocator::AquireEpochGuard();
    for (uint64_t i = 0; i < EPOCH_SIZE; i++) {
      uint64_t curr = start + epoch * EPOCH_SIZE + i;
      if (curr - start >= n) {
        break;
      }
      //auto ret = ht->Insert(keys[curr], (const char *)&values[curr], true);
      auto ret = ht->Insert(keys[curr], DEFAULT, true);
      if (ret == -1) {
        printf("Duplicate insert\n");
        exit(0);
      }
    }
  }
}

void do_queries(uint8_t id, uint64_t *keys, uint64_t start, uint64_t n, bool positive) {
  uint64_t num_epochs = (n - 1) / EPOCH_SIZE + 1;
  for (uint64_t epoch = 0; epoch < num_epochs; epoch++) {
    auto epoch_guard = Allocator::AquireEpochGuard();
    for (uint64_t i = 0; i < EPOCH_SIZE; i++) {
      uint64_t curr = start + epoch * EPOCH_SIZE + i;
      if (curr - start >= n) {
        break;
      }
      auto ret = ht->Get(keys[curr], true);
      if ((ret == NONE) == positive) {
        if (positive) {
          printf("False negative query key: %lu\n", keys[i]);
        } else {
          printf("False positive query\n");
        }
        exit(0);
      }
    }
  }
}

void do_removals(uint8_t id, uint64_t *keys, uint64_t start, uint64_t n) {
  uint64_t num_epochs = (n - 1) / EPOCH_SIZE + 1;
  for (uint64_t epoch = 0; epoch < num_epochs; epoch++) {
    auto epoch_guard = Allocator::AquireEpochGuard();
    for (uint64_t i = 0; i < EPOCH_SIZE; i++) {
      uint64_t curr = start + epoch * EPOCH_SIZE + i;
      if (curr - start >= n) {
        break;
      }
      auto ret = ht->Delete(keys[curr], true);
      if (ret == -1) {
        printf("Failed removal\n");
        exit(0);
      }
    }
  }
}

void safe_rand_bytes(unsigned char *v, size_t n) {

  for (uint64_t i = 0; i < n; i++) {
    v[i] = rand();
  }
}

int main (int argc, char** argv) {

  if (argc != 3 && argc != 4) {
    fprintf(stderr, "Specify the log of the number of slots in the table and the number of threads to use.\n");
    exit(1);
  }

  bool is_benchmark = false;
  if (argc == 4) {
    assert(strcmp(argv[3], "-b") == 0);
    is_benchmark = true;
  }

  uint64_t tbits = atoi(argv[1]);
  uint64_t initbits = tbits - 9;
  //uint64_t initbits = 16;
  uint64_t threads = atoi(argv[2]);
  uint64_t N = (1ULL << tbits) * 0.95;

  high_resolution_clock::time_point t1 = high_resolution_clock::now();

  // Step 1: create (if not exist) and open the pool
  bool file_exist = false;
  if (FileExists(pool_name)) {
    printf("Cannot run on existing pool!\n");
    exit(1);
  }
  Allocator::Initialize(pool_name, pool_size);

  // Step 2: Allocate the initial space for the hash table on PM and get the
  // root; we use Dash-EH in this case.
  ht = reinterpret_cast<Hash<uint64_t> *>(
      Allocator::GetRoot(sizeof(extendible::Finger_EH<uint64_t>)));

  // Step 3: Initialize the hash table
  size_t segment_number = 1 << initbits;
  new (ht) extendible::Finger_EH<uint64_t>(
      segment_number, Allocator::Get()->pm_pool_);

  high_resolution_clock::time_point t2 = high_resolution_clock::now();
  if (!is_benchmark) {
    printf("Creation time: %f\n", elapsed(t1, t2));
  }

  //srand(time(NULL));
  srand(0);

  //Generating vectors of size N for data contained and not contained in the tablea
  //
  uint64_t splits = 1;

  uint64_t size = N / splits / threads;

  N = N / size * size;

  if (!is_benchmark) {
    printf("%ld\n", N * 2 * sizeof(uint64_t));
  }

  uint64_t *in_keys = (uint64_t *)malloc(N * sizeof(uint64_t));
  if(!in_keys) {
    printf("Malloc in_keys failed\n");
    exit(0);
  }
  safe_rand_bytes((unsigned char *)in_keys, sizeof(*in_keys) * N);

  uint64_t *in_values = (uint64_t *)malloc(N * sizeof(uint64_t));
  if(!in_values) {
    printf("Malloc in_values failed\n");
    exit(0);
  }
  safe_rand_bytes((unsigned char *)in_values, sizeof(*in_values) * N);

  uint64_t *out_keys = (uint64_t *)malloc(N * sizeof(uint64_t));
  if(!out_keys) {
    printf("Malloc out_keys failed\n");
    exit(0);
  }
  safe_rand_bytes((unsigned char *)out_keys, sizeof(*out_keys) * N);

  uint64_t *out_values = (uint64_t *)malloc(N * sizeof(uint64_t));
  if(!out_values) {
    printf("Malloc out_values failed\n");
    exit(0);
  }
  safe_rand_bytes((unsigned char *)out_values, sizeof(*out_values) * N);

  if (!is_benchmark) {
    printf("INSERTIONS\n");
  }

  //	exit(0);

  t1 = high_resolution_clock::now();

  std::vector<std::thread> thread_list;
  for(uint64_t i = 0; i < splits; ++i) {

    //t1 = high_resolution_clock::now();

    for(uint64_t j = 0; j < threads; j++)
      thread_list.emplace_back(do_inserts, j, in_keys, in_values, (i * threads + j) * size, size);
    for(uint64_t j = 0; j < threads; j++)
      thread_list[j].join();

    //t2 = high_resolution_clock::now();

    //printf("%f\n", size * threads / elapsed(t1, t2));
    thread_list.clear();
  }

  t2 = high_resolution_clock::now();

  double insert_throughput = N / elapsed(t1, t2);
  //	exit(0);

  if (!is_benchmark) {
    printf("Insertions: %f\n", N / elapsed(t1, t2));
  }

  if (!is_benchmark) {
    printf("\nQUERIES\n");
  }

  std::mt19937 g(__builtin_ia32_rdtsc());
  //std::shuffle(&in_keys[0], &in_keys[N], g);

  //	exit(0);

  t1 = high_resolution_clock::now();

  for(uint64_t i = 0; i < threads; ++i)
    thread_list.emplace_back(do_queries, i, out_keys, i * (N / threads), N / threads, false);
  for(uint64_t i = 0; i < threads; ++i)
    thread_list[i].join();

  t2 = high_resolution_clock::now();
  double negative_throughput = N / elapsed(t1, t2);
  if (!is_benchmark) {
    printf("Negative queries: %f /sec\n", N / elapsed(t1, t2));	
  }
  thread_list.clear();

  //	exit(0);

  t1 = high_resolution_clock::now();

  for(uint64_t i = 0; i < threads; ++i)
    thread_list.emplace_back(do_queries, i, in_keys, i * (N / threads), N / threads, true);
  for(uint64_t i = 0; i < threads; ++i)
    thread_list[i].join();

  t2 = high_resolution_clock::now();
  double positive_throughput = N / elapsed(t1, t2);
  if (!is_benchmark) {
    printf("Positive queries: %f /sec\n", N / elapsed(t1, t2));
  }
  thread_list.clear();

  //	exit(0);

  if (!is_benchmark) {
    printf("\nREMOVALS\n");
  }

  uint64_t num_removed = N / 2 / threads * threads;
  uint64_t *removed = in_keys;
  uint64_t *non_removed = in_keys + num_removed;

  //shuffle(&removed[0], &removed[num_removed], g);
  //shuffle(&non_removed[0], &non_removed[N - num_removed], g);

  t1 = high_resolution_clock::now();

  for(uint64_t i = 0; i < threads; ++i)
    thread_list.emplace_back(do_removals, i, removed, i * (N / 2 / threads), N / 2 / threads);
  for(uint64_t i = 0; i < threads; ++i)
    thread_list[i].join();

  t2 = high_resolution_clock::now();
  double removal_throughput = num_removed / elapsed(t1, t2);
  if (!is_benchmark) {
    printf("Removals: %f /sec\n", num_removed / elapsed(t1, t2));
  }
  thread_list.clear();

  shuffle(&removed[0], &removed[num_removed], g);

  t1 = high_resolution_clock::now();

  if (is_benchmark) {
    printf("%f %f %f %f\n", insert_throughput, negative_throughput, positive_throughput, removal_throughput);
    return 0;
  }

  for(uint64_t i = 0; i < threads; ++i)
    thread_list.emplace_back(do_queries, i, removed, i * (N / 2 / threads), N / 2 / threads, false);
  for(uint64_t i = 0; i < threads; ++i)
    thread_list[i].join();

  t2 = high_resolution_clock::now();
  printf("Negative queries after removals: %f /sec\n", num_removed / elapsed(t1, t2));
  thread_list.clear();

  t1 = high_resolution_clock::now();

  for(uint64_t i = 0; i < threads; ++i)
    thread_list.emplace_back(do_queries, i, non_removed, i * (N / 2 / threads), N / 2 / threads, true);
  for(uint64_t i = 0; i < threads; ++i)
    thread_list[i].join();

  t2 = high_resolution_clock::now();
  printf("Positive queries after removals: %f /sec\n", num_removed / elapsed(t1, t2));
  thread_list.clear();

  /*
     printf("\nMIXED WORKLOAD, HIGH LOAD FACTOR\n");

     for(uint64_t i = 0; i < N; ++i) in_table.push_back(not_in_table[i]);

     shuffle(in_table.begin(), in_table.end(), g);

     t1 = high_resolution_clock::now();

     for(uint64_t i = 0; i < threads; ++i)
     thread_list.emplace_back(do_mixed, i, std::ref(in_table), i * 2 * N / threads, 2 * N / threads);
     for(uint64_t i = 0; i < threads; ++i)
     thread_list[i].join();

     t2 = high_resolution_clock::now();
     printf("Mixed operations at high load factor: %f /sec\n", 4 * N / elapsed(t1, t2));
     thread_list.clear();

     max_size = sum_sizes = 0;
     for(uint64_t i = 0; i < table.metadata.nblocks; ++i) {
     max_size = std::max(max_size, table.metadata.lv3_sizes[i]);
     sum_sizes += table.metadata.lv3_sizes[i];
     }

     printf("Average list size: %f\n", sum_sizes / (double)table.metadata.nblocks);
     printf("Max list size: %ld\n", max_size);*/
}
