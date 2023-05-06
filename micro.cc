#include "iceberg_table.h"
#include <algorithm>
#include <assert.h>
#include <chrono>
#include <fstream>
#include <immintrin.h>
#include <iostream>
#include <limits.h>
#include <random>
#include <string.h>
#include <thread>
#include <time.h>
#include <tmmintrin.h>
#include <unistd.h>

#include <sys/resource.h>
#include <sys/time.h>

using namespace std::chrono;

// vectors of key/value pairs in the table and not in the table
std::vector<std::pair<uint64_t, uint64_t>> in_table, not_in_table;

iceberg_table table;

double
elapsed(high_resolution_clock::time_point t1,
        high_resolution_clock::time_point t2)
{
  return (duration_cast<duration<double>>(t2 - t1)).count();
}

void
do_inserts(uint8_t   id,
           uint64_t *keys,
           uint64_t *values,
           uint64_t  start,
           uint64_t  n)
{
#ifdef LATENCY
  std::vector<double> times;
#endif
  for (uint64_t i = start; i < start + n; ++i) {
#ifdef LATENCY
    high_resolution_clock::time_point t1 = high_resolution_clock::now();
#endif
    if (!iceberg_insert(&table, keys[i], values[i], id)) {
      printf("Failed insert\n");
      exit(0);
    }
#ifdef LATENCY
    high_resolution_clock::time_point t2 = high_resolution_clock::now();
    times.emplace_back(duration_cast<nanoseconds>(t2 - t1).count());
#endif
    // uint64_t val;
    // for(uint64_t j = start; j <= i; ++j) {
    //   if (iceberg_query(&table, keys[j], &val, id) != true) {
    //     printf("False negative query key: 0x%" PRIx64 "\n", keys[j]);
    //     assert(0);
    //   }
    // }
  }
#ifdef LATENCY
  std::ofstream f;
  f.open("insert_times_" + std::to_string(id) + ".log");
  for (auto time : times) {
    f << time << '\n';
  }
  f.close();
#endif
}

void
do_queries(uint8_t   id,
           uint64_t *keys,
           uint64_t  start,
           uint64_t  n,
           bool      positive)
{

  uint64_t val;
#ifdef LATENCY
  std::vector<double> times;
#endif
  for (uint64_t i = start; i < start + n; ++i) {
#ifdef LATENCY
    high_resolution_clock::time_point t1 = high_resolution_clock::now();
#endif
    if (iceberg_query(&table, keys[i], &val, id) != positive) {
      if (positive)
        printf("False negative query key: %8" PRIu64 " : "
               "%" PRIu64 "\n",
               i,
               keys[i]);
      else
        printf("False positive query\n");
      exit(0);
    }
#ifdef LATENCY
    high_resolution_clock::time_point t2 = high_resolution_clock::now();
    times.emplace_back(duration_cast<nanoseconds>(t2 - t1).count());
#endif
  }
#ifdef LATENCY
  std::ofstream f;
  f.open("query_times_" + std::to_string(positive) + "_" + std::to_string(id) +
         ".log");
  for (auto time : times) {
    f << time << '\n';
  }
  f.close();
#endif
}

void
do_deletions(uint8_t id, uint64_t *keys, uint64_t start, uint64_t n)
{
  // uint64_t val;
  for (uint64_t i = start; i < start + n; ++i)
    if (!iceberg_delete(&table, keys[i], id)) {
      printf("Failed deletion\n");
      exit(0);
    }
}

void
safe_rand_bytes(unsigned char *v, size_t n)
{
  size_t round_size = n >= INT_MAX ? INT_MAX - 1 : n;
  for (uint64_t i = 0; i < round_size; ++i) {
    v[i] = rand();
  }
}

void
do_mixed(uint8_t   id,
         uint64_t *keys,
         uint64_t *values,
         uint64_t  start,
         uint64_t  n)
{
  uint64_t val;
  for (uint64_t i = start; i < start + n; ++i)
    if (iceberg_query(&table, keys[i], &val, id))
      iceberg_delete(&table, keys[i], id);
    else
      iceberg_insert(&table, keys[i], values[i], id);
}

int
main(int argc, char **argv)
{
  if (argc != 3 && argc != 4) {
    fprintf(stderr,
            "Specify the log of the number of slots in the table and the "
            "number of threads to use.\n");
    exit(1);
  }

  bool is_benchmark = false;
  if (argc == 5) {
    assert(strcmp(argv[3], "-b") == 0);
    is_benchmark = true;
  }

  uint64_t tbits   = atoi(argv[1]);
  uint64_t resizes = atoi(argv[2]);
  uint64_t threads = atoi(argv[3]);

  high_resolution_clock::time_point t1 = high_resolution_clock::now();

  iceberg_init(&table, tbits - resizes);

  uint64_t capacity = iceberg_capacity(&table);
  uint64_t N        = capacity * (resizes + 1);

  high_resolution_clock::time_point t2 = high_resolution_clock::now();
  if (!is_benchmark) {
    printf("Creation time: %f\n", elapsed(t1, t2));
  }

  // srand(time(NULL));
  srand(0);

  // Generating vectors of size N for data contained and not contained in the
  // tablea
  uint64_t splits = 1;
#ifdef INSTAN_THRPT
  splits = 19;
#endif

  uint64_t size = N / splits / threads;

  N = N / size * size;

#ifdef INSTAN_THRPT
  uint64_t total_alloc = (N * sizeof(uint64_t) * 4) / 1024;
#endif
  if (!is_benchmark) {
    printf("%" PRIu64 "\n", N * 2 * sizeof(uint64_t));
  }

  uint64_t *in_keys = (uint64_t *)malloc(N * sizeof(uint64_t));
  if (!in_keys) {
    printf("Malloc in_keys failed\n");
    exit(0);
  }
  safe_rand_bytes((unsigned char *)in_keys, sizeof(*in_keys) * N);

  uint64_t *in_values = (uint64_t *)malloc(N * sizeof(uint64_t));
  if (!in_values) {
    printf("Malloc in_values failed\n");
    exit(0);
  }
  safe_rand_bytes((unsigned char *)in_values, sizeof(*in_values) * N);

  uint64_t *out_keys = (uint64_t *)malloc(N * sizeof(uint64_t));
  if (!out_keys) {
    printf("Malloc out_keys failed\n");
    exit(0);
  }
  safe_rand_bytes((unsigned char *)out_keys, sizeof(*out_keys) * N);

  uint64_t *out_values = (uint64_t *)malloc(N * sizeof(uint64_t));
  if (!out_values) {
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
#ifdef INSTAN_THRPT
  struct rusage usage;
#endif
  for (uint64_t i = 0; i < splits; ++i) {
#ifdef INSTAN_THRPT
    high_resolution_clock::time_point t1, t2;
    t1 = high_resolution_clock::now();
#endif
    for (uint64_t j = 0; j < threads; j++)
      thread_list.emplace_back(
        do_inserts, j, in_keys, in_values, (i * threads + j) * size, size);
    for (uint64_t j = 0; j < threads; j++)
      thread_list[j].join();

#ifdef INSTAN_THRPT
    auto num = i * size + size;
    getrusage(RUSAGE_SELF, &usage);
    printf("Num: %ld MaxRSS: %ld\n", num, usage.ru_maxrss - total_alloc);
    t2 = high_resolution_clock::now();
    printf("%f\n", size * threads / elapsed(t1, t2));
#endif
    thread_list.clear();
  }

#ifdef ENABLE_RESIZE
  iceberg_end(&table, 0);
#endif
  t2 = high_resolution_clock::now();

  double insert_throughput = N / elapsed(t1, t2);
  //	exit(0);

  if (!is_benchmark) {
    printf("Insertions: %f\n", N / elapsed(t1, t2));

    printf("Load factor: %f\n", iceberg_load_factor(&table));
    printf("Number level 1 inserts: %" PRIu64 "\n", level1_load(&table));
    printf("Number level 2 inserts: %" PRIu64 "\n", level2_load(&table));
    printf("Number level 3 inserts: %" PRIu64 "\n", level3_load(&table));
    printf("Total inserts: %" PRIu64 "\n", iceberg_load(&table));
  }

  // uint64_t max_size = 0, sum_sizes = 0;
  // for(uint64_t i = 0; i < LEVEL3_BLOCKS; ++i) {
  //   max_size = std::max(max_size, table.metadata.lv3_sizes[i]);
  //   sum_sizes += table.metadata.lv3_sizes[i];
  // }

  if (!is_benchmark) {
    printf("Average list size: %f\n",
           level3_load(&table) / (double)LEVEL3_BLOCKS);
    // printf("Max list size: %" PRIu64 "\n", max_size);

    printf("\nQUERIES\n");
  }

  std::mt19937 g(__builtin_ia32_rdtsc());
  std::shuffle(&in_keys[0], &in_keys[N], g);

  //	exit(0);

  t1 = high_resolution_clock::now();

  for (uint64_t i = 0; i < threads; ++i)
    thread_list.emplace_back(
      do_queries, i, out_keys, i * (N / threads), N / threads, false);
  for (uint64_t i = 0; i < threads; ++i)
    thread_list[i].join();

  t2                         = high_resolution_clock::now();
  double negative_throughput = N / elapsed(t1, t2);
  if (!is_benchmark) {
    printf("Negative queries: %f /sec\n", N / elapsed(t1, t2));
  }
  thread_list.clear();

  //	exit(0);

  t1 = high_resolution_clock::now();

  for (uint64_t i = 0; i < threads; ++i)
    thread_list.emplace_back(
      do_queries, i, in_keys, i * (N / threads), N / threads, true);
  for (uint64_t i = 0; i < threads; ++i)
    thread_list[i].join();

  t2                         = high_resolution_clock::now();
  double positive_throughput = N / elapsed(t1, t2);
  if (!is_benchmark) {
    printf("Positive queries: %f /sec\n", N / elapsed(t1, t2));
  }
  thread_list.clear();

  //	exit(0);

  if (!is_benchmark) {
    printf("\nREMOVALS\n");
  }

  uint64_t  num_deleted = N / 2 / threads * threads;
  uint64_t *deleted     = in_keys;
  uint64_t *non_deleted = in_keys + num_deleted;

  shuffle(&deleted[0], &deleted[num_deleted], g);
  shuffle(&non_deleted[0], &non_deleted[N - num_deleted], g);

  t1 = high_resolution_clock::now();

  for (uint64_t i = 0; i < threads; ++i)
    thread_list.emplace_back(
      do_deletions, i, deleted, i * (N / 2 / threads), N / 2 / threads);
  for (uint64_t i = 0; i < threads; ++i)
    thread_list[i].join();

  t2                        = high_resolution_clock::now();
  double deletion_throughput = num_deleted / elapsed(t1, t2);
  if (!is_benchmark) {
    printf("Removals: %f /sec\n", num_deleted / elapsed(t1, t2));
    printf("Load factor: %f\n", iceberg_load_factor(&table));
  }
  thread_list.clear();

  shuffle(&deleted[0], &deleted[num_deleted], g);

  t1 = high_resolution_clock::now();

  if (is_benchmark) {
    printf("%f %f %f %f\n",
           insert_throughput,
           negative_throughput,
           positive_throughput,
           deletion_throughput);
    return 0;
  }

  for (uint64_t i = 0; i < threads; ++i)
    thread_list.emplace_back(
      do_queries, i, deleted, i * (N / 2 / threads), N / 2 / threads, false);
  for (uint64_t i = 0; i < threads; ++i)
    thread_list[i].join();

  t2 = high_resolution_clock::now();
  printf("Negative queries after deletions: %f /sec\n",
         num_deleted / elapsed(t1, t2));
  thread_list.clear();

  t1 = high_resolution_clock::now();

  for (uint64_t i = 0; i < threads; ++i)
    thread_list.emplace_back(
      do_queries, i, non_deleted, i * (N / 2 / threads), N / 2 / threads, true);
  for (uint64_t i = 0; i < threads; ++i)
    thread_list[i].join();

  t2 = high_resolution_clock::now();
  printf("Positive queries after deletions: %f /sec\n",
         num_deleted / elapsed(t1, t2));
  thread_list.clear();
}
