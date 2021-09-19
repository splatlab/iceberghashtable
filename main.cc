#include "iceberg_table.h"
#include <algorithm>
#include <assert.h>
#include <chrono>
#include <immintrin.h>
#include <openssl/rand.h>
#include <random>
#include <string.h>
#include <thread>
#include <time.h>
#include <tmmintrin.h>
#include <unistd.h>

using namespace std::chrono;

// vectors of key/value pairs in the table and not in the table
std::vector<std::pair<uint64_t, uint64_t>> in_table, not_in_table;

iceberg_table *table;

#define THOUSAND (1000UL)
#define MILLION (THOUSAND * THOUSAND)
#define BILLION (THOUSAND * MILLION)

#define USEC_TO_SEC(x) ((x) / MILLION)
#define USEC_TO_NSEC(x) ((x) * THOUSAND)
#define NSEC_TO_SEC(x) ((x) / BILLION)
#define NSEC_TO_MSEC(x) ((x) / MILLION)
#define NSEC_TO_USEC(x) ((x) / THOUSAND)
#define SEC_TO_MSEC(x) ((x) * THOUSAND)
#define SEC_TO_USEC(x) ((x) * MILLION)
#define SEC_TO_NSEC(x) ((x) * BILLION)

typedef uint64_t timestamp;

static inline timestamp
get_timestamp(void)
{
   struct timespec ts;
   clock_gettime(CLOCK_MONOTONIC, &ts);
   return SEC_TO_NSEC(ts.tv_sec) + ts.tv_nsec;
}

static inline timestamp
timestamp_elapsed(timestamp tv)
{
   struct timespec ts;
   clock_gettime(CLOCK_MONOTONIC, &ts);
   return SEC_TO_NSEC(ts.tv_sec) + ts.tv_nsec - tv;
}

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

   for (uint64_t i = start; i < start + n; ++i)
      if (!iceberg_insert(table, keys[i], values[i], id)) {
         printf("Failed insert\n");
         exit(0);
      }
}

void
do_queries(uint64_t *keys, uint64_t start, uint64_t n, bool positive)
{

   uint64_t *val;
   for (uint64_t i = start; i < start + n; ++i)
      if (iceberg_get_value(table, keys[i], &val) != positive) {

         if (positive)
            printf("False negative query\n");
         else
            printf("False positive query\n");
         exit(0);
      }
}

void
do_removals(uint8_t id, uint64_t *keys, uint64_t start, uint64_t n)
{

   // uint64_t val;

   for (uint64_t i = start; i < start + n; ++i)
      if (!iceberg_remove(table, keys[i], id)) {
         printf("Failed removal\n");
         exit(0);
      }
}

void
safe_rand_bytes(unsigned char *v, size_t n)
{
   for (uint64_t i = 0; i < n; i++) {
      v[i] = rand();
   }
   // while (n > 0) {
   //	size_t round_size = n >= INT_MAX ? INT_MAX - 1 : n;
   //	RAND_bytes(v, round_size);
   //	v += round_size;
   //	n -= round_size;
   // }
}

void
do_mixed(uint8_t                                     id,
         std::vector<std::pair<uint64_t, uint64_t>> &v,
         uint64_t                                    start,
         uint64_t                                    n)
{

   uint64_t *val;
   for (uint64_t i = start; i < start + n; ++i)
      if (iceberg_get_value(table, v[i].first, &val)) {

         iceberg_remove(table, v[i].first, id);
      } else
         iceberg_insert(table, v[i].first, v[i].second, id);
}

int
main(int argc, char **argv)
{
   if (argc < 4) {
      fprintf(stderr,
              "Specify the log of the number of slots in the table and the "
              "number of threads to use.\n");
      exit(1);
   }


   uint64_t tbits     = atoi(argv[1]);
   uint64_t inittbits = tbits - atoi(argv[2]);
   uint64_t threads = atoi(argv[3]);
   uint64_t N       = (1ULL << tbits) * 1.05;

   bool is_benchmark = false;
   bool use_hugepages = false;
   for (uint64_t arg_i = 4; arg_i < argc; arg_i++) {
      if (strcmp(argv[arg_i], "-b") == 0) {
         is_benchmark = true;
      } else if (strcmp(argv[arg_i], "-h") == 0) {
         use_hugepages = true;
      } else {
         fprintf(stderr,
                 "Specify the log of the number of slots in the table and the "
                 "number of threads to use.\n");
         exit(1);
      }
   }

   timestamp creation_start = get_timestamp();

   table = (iceberg_table *)malloc(sizeof(iceberg_table));
   assert(table);

   iceberg_init(table, inittbits, tbits, use_hugepages);

   uint64_t creation_time = timestamp_elapsed(creation_start);
   if (!is_benchmark) {
      printf("Creation time: %luus\n", NSEC_TO_USEC(creation_time));
   }

   srand(100);
   // srand(time(NULL));

   // Generating vectors of size N for data contained and not contained in the
   // table
   //
   uint64_t size = N / threads;

   N = N / size * size;

   if (!is_benchmark) {
      printf("%ld\n", N * 2 * sizeof(uint64_t));
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

   timestamp insert_start = get_timestamp();

   std::vector<std::thread> thread_list;
   for (uint64_t j = 0; j < threads; j++) {
      thread_list.emplace_back(
         do_inserts, j, in_keys, in_values, j * size, size);
   }
   for (uint64_t j = 0; j < threads; j++) {
      thread_list[j].join();
   }
   thread_list.clear();

   uint64_t insert_time_ns = timestamp_elapsed(insert_start);
   double insert_throughput_mil = N / (double)NSEC_TO_USEC(insert_time_ns);

   if (!is_benchmark) {
      printf("Time elapsed: %luus\n", NSEC_TO_USEC(insert_time_ns));
      printf("Insertions: %fM/sec\n", insert_throughput_mil);

      printf("Load factor: %f\n", iceberg_load_factor(table));
      printf("Number level 1 inserts: %ld\n", lv1_balls(table));
      printf("Number level 2 inserts: %ld\n", lv2_balls(table));
      printf("Number level 3 inserts: %ld\n", lv3_balls(table));
      printf("Total inserts: %ld\n", tot_balls(table));
   }

   //	exit(0);

   uint64_t max_size = 0, sum_sizes = 0;
   uint64_t nblocks = 1 << table->metadata->log_nblocks;
   for (uint64_t i = 0; i < nblocks; ++i) {
      max_size = std::max(max_size, table->metadata->lv3_sizes[i]);
      sum_sizes += table->metadata->lv3_sizes[i];
   }

   if (!is_benchmark) {
      printf("Average list size: %f\n", sum_sizes / (double)nblocks);
      printf("Max list size: %ld\n", max_size);

      printf("\nQUERIES\n");
   }

   std::mt19937 g(__builtin_ia32_rdtsc());
   std::shuffle(&in_keys[0], &in_keys[N], g);

   //	exit(0);

   timestamp negative_start = get_timestamp();

   for (uint64_t i = 0; i < threads; ++i)
      thread_list.emplace_back(
         do_queries, out_keys, i * (N / threads), N / threads, false);
   for (uint64_t i = 0; i < threads; ++i)
      thread_list[i].join();

   uint64_t negative_time_ns = timestamp_elapsed(negative_start);
   double negative_throughput_mil = N / (double)NSEC_TO_SEC(negative_time_ns);
   if (!is_benchmark) {
      printf("Negative queries: %fM/sec\n", negative_throughput_mil);
   }
   thread_list.clear();

   //	exit(0);

   timestamp positive_start = get_timestamp();

   for (uint64_t i = 0; i < threads; ++i) {
      thread_list.emplace_back(
         do_queries, in_keys, i * (N / threads), N / threads, true);
   }
   for (uint64_t i = 0; i < threads; ++i) {
      thread_list[i].join();
   }
   thread_list.clear();

   uint64_t positive_time_ns = timestamp_elapsed(positive_start);
   double positive_throughput_mil = N / (double)NSEC_TO_SEC(positive_time_ns);
   if (!is_benchmark) {
      printf("Positive queries: %fM/sec\n", positive_throughput_mil);
   }

   //	exit(0);

   if (!is_benchmark) {
      printf("\nREMOVALS\n");
   }

   uint64_t  num_removed = N / 2 / threads * threads;
   uint64_t *removed     = in_keys;
   uint64_t *non_removed = in_keys + num_removed;

   shuffle(&removed[0], &removed[num_removed], g);
   shuffle(&non_removed[0], &non_removed[N - num_removed], g);

   timestamp removal_start = get_timestamp();
   for (uint64_t i = 0; i < threads; ++i) {
      thread_list.emplace_back(
         do_removals, i, removed, i * (N / 2 / threads), N / 2 / threads);
   }
   for (uint64_t i = 0; i < threads; ++i) {
      thread_list[i].join();
   }
   thread_list.clear();

   uint64_t removal_time_ns = timestamp_elapsed(removal_start);
   double removal_throughput_mil = N / (double)NSEC_TO_SEC(removal_time_ns);
   if (!is_benchmark) {
      printf("Removals: %fM/sec\n", removal_throughput_mil);
      printf("Load factor: %f\n", iceberg_load_factor(table));
   }

   shuffle(&removed[0], &removed[num_removed], g);

   if (is_benchmark) {
      printf("%f %f %f %f\n",
             insert_throughput_mil,
             negative_throughput_mil,
             positive_throughput_mil,
             removal_throughput_mil);
      return 0;
   }

   timestamp negative_after_removal_start = get_timestamp();

   for (uint64_t i = 0; i < threads; ++i) {
      thread_list.emplace_back(
         do_queries, removed, i * (N / 2 / threads), N / 2 / threads, false);
   }
   for (uint64_t i = 0; i < threads; ++i) {
      thread_list[i].join();
   }
   thread_list.clear();

   uint64_t negative_after_removal_time_ns = timestamp_elapsed(negative_after_removal_start);
   double negative_after_removal_throughput_mil = N / (double)NSEC_TO_SEC(negative_after_removal_time_ns);
   if (!is_benchmark) {
      printf("Negative queries after removals: %fM/sec\n", negative_after_removal_throughput_mil);
      printf("Load factor: %f\n", iceberg_load_factor(table));
   }

   timestamp positive_after_removal_start = get_timestamp();

   for (uint64_t i = 0; i < threads; ++i) {
      thread_list.emplace_back(
         do_queries, non_removed, i * (N / 2 / threads), N / 2 / threads, true);
   }
   for (uint64_t i = 0; i < threads; ++i) {
      thread_list[i].join();
   }
   thread_list.clear();

   uint64_t positive_after_removal_time_ns = timestamp_elapsed(positive_after_removal_start);
   double positive_after_removal_throughput_mil = N / (double)NSEC_TO_SEC(positive_after_removal_time_ns);
   printf("Positive queries after removals: %fM/sec\n",
          positive_after_removal_throughput_mil);

   /*
           printf("\nMIXED WORKLOAD, HIGH LOAD FACTOR\n");

           for(uint64_t i = 0; i < N; ++i) in_table.push_back(not_in_table[i]);

           shuffle(in_table.begin(), in_table.end(), g);

           t1 = high_resolution_clock::now();

           for(uint64_t i = 0; i < threads; ++i)
                   thread_list.emplace_back(do_mixed, i, std::ref(in_table), i *
      2 * N / threads, 2 * N / threads); for(uint64_t i = 0; i < threads; ++i)
                   thread_list[i].join();

           t2 = high_resolution_clock::now();
           printf("Mixed operations at high load factor: %f /sec\n", 4 * N /
      elapsed(t1, t2)); thread_list.clear();

           max_size = sum_sizes = 0;
           for(uint64_t i = 0; i < table->metadata->nblocks; ++i) {
                   max_size = std::max(max_size, table->metadata->lv3_sizes[i]);
                   sum_sizes += table->metadata->lv3_sizes[i];
           }

           printf("Average list size: %f\n", sum_sizes /
      (double)table->metadata->nblocks); printf("Max list size: %ld\n",
      max_size);*/
}
