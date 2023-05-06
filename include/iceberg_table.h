#ifndef _POTC_TABLE_H_
#define _POTC_TABLE_H_

#include "public_counter.h"
#include <inttypes.h>
#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
#  define __restrict__
extern "C" {
#endif

#define LEVEL3_BLOCKS  1024ULL
#define MAX_PARTITIONS 8ULL

typedef uint64_t iceberg_key_t;
typedef uint64_t iceberg_value_t;
typedef uint8_t  fingerprint_t;
typedef struct kv_pair {
  iceberg_key_t   key;
  iceberg_value_t val;
} kv_pair;

_Static_assert(sizeof(kv_pair) == 16,
               "kv_pair needs to be 16B for atomic loads and stores");

typedef struct iceberg_level3_node {
  iceberg_key_t               key;
  iceberg_value_t             val;
  struct iceberg_level3_node *next_node;
} iceberg_level3_node;

typedef struct iceberg_level3_list {
  iceberg_level3_node *head;
  volatile bool        lock;
} iceberg_level3_list;

typedef struct iceberg_table {
  // Level 1
  kv_pair       *level1[MAX_PARTITIONS];
  fingerprint_t *level1_sketch[MAX_PARTITIONS];

  // Level 2
  kv_pair       *level2[MAX_PARTITIONS];
  fingerprint_t *level2_sketch[MAX_PARTITIONS];

  // Level 3
  iceberg_level3_list level3[LEVEL3_BLOCKS];

  // Metadata
  uint64_t num_blocks;
  uint64_t log_num_blocks;
  uint64_t log_initial_num_blocks;
  counter  num_items_per_level;

#ifdef ENABLE_RESIZE
  volatile bool lock;
  uint64_t      resize_threshold;
  uint64_t      max_partition_num;
  uint64_t      level1_resize_counter;
  uint64_t      level2_resize_counter;
  uint8_t      *level1_resize_marker;
  uint8_t      *level2_resize_marker;
#endif
} iceberg_table;

uint64_t level1_load(iceberg_table *table);
uint64_t level2_load(iceberg_table *table);
uint64_t level3_load(iceberg_table *table);
uint64_t iceberg_load(iceberg_table *table);

uint64_t iceberg_capacity(iceberg_table *table);

void iceberg_init(iceberg_table *table, uint64_t log_slots);

double iceberg_load_factor(iceberg_table *table);

bool iceberg_insert(iceberg_table  *table,
                    iceberg_key_t   key,
                    iceberg_value_t value,
                    uint64_t        tid);

bool iceberg_delete(iceberg_table *table, iceberg_key_t key, uint64_t tid);

bool iceberg_query(iceberg_table   *table,
                   iceberg_key_t    key,
                   iceberg_value_t *value,
                   uint64_t         tid);

#ifdef ENABLE_RESIZE
void iceberg_end(iceberg_table *table, uint64_t tid);
#endif

#ifdef __cplusplus
}
#endif

#endif // _POTC_TABLE_H_
