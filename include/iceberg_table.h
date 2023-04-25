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

typedef struct iceberg_lv3_node {
  iceberg_key_t            key;
  iceberg_value_t          val;
  struct iceberg_lv3_node *next_node;
} iceberg_lv3_node;

typedef struct iceberg_lv3_list {
  iceberg_lv3_node *head;
  volatile bool     lock;
} iceberg_lv3_list;

typedef struct iceberg_table {
  // Level 1
  kv_pair       *level1[MAX_PARTITIONS];
  fingerprint_t *level1_sketch[MAX_PARTITIONS];

  // Level2
  kv_pair       *level2[MAX_PARTITIONS];
  fingerprint_t *level2_sketch[MAX_PARTITIONS];

  // Level 3
  iceberg_lv3_list level3[LEVEL3_BLOCKS];

  // Metadata
  uint64_t nblocks;
  uint64_t log_num_blocks;
  uint64_t log_initial_num_blocks;
  counter  num_items_per_level;

#ifdef ENABLE_RESIZE
  volatile bool lock;
  uint64_t      resize_threshold;
  uint64_t      num_partitions;
  uint64_t      lv1_resize_ctr;
  uint64_t      lv2_resize_ctr;
  uint64_t      marker_sizes[MAX_PARTITIONS];
  uint8_t      *lv1_resize_marker[MAX_PARTITIONS];
  uint8_t      *lv2_resize_marker[MAX_PARTITIONS];
#endif
} iceberg_table;

uint64_t lv1_balls(iceberg_table *table);
uint64_t lv2_balls(iceberg_table *table);
uint64_t lv3_balls(iceberg_table *table);
uint64_t tot_balls(iceberg_table *table);

void iceberg_init(iceberg_table *table, uint64_t log_slots);

double iceberg_load_factor(iceberg_table *table);

bool iceberg_insert(iceberg_table  *table,
                    iceberg_key_t   key,
                    iceberg_value_t value,
                    uint64_t        tid);

bool iceberg_remove(iceberg_table *table, iceberg_key_t key, uint64_t tid);

bool iceberg_get_value(iceberg_table   *table,
                       iceberg_key_t    key,
                       iceberg_value_t *value,
                       uint64_t         tid);

#ifdef ENABLE_RESIZE
void iceberg_end(iceberg_table *table);
#endif

#ifdef __cplusplus
}
#endif

#endif // _POTC_TABLE_H_
