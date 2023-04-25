#ifndef _POTC_TABLE_H_
#define _POTC_TABLE_H_

#include <inttypes.h>
#include <stdbool.h>
#include <stddef.h>
#include "lock.h"

#ifdef __cplusplus
#define __restrict__
extern "C" {
#endif

#define LEVEL3_BLOCKS 1024ULL
#define MAX_PARTITIONS 8ULL

  typedef uint64_t iceberg_key_t;
  typedef uint64_t iceberg_value_t;
  typedef uint8_t fingerprint_t;

  typedef struct kv_pair {
    iceberg_key_t key;
    iceberg_value_t val;
  } kv_pair;

  _Static_assert(sizeof(kv_pair) == 16, "kv_pair needs to be 16B for atomic loads and stores");

  typedef struct iceberg_lv3_node {
    iceberg_key_t key;
    iceberg_value_t val;
    struct iceberg_lv3_node * next_node;
  } iceberg_lv3_node;

  typedef struct iceberg_lv3_list {
    iceberg_lv3_node * head;
  } iceberg_lv3_list;

  typedef struct iceberg_metadata {
    uint64_t nblocks;
    uint64_t log_num_blocks;
    uint64_t log_initial_num_blocks;

    int64_t lv1_ctr;
    int64_t lv2_ctr;
    int64_t lv3_ctr;
    pc_t lv1_balls;
    pc_t lv2_balls;
    pc_t lv3_balls;
    uint64_t *lv3_sizes;
    uint8_t *lv3_locks;
#ifdef ENABLE_RESIZE
    volatile int lock;
    uint64_t resize_threshold;
    uint64_t num_partitions;
    uint64_t marker_sizes[MAX_PARTITIONS];
    uint64_t lv1_resize_ctr;
    uint64_t lv2_resize_ctr;
    uint8_t *lv1_resize_marker[MAX_PARTITIONS];
    uint8_t *lv2_resize_marker[MAX_PARTITIONS];
#endif
  } iceberg_metadata;

  typedef struct iceberg_table {
    iceberg_metadata metadata;
    fingerprint_t *level1_sketch[MAX_PARTITIONS];
    fingerprint_t *level2_sketch[MAX_PARTITIONS];
    kv_pair *level1[MAX_PARTITIONS];
    kv_pair *level2[MAX_PARTITIONS];
    iceberg_lv3_list level3[LEVEL3_BLOCKS];
  } iceberg_table;

  uint64_t lv1_balls(iceberg_table * table);
  uint64_t lv2_balls(iceberg_table * table);
  uint64_t lv3_balls(iceberg_table * table);
  uint64_t tot_balls(iceberg_table * table);

  int iceberg_init(iceberg_table *table, uint64_t log_slots);

  double iceberg_load_factor(iceberg_table * table);

  bool iceberg_insert(iceberg_table * table, iceberg_key_t key, iceberg_value_t value, uint8_t thread_id);

  bool iceberg_remove(iceberg_table * table, iceberg_key_t key, uint8_t thread_id);

  bool iceberg_get_value(iceberg_table * table, iceberg_key_t key, iceberg_value_t *value, uint8_t thread_id);

#ifdef ENABLE_RESIZE
  void iceberg_end(iceberg_table * table);
#endif

#ifdef __cplusplus
}
#endif

#endif	// _POTC_TABLE_H_

