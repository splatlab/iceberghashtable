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

#define SLOT_BITS 6
#define FPRINT_BITS 8
#define D_CHOICES 2
#define MAX_LG_LG_N 4
#define C_LV2 6
#define MAX_RESIZES 8
#define LEVEL3_BLOCKS 1024

  typedef uint64_t KeyType;
  typedef uint64_t ValueType;

  typedef struct __attribute__ ((__packed__)) kv_pair {
    KeyType key;
    ValueType val;
  } kv_pair;

  typedef struct __attribute__ ((__packed__)) iceberg_lv1_block {
    kv_pair slots[1 << SLOT_BITS];
  } iceberg_lv1_block;

  typedef struct __attribute__ ((__packed__)) iceberg_lv1_block_md {
    uint8_t block_md[1 << SLOT_BITS];
  } iceberg_lv1_block_md;

  typedef struct __attribute__ ((__packed__)) iceberg_lv2_block {
    kv_pair slots[C_LV2 + MAX_LG_LG_N / D_CHOICES];
  } iceberg_lv2_block;

  typedef struct __attribute__ ((__packed__)) iceberg_lv2_block_md {
    uint8_t block_md[C_LV2 + MAX_LG_LG_N / D_CHOICES];
  } iceberg_lv2_block_md;

  typedef struct iceberg_lv3_node {
    KeyType key;
    ValueType val;
#ifdef PMEM
    bool in_use;
    ptrdiff_t next_idx;
#else
    struct iceberg_lv3_node * next_node;
#endif
  } iceberg_lv3_node;

  typedef struct iceberg_lv3_list {
#ifdef PMEM
    ptrdiff_t head_idx;
#else
    iceberg_lv3_node * head;
#endif
  } iceberg_lv3_list;

  typedef struct iceberg_metadata {
    uint64_t total_size_in_bytes;
    uint64_t nblocks;
    uint64_t nslots;
    uint64_t block_bits;
    uint64_t init_size;
    uint64_t log_init_size;
    uint64_t resize_threshold;
    int64_t lv1_ctr;
    int64_t lv2_ctr;
    int64_t lv3_ctr;
    pc_t lv1_balls;
    pc_t lv2_balls;
    pc_t lv3_balls;
    iceberg_lv1_block_md * lv1_md[MAX_RESIZES];
    iceberg_lv2_block_md * lv2_md[MAX_RESIZES];
    uint64_t * lv3_sizes;
    uint8_t * lv3_locks;
    uint64_t nblocks_parts[MAX_RESIZES];
#ifdef ENABLE_RESIZE
    volatile int lock;
    uint64_t resize_cnt;
    uint64_t marker_sizes[MAX_RESIZES];
    uint64_t lv1_resize_ctr;
    uint64_t lv2_resize_ctr;
    uint8_t * lv1_resize_marker[MAX_RESIZES];
    uint8_t * lv2_resize_marker[MAX_RESIZES];
#endif
  } iceberg_metadata;

  typedef struct iceberg_table {
    iceberg_metadata metadata;
    /* Only things that are persisted on PMEM */
    iceberg_lv1_block *level1[MAX_RESIZES];
    iceberg_lv2_block *level2[MAX_RESIZES];
    iceberg_lv3_list *level3;
#ifdef PMEM
    iceberg_lv3_node * level3_nodes;
#endif
  } iceberg_table;

  uint64_t lv1_balls(iceberg_table * table);
  uint64_t lv2_balls(iceberg_table * table);
  uint64_t lv3_balls(iceberg_table * table);
  uint64_t tot_balls(iceberg_table * table);

  int iceberg_init(iceberg_table *table, uint64_t log_slots);

  double iceberg_load_factor(iceberg_table * table);

  bool iceberg_insert(iceberg_table * table, KeyType key, ValueType value, uint8_t thread_id);

  bool iceberg_remove(iceberg_table * table, KeyType key, uint8_t thread_id);

  bool iceberg_get_value(iceberg_table * table, KeyType key, ValueType *value, uint8_t thread_id);

#ifdef PMEM
  uint64_t iceberg_dismount(iceberg_table *table);
  int iceberg_mount(iceberg_table *table, uint64_t log_slots, uint64_t resize_cnt);
#endif

#ifdef ENABLE_RESIZE
  void iceberg_end(iceberg_table * table);
#endif

#ifdef __cplusplus
}
#endif

#endif	// _POTC_TABLE_H_

