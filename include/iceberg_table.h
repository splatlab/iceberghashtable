#ifndef _POTC_TABLE_H_
#define _POTC_TABLE_H_

#include <inttypes.h>
#include <stdbool.h>
#include "lock.h"

#ifdef __cplusplus
#define __restrict__
extern "C" {
#endif

#if 1
#define ENABLE_RESIZE 1
#else
#undef ENABLE_RESIZE
#endif

#define SLOT_BITS 6
#define FPRINT_BITS 8
#define D_CHOICES 2
#define MAX_LG_LG_N 4
#define C_LV2 6

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
    struct iceberg_lv3_node * next_node;
  } iceberg_lv3_node;

  typedef struct iceberg_lv3_list {
    iceberg_lv3_node * head;
  } iceberg_lv3_list;

  typedef struct iceberg_metadata {
    uint64_t total_size_in_bytes;
    uint64_t nblocks;
    uint64_t nslots;
    uint64_t block_bits;
    int64_t lv1_ctr;
    int64_t lv2_ctr;
    int64_t lv3_ctr;
    pc_t lv1_balls;
    pc_t lv2_balls;
    pc_t lv3_balls;
    iceberg_lv1_block_md * lv1_md;
    iceberg_lv2_block_md * lv2_md;
    uint64_t * lv3_sizes;
    uint8_t * lv3_locks;
#ifdef ENABLE_RESIZE
    ReaderWriterLock rw_lock;
    uint64_t lv1_resize_ctr;
    uint64_t lv2_resize_ctr;
    uint64_t lv3_resize_ctr;
    uint8_t * lv1_resize_marker;
    uint8_t * lv2_resize_marker;
    uint8_t * lv3_resize_marker;
#endif
  } iceberg_metadata;

  typedef struct iceberg_table {
    iceberg_metadata metadata;
    iceberg_lv1_block * level1;
    iceberg_lv2_block * level2;
    iceberg_lv3_list * level3;
  } iceberg_table;

  uint64_t lv1_balls(iceberg_table * table);
  uint64_t lv2_balls(iceberg_table * table);
  uint64_t lv3_balls(iceberg_table * table);
  uint64_t tot_balls(iceberg_table * table);

  int iceberg_init(iceberg_table *table, uint64_t log_slots);

  double iceberg_load_factor(iceberg_table * table);

  typedef enum {
    LEVEL1,
    LEVEL1_RESIZE,
    LEVEL2,
    LEVEL2_RESIZE,
    LEVEL3,
    LEVEL3_RESIZE,
    FAILED,
  } iceberg_insert_rc;
  iceberg_insert_rc iceberg_insert(iceberg_table * table, KeyType key, ValueType value, uint8_t thread_id);

  bool iceberg_remove(iceberg_table * table, KeyType key, uint8_t thread_id);

  typedef enum {
    Q_LEVEL1,
    Q_LEVEL1_OLD,
    Q_LEVEL21,
    Q_LEVEL21_OLD,
    Q_LEVEL22,
    Q_LEVEL22_OLD,
    Q_LEVEL3,
    Q_LEVEL3_OLD,
    Q_NOT_FOUND,
  } iceberg_query_rc;
  iceberg_query_rc iceberg_get_value(iceberg_table * table, KeyType key, ValueType **value, uint8_t thread_id);

#ifdef ENABLE_RESIZE
  void iceberg_end(iceberg_table * table);
#endif

#ifdef __cplusplus
}
#endif

#endif	// _POTC_TABLE_H_

