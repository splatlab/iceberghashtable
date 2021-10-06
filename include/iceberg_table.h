#ifndef _POTC_TABLE_H_
#define _POTC_TABLE_H_

#include "partitioned_counter.h"
#include <inttypes.h>
#include <stdbool.h>

#ifdef __cplusplus
#   define __restrict__
extern "C" {
#endif

#define SLOT_BITS       6
#define FPRINT_BITS     8
#define LV2_SLOTS       8
#define MAX_GENERATIONS 8
#define CACHE_LINE_SIZE 64

typedef uint64_t KeyType;
typedef uint64_t ValueType;

typedef struct kv_pair {
   KeyType   key;
   ValueType val;
} kv_pair;

typedef struct __attribute__((__packed__)) iceberg_lv1_block {
   kv_pair slots[1 << SLOT_BITS];
} iceberg_lv1_block;

typedef struct __attribute__((__packed__)) iceberg_lv1_block_md {
   uint8_t block_md[1 << SLOT_BITS];
} iceberg_lv1_block_md;

typedef struct __attribute__((__packed__)) iceberg_lv2_block {
   kv_pair slots[LV2_SLOTS];
} iceberg_lv2_block;

typedef struct __attribute__((__packed__)) iceberg_lv2_block_md {
   uint8_t block_md[LV2_SLOTS];
} iceberg_lv2_block_md;

typedef struct iceberg_lv3_node {
   KeyType                  key;
   ValueType                val;
   struct iceberg_lv3_node *next_node;
} iceberg_lv3_node;

typedef struct iceberg_lv3_list {
   iceberg_lv3_node *head;
} iceberg_lv3_list;

typedef struct iceberg_metadata {
   uint64_t              total_size_in_bytes;
   uint8_t               initial_log_nblocks;
   uint8_t               log_nblocks;
   uint32_t              resize_lock;
   uint8_t               lv2_log_nblocks;
   uint8_t               lv3_log_nblocks;
   uint64_t              nslots;
   pc_t                 *lv1_balls;
   pc_t                 *lv2_balls;
   pc_t                 *lv3_balls;
   iceberg_lv1_block_md *lv1_md[MAX_GENERATIONS];
   iceberg_lv2_block_md *lv2_md;
   uint64_t             *lv3_sizes;
   uint8_t              *lv3_locks;
   int                   mmap_flags;
} iceberg_metadata;

typedef struct iceberg_table {
   iceberg_metadata  *metadata;
   iceberg_lv1_block *level1[MAX_GENERATIONS];
   iceberg_lv2_block *level2;
   iceberg_lv3_list  *level3;
} iceberg_table;

uint64_t
lv1_balls(iceberg_table *table);
uint64_t
lv2_balls(iceberg_table *table);
uint64_t
lv3_balls(iceberg_table *table);
uint64_t
tot_balls(iceberg_table *table);

void
iceberg_init(iceberg_table *table,
             uint64_t       log_slots,
             uint64_t       final_log_slots,
             bool           use_hugepages);

double
iceberg_load_factor(iceberg_table *table);

bool
iceberg_insert(iceberg_table *table,
               KeyType        key,
               ValueType      value,
               uint8_t        thread_id);

bool
iceberg_remove(iceberg_table *table, KeyType key, uint8_t thread_id);

bool
iceberg_get_value(iceberg_table *table, KeyType key, ValueType **value);

#ifdef __cplusplus
}
#endif

#endif // _POTC_TABLE_H_
