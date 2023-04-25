#define _GNU_SOURCE
#include <assert.h>
#include <immintrin.h>
#include <math.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <tmmintrin.h>

#include "counter.h"
#include "iceberg_precompute.h"
#include "iceberg_table.h"
#include "lock.h"
#include "verbose.h"
#include "xxhash.h"

#define RESIZE_THRESHOLD 0.96
/*#define RESIZE_THRESHOLD 0.85 // For YCSB*/
#define MAX_PROCS             64
#define LEVEL1_BLOCK_SIZE     64ULL
#define LEVEL1_LOG_BLOCK_SIZE 6ULL
#define LEVEL2_BLOCK_SIZE     8ULL

_Static_assert((1 << LEVEL1_LOG_BLOCK_SIZE) == LEVEL1_BLOCK_SIZE,
               "Level1 block width inconsistent with level1 block size");

uint64_t seed = 12351327692179052ll;

static inline uint8_t
word_select(uint64_t val, int rank)
{
  val = _pdep_u64(one[rank], val);
  return _tzcnt_u64(val);
}

typedef enum {
  LEVEL1 = 0,
  LEVEL2,
  LEVEL3,
} level_type;

uint64_t
lv1_balls(iceberg_table *table)
{
  counter *cntr = &table->num_items_per_level;
  counter_sync(cntr, LEVEL1);
  return counter_get(cntr, LEVEL1);
}

static inline uint64_t
lv1_balls_aprox(iceberg_table *table)
{
  counter *cntr = &table->num_items_per_level;
  return counter_get(cntr, LEVEL1);
}

uint64_t
lv2_balls(iceberg_table *table)
{
  counter *cntr = &table->num_items_per_level;
  counter_sync(cntr, LEVEL2);
  return counter_get(cntr, LEVEL2);
}

static inline uint64_t
lv2_balls_aprox(iceberg_table *table)
{
  counter *cntr = &table->num_items_per_level;
  return counter_get(cntr, LEVEL2);
}

uint64_t
lv3_balls(iceberg_table *table)
{
  counter *cntr = &table->num_items_per_level;
  counter_sync(cntr, LEVEL3);
  return counter_get(cntr, LEVEL3);
}

static inline uint64_t
lv3_balls_aprox(iceberg_table *table)
{
  counter *cntr = &table->num_items_per_level;
  return counter_get(cntr, LEVEL3);
}

uint64_t
tot_balls(iceberg_table *table)
{
  return lv1_balls(table) + lv2_balls(table) + lv3_balls(table);
}

uint64_t
tot_balls_aprox(iceberg_table *table)
{
  return lv1_balls_aprox(table) + lv2_balls_aprox(table) +
         lv3_balls_aprox(table);
}

static inline uint64_t
total_capacity(iceberg_table *table)
{
  return table->nblocks * (LEVEL1_BLOCK_SIZE + LEVEL2_BLOCK_SIZE);
}

inline double
iceberg_load_factor(iceberg_table *table)
{
  return (double)tot_balls(table) / (double)total_capacity(table);
}

#ifdef ENABLE_RESIZE
static inline bool
need_resize(iceberg_table *table)
{
  return tot_balls_aprox(table) >= table->resize_threshold;
}
#endif

typedef struct __attribute__((packed)) {
  uint64_t level1_raw_block : 40;
  uint64_t level2_raw_block1 : 40;
  uint64_t level2_raw_block2 : 40;
  uint8_t  fingerprint : 8;
} raw_hash;

_Static_assert(sizeof(raw_hash) == 16, "hash not 16B\n");

typedef enum {
  LEVEL1_BLOCK = 0,
  LEVEL2_BLOCK1,
  LEVEL2_BLOCK2,
  NUM_LEVELS,
} block_type;

typedef struct {
  raw_hash raw;
  uint64_t raw_block[NUM_LEVELS];
  uint8_t  fingerprint;
} hash;

static inline uint8_t
ensure_nonzero_fingerprint(raw_hash *h)
{
  return (h->fingerprint <= 1 ? h->fingerprint | 2 : h->fingerprint);
}

static inline uint64_t
truncate_to_current_num_raw_blocks(iceberg_table *table, uint64_t raw_block)
{
  return raw_block & ((1 << table->log_num_blocks) - 1);
}

static inline hash
hash_key(iceberg_table *table, iceberg_value_t *key)
{
  hash           h   = {0};
  XXH128_hash_t *raw = (XXH128_hash_t *)&h.raw;
  *raw               = XXH128(key, sizeof(*key), seed);
  h.fingerprint      = ensure_nonzero_fingerprint(&h.raw);
  h.raw_block[LEVEL1_BLOCK] =
    truncate_to_current_num_raw_blocks(table, h.raw.level1_raw_block);
  h.raw_block[LEVEL2_BLOCK1] =
    truncate_to_current_num_raw_blocks(table, h.raw.level2_raw_block1);
  h.raw_block[LEVEL2_BLOCK2] =
    truncate_to_current_num_raw_blocks(table, h.raw.level2_raw_block2);
  verbose_print_hash(h.raw_block[LEVEL1_BLOCK],
                     h.raw_block[LEVEL2_BLOCK1],
                     h.raw_block[LEVEL2_BLOCK2],
                     h.fingerprint);
  return h;
}

typedef struct {
  uint64_t partition;
  uint64_t block;
} partition_block;

static inline partition_block
decode_raw_internal(uint64_t init_log, uint64_t raw_block)
{
  partition_block pb  = {0};
  uint64_t        shf = raw_block >> init_log;
  pb.partition        = 64 - _lzcnt_u64(shf);
  uint64_t adj        = 1ULL << pb.partition;
  adj                 = adj >> 1;
  adj                 = adj << init_log;
  pb.block            = raw_block - adj;
  return pb;
}

static inline partition_block
decode_raw_block(iceberg_table *table, uint64_t raw_block)
{
  return decode_raw_internal(table->log_initial_num_blocks, raw_block);
}

static inline partition_block
get_block(iceberg_table *table, hash *h, block_type lvl)
{
  return decode_raw_block(table, h->raw_block[lvl]);
}

static inline uint64_t
get_level3_block(hash *h)
{
  return h->raw_block[LEVEL1_BLOCK] % LEVEL3_BLOCKS;
}

static inline partition_block
decode_raw_chunk(iceberg_table *table, uint64_t raw_chunk)
{
  return decode_raw_internal(table->log_initial_num_blocks - 3, raw_chunk);
}

#define LOCK_MASK   1
#define UNLOCK_MASK ~1

static inline void
lock_block(fingerprint_t *sketch)
{
  fingerprint_t *lock_fp = &sketch[63];
  while ((__atomic_fetch_or(lock_fp, LOCK_MASK, __ATOMIC_SEQ_CST) &
          LOCK_MASK) != 0) {
    _mm_pause();
  }
}

static inline void
unlock_block(fingerprint_t *sketch)
{
  fingerprint_t *lock_fp = &sketch[63];
  *lock_fp               = *lock_fp & UNLOCK_MASK;
}

static inline uint32_t
slot_mask_32(uint8_t *sketch, uint8_t fprint)
{
  __m256i bcast = _mm256_set1_epi8(fprint);
  __m256i block = _mm256_maskz_loadu_epi64(1, (const __m256i *)(sketch));
#if defined __AVX512BW__ && defined __AVX512VL__
  return _mm256_mask_cmp_epi8_mask(0xff, bcast, block, _MM_CMPINT_EQ);
#else
  __m256i cmp = _mm256_cmpeq_epi8(bcast, block);
  return _mm256_movemask_epi8(cmp);
#endif
}

#if defined __AVX512F__ && defined __AVX512BW__
static inline uint64_t
slot_mask_64(fingerprint_t *sketch, fingerprint_t fp)
{
  __m512i mask  = _mm512_load_si512((const __m512i *)(broadcast_mask));
  __m512i bcast = _mm512_set1_epi8(fp);
  bcast         = _mm512_or_epi64(bcast, mask);
  __m512i block = _mm512_load_si512((const __m512i *)(sketch));
  block         = _mm512_or_epi64(block, mask);
  return _mm512_cmp_epi8_mask(bcast, block, _MM_CMPINT_EQ);
}
#else  /* ! (defined __AVX512F__ && defined __AVX512BW__) */
static inline uint32_t
slot_mask_64_half(__m256i fprint, __m256i md, __m256i mask)
{
  __m256i masked_fp = _mm256_or_si256(fprint, mask);
  __m256i masked_md = _mm256_or_si256(md, mask);
  __m256i cmp = _mm256_cmpeq_epi8(masked_md, masked_fp);
  return _mm256_movemask_epi8(cmp);
}

static inline uint64_t
slot_mask_64(fingerprint_t *sketch, fingerprint_t fp)
{
  __m256i fprint = _mm256_set1_epi8(fp);

  __m256i md1 = _mm256_load_si256((const __m256i *)(sketch));
  __m256i mask1 = _mm256_load_si256((const __m256i *)(broadcast_mask));
  uint64_t result1 = slot_mask_64_half(fprint, md1, mask1);

  __m256i md2 = _mm256_load_si256((const __m256i *)(&sketch[32]));
  __m256i mask2 = _mm256_load_si256((const __m256i *)(&broadcast_mask[32]));
  uint64_t result2 = slot_mask_64_half(fprint, md2, mask2);

  return ((uint64_t)result2 << 32) | result1;
}
#endif /* ! (defined __AVX512F__ && defined __AVX512BW__) */


static inline void
atomic_write_128(uint64_t key, uint64_t val, kv_pair *kv)
{
  uint64_t arr[2] = {key, val};
  __m128d  a      = _mm_load_pd((double *)arr);
  _mm_store_pd((double *)kv, a);
}

static inline uint64_t
kv_pair_offset(partition_block pb, uint64_t block_size, uint64_t slot_in_block)
{
  return block_size * pb.block + slot_in_block;
}

static inline kv_pair *
get_level1_kv_pair(iceberg_table  *table,
                   partition_block pb,
                   uint64_t        slot_in_block)
{
  uint64_t slot = kv_pair_offset(pb, LEVEL1_BLOCK_SIZE, slot_in_block);
  return &table->level1[pb.partition][slot];
}

static inline kv_pair *
get_level2_kv_pair(iceberg_table  *table,
                   partition_block pb,
                   uint64_t        slot_in_block)
{
  uint64_t slot = kv_pair_offset(pb, LEVEL2_BLOCK_SIZE, slot_in_block);
  return &table->level2[pb.partition][slot];
}

static inline fingerprint_t *
get_level1_sketch(iceberg_table *table, partition_block pb)
{
  uint64_t offset = pb.block * LEVEL1_BLOCK_SIZE * sizeof(fingerprint_t);
  return &table->level1_sketch[pb.partition][offset];
}

static inline fingerprint_t *
get_level2_sketch(iceberg_table *table, partition_block pb)
{
  uint64_t offset = pb.block * LEVEL2_BLOCK_SIZE * sizeof(fingerprint_t);
  return &table->level2_sketch[pb.partition][offset];
}

static inline uint64_t
compute_level1_bytes(uint64_t blocks)
{
  return blocks * LEVEL1_BLOCK_SIZE * sizeof(kv_pair);
}

static inline uint64_t
compute_level2_bytes(uint64_t blocks)
{
  return blocks * LEVEL2_BLOCK_SIZE * sizeof(kv_pair);
}

static inline uint64_t
compute_level1_sketch_bytes(uint64_t blocks)
{
  return blocks * LEVEL1_BLOCK_SIZE * sizeof(fingerprint_t);
}

static inline uint64_t
compute_level2_sketch_bytes(uint64_t blocks)
{
  return blocks * LEVEL2_BLOCK_SIZE * sizeof(fingerprint_t);
}

static inline void
iceberg_allocate(iceberg_table *table,
                 uint64_t       partition_num,
                 uint64_t       num_blocks)
{
  // Level 1
  uint64_t level1_bytes               = compute_level1_bytes(num_blocks);
  table->level1[partition_num]        = util_mmap(level1_bytes);
  uint64_t level1_sketch_bytes        = compute_level1_sketch_bytes(num_blocks);
  table->level1_sketch[partition_num] = util_mmap(level1_sketch_bytes);

  // Level 2
  uint64_t level2_bytes               = compute_level2_bytes(num_blocks);
  table->level2[partition_num]        = util_mmap(level2_bytes);
  uint64_t level2_sketch_bytes        = compute_level2_sketch_bytes(num_blocks);
  table->level2_sketch[partition_num] = util_mmap(level2_sketch_bytes);

  // Level 3
  // No allocations required (static list heads in the iceberg_table struct

  // Resize Metadata
#ifdef ENABLE_RESIZE
  // create one marker for 8 blocks.
  size_t resize_marker_size   = sizeof(uint8_t) * num_blocks / 8;
  table->lv1_resize_marker[0] = util_mmap(resize_marker_size);
  table->lv2_resize_marker[0] = util_mmap(resize_marker_size);
  table->marker_sizes[0]      = resize_marker_size;
  table->lock                 = 0;
#endif
}

void
iceberg_init(iceberg_table *table, uint64_t log_slots)
{
  assert(table);
  memset(table, 0, sizeof(*table));

  uint64_t num_blocks           = 1 << (log_slots - LEVEL1_LOG_BLOCK_SIZE);
  table->nblocks                = num_blocks;
  table->log_num_blocks         = log_slots - LEVEL1_LOG_BLOCK_SIZE;
  table->log_initial_num_blocks = log2(num_blocks);
  table->resize_threshold       = RESIZE_THRESHOLD * total_capacity(table);

  iceberg_allocate(table, 0, num_blocks);

  counter_init(&table->num_items_per_level);

#ifdef ENABLE_RESIZE
  table->num_partitions = 0;
  table->lv1_resize_ctr = 0;
  table->lv2_resize_ctr = 0;
#endif
}

#ifdef ENABLE_RESIZE
static inline bool
is_lv1_resize_active(iceberg_table *table)
{
  return __atomic_load_n(&table->lv1_resize_ctr, __ATOMIC_SEQ_CST);
}

static inline bool
is_lv2_resize_active(iceberg_table *table)
{
  return __atomic_load_n(&table->lv2_resize_ctr, __ATOMIC_SEQ_CST);
}

static bool
is_resize_active(iceberg_table *table)
{
  return is_lv2_resize_active(table) || is_lv1_resize_active(table);
}

static bool
iceberg_setup_resize(iceberg_table *table)
{
  // grab write lock
  if (!lock(&table->lock)) {
    return false;
  }

  if (unlikely(!need_resize(table))) {
    unlock(&table->lock);
    return false;
  }
  if (is_resize_active(table)) {
    // finish the current resize
    iceberg_end(table);
    unlock(&table->lock);
    return false;
  }

  // Compute size of new partition
  uint64_t new_partition_num_blocks = table->nblocks;
  uint64_t new_partition_num        = table->num_partitions + 1;

  // Allocate the partition
  iceberg_allocate(table, new_partition_num, new_partition_num_blocks);

  // Reset the resize markers
  size_t resize_marker_size = sizeof(uint8_t) * new_partition_num_blocks / 8;
  table->marker_sizes[new_partition_num] = resize_marker_size;
  for (uint64_t i = 0; i <= new_partition_num; ++i) {
    memset(table->lv1_resize_marker[i], 0, table->marker_sizes[i]);
    memset(table->lv2_resize_marker[i], 0, table->marker_sizes[i]);
  }

  uint64_t total_blocks = table->nblocks * 2;

  // This is where the new partition becomes live
  table->num_partitions += 1;

  // update metadata
  table->nblocks = total_blocks;
  table->log_num_blocks += 1;
  table->resize_threshold = RESIZE_THRESHOLD * total_capacity(table);

  // reset the block ctr
  table->lv1_resize_ctr = table->nblocks / 2;
  table->lv2_resize_ctr = table->nblocks / 2;

  /*printf("Setting up finished\n");*/
  unlock(&table->lock);
  return true;
}

static bool iceberg_lv1_move_block(iceberg_table *table,
                                   uint64_t       bnum,
                                   uint64_t       tid);
static bool iceberg_lv2_move_block(iceberg_table *table,
                                   uint64_t       bnum,
                                   uint64_t       tid);

// finish moving blocks that are left during the last resize.
void
iceberg_end(iceberg_table *table)
{
  if (is_lv1_resize_active(table)) {
    for (uint64_t chunk = 0; chunk < table->nblocks / 8; ++chunk) {
      partition_block pb = decode_raw_chunk(table, chunk);
      // if fixing is needed set the marker
      if (!__sync_lock_test_and_set(
            &table->lv1_resize_marker[pb.partition][pb.block], 1)) {
        for (uint8_t i = 0; i < 8; ++i) {
          uint64_t idx = chunk * 8 + i;
          iceberg_lv1_move_block(table, idx, 0);
        }
        // set the marker for the dest block
        uint64_t dest_chunk = chunk + table->nblocks / 8 / 2;
        pb                  = decode_raw_chunk(table, dest_chunk);
        __sync_lock_test_and_set(
          &table->lv1_resize_marker[pb.partition][pb.block], 1);
      }
    }
  }
  if (is_lv2_resize_active(table)) {
    for (uint64_t chunk = 0; chunk < table->nblocks / 8; ++chunk) {
      partition_block pb = decode_raw_chunk(table, chunk);
      // if fixing is needed set the marker
      if (!__sync_lock_test_and_set(
            &table->lv2_resize_marker[pb.partition][pb.block], 1)) {
        for (uint8_t i = 0; i < 8; ++i) {
          uint64_t idx = chunk * 8 + i;
          iceberg_lv2_move_block(table, idx, 0);
        }
        // set the marker for the dest block
        uint64_t dest_chunk = chunk + table->nblocks / 8 / 2;
        pb                  = decode_raw_chunk(table, dest_chunk);
        __sync_lock_test_and_set(
          &table->lv2_resize_marker[pb.partition][pb.block], 1);
      }
    }
  }
}
#endif

static inline void
level3_lock_block(iceberg_table *table, uint64_t block)
{
  while (__atomic_test_and_set(&table->level3[block].lock, __ATOMIC_SEQ_CST)) {
    _mm_pause();
  }
}

static inline void
level3_unlock_block(iceberg_table *table, uint64_t block)
{
  __atomic_clear(&table->level3[block].lock, __ATOMIC_SEQ_CST);
}

static inline bool
iceberg_lv3_insert(iceberg_table  *table,
                   iceberg_value_t key,
                   iceberg_value_t value,
                   hash           *h,
                   uint64_t        tid)
{
  uint64_t block = get_level3_block(h);
  level3_lock_block(table, block);
  iceberg_lv3_node *new_node = malloc(sizeof(iceberg_lv3_node));
  assert(new_node);
  verbose_print_location(3, 0, block, 0, new_node);
  new_node->key             = key;
  new_node->val             = value;
  new_node->next_node       = table->level3[block].head;
  table->level3[block].head = new_node;
  counter_add(&table->num_items_per_level, LEVEL3, 1, tid);
  level3_unlock_block(table, block);
  return true;
}

static inline bool
iceberg_lv2_insert_internal(iceberg_table  *table,
                            iceberg_value_t key,
                            iceberg_value_t value,
                            hash           *h,
                            partition_block pb,
                            uint64_t        tid)
{
  fingerprint_t *sketch  = get_level2_sketch(table, pb);
  __mmask32      md_mask = slot_mask_32(sketch, 0);
  verbose_print_sketch(sketch, 8);
  verbose_print_mask8(md_mask);
  uint8_t popct = __builtin_popcountll(md_mask);

  if (unlikely(!popct))
    return false;

start:;
  uint64_t slot = __builtin_ctzll(md_mask);
  md_mask       = md_mask & ~(1ULL << slot);

  if (__sync_bool_compare_and_swap(&sketch[slot], 0, 1)) {
    counter_add(&table->num_items_per_level, LEVEL2, 1, tid);
    kv_pair *kv = get_level2_kv_pair(table, pb, slot);
    verbose_print_location(2, pb.partition, pb.block, slot, kv);
    atomic_write_128(key, value, kv);
    sketch[slot] = h->fingerprint;
    verbose_print_sketch(sketch, 8);
    return true;
  }
  goto start;

  return false;
}

static inline bool
iceberg_lv2_insert(iceberg_table  *table,
                   iceberg_value_t key,
                   iceberg_value_t value,
                   hash           *h,
                   uint64_t        tid)
{

  partition_block pb1     = get_block(table, h, LEVEL2_BLOCK1);
  fingerprint_t  *sketch1 = get_level2_sketch(table, pb1);
  verbose_print_sketch(sketch1, 8);
  __mmask32 md_mask1 = slot_mask_32(sketch1, 0);
  verbose_print_mask8(md_mask1);
  uint8_t popct1 = __builtin_popcountll(md_mask1);

  partition_block pb2      = get_block(table, h, LEVEL2_BLOCK2);
  fingerprint_t  *sketch2  = get_level2_sketch(table, pb2);
  __mmask32       md_mask2 = slot_mask_32(sketch2, 0);
  verbose_print_sketch(sketch2, 8);
  verbose_print_mask8(md_mask2);
  uint8_t popct2 = __builtin_popcountll(md_mask2);

  uint64_t raw_block = h->raw_block[LEVEL2_BLOCK1];
  if (popct2 > popct1) {
    pb1       = pb2;
    raw_block = h->raw_block[LEVEL2_BLOCK2];
  }

#ifdef ENABLE_RESIZE
  // move blocks if resize is active and not already moved.
  if (unlikely(is_lv2_resize_active(table) &&
               raw_block < (table->nblocks >> 1))) {
    uint64_t        chunk = raw_block / 8;
    partition_block pb    = decode_raw_chunk(table, chunk);
    // if fixing is needed set the marker
    if (!__sync_lock_test_and_set(
          &table->lv2_resize_marker[pb.partition][pb.block], 1)) {
      for (uint8_t i = 0; i < 8; ++i) {
        uint64_t idx = chunk * 8 + i;
        /*printf("LV2 Before: Moving block: %ld load: %f\n", idx,
         * iceberg_block_load(table, idx, 2));*/
        iceberg_lv2_move_block(table, idx, tid);
        /*printf("LV2 After: Moving block: %ld load: %f\n", idx,
         * iceberg_block_load(table, idx, 2));*/
      }
      // set the marker for the dest block
      uint64_t dest_chunk = chunk + table->nblocks / 8 / 2;
      pb                  = decode_raw_chunk(table, dest_chunk);
      __sync_lock_test_and_set(
        &table->lv2_resize_marker[pb.partition][pb.block], 1);
    }
  }
#endif

  if (iceberg_lv2_insert_internal(table, key, value, h, pb1, tid)) {
    return true;
  }

  return iceberg_lv3_insert(table, key, value, h, tid);
}

static inline bool
iceberg_insert_internal(iceberg_table  *table,
                        iceberg_value_t key,
                        iceberg_value_t value,
                        hash           *h,
                        partition_block pb,
                        uint64_t        tid)
{
  fingerprint_t *sketch  = get_level1_sketch(table, pb);
  __mmask64      md_mask = slot_mask_64(sketch, 0);
  verbose_print_sketch(sketch, 64);
  verbose_print_mask64(md_mask);

  uint8_t popct = __builtin_popcountll(md_mask);

  if (unlikely(!popct)) {
    return false;
  }

  uint64_t slot = __builtin_ctzll(md_mask);
  counter_add(&table->num_items_per_level, LEVEL1, 1, tid);
  kv_pair *kv = get_level1_kv_pair(table, pb, slot);
  verbose_print_location(1, pb.partition, pb.block, slot, kv);
  atomic_write_128(key, value, kv);
  sketch[slot] = h->fingerprint;
  verbose_print_sketch(sketch, 64);
  return true;
}

static inline bool iceberg_get_value_internal(iceberg_table   *table,
                                              iceberg_value_t  key,
                                              iceberg_value_t *value,
                                              hash            *h,
                                              uint64_t         tid);

__attribute__((always_inline)) inline bool
iceberg_insert(iceberg_table  *table,
               iceberg_value_t key,
               iceberg_value_t value,
               uint64_t        tid)
{
  verbose_print_operation("INSERT:", key, value);

#ifdef ENABLE_RESIZE
  if (unlikely(need_resize(table))) {
    iceberg_setup_resize(table);
  }
#endif

  hash h = hash_key(table, &key);

#ifdef ENABLE_RESIZE
  // move blocks if resize is active and not already moved.
  if (unlikely(is_lv1_resize_active(table) &&
               h.raw_block[LEVEL1_BLOCK] < (table->nblocks >> 1))) {
    uint64_t        chunk = h.raw_block[LEVEL1_BLOCK] / 8;
    partition_block pb    = decode_raw_chunk(table, chunk);
    // if fixing is needed set the marker
    if (!__sync_lock_test_and_set(
          &table->lv1_resize_marker[pb.partition][pb.block], 1)) {
      for (uint8_t i = 0; i < 8; ++i) {
        uint64_t idx = chunk * 8 + i;
        /*printf("LV1 Before: Moving block: %ld load: %f\n", idx,
         * iceberg_block_load(table, idx, 1));*/
        iceberg_lv1_move_block(table, idx, tid);
        /*printf("LV1 After: Moving block: %ld load: %f\n", idx,
         * iceberg_block_load(table, idx, 1));*/
      }
      // set the marker for the dest block
      uint64_t dest_chunk = chunk + table->nblocks / 8 / 2;
      pb                  = decode_raw_chunk(table, dest_chunk);
      __sync_lock_test_and_set(
        &table->lv1_resize_marker[pb.partition][pb.block], 1);
    }
  }
#endif

  partition_block pb     = get_block(table, &h, LEVEL1_BLOCK);
  fingerprint_t  *sketch = get_level1_sketch(table, pb);
  lock_block(sketch);
  verbose_print_sketch(sketch, 64);
  iceberg_value_t v;
  verbose_print_operation("INTERNAL QUERY:", key, value);
  bool ret = true;
  if (unlikely(iceberg_get_value_internal(table, key, &v, &h, tid))) {
    ret = true;
    goto out;
  }
  verbose_end("INTERNAL QUERY", true);

  ret = iceberg_insert_internal(table, key, value, &h, pb, tid);
  if (!ret) {
    ret = iceberg_lv2_insert(table, key, value, &h, tid);
  }

out:
  verbose_end("INSERT", false);
  unlock_block(sketch);
  verbose_print_sketch(sketch, 64);
  return ret;
}

__attribute__((unused)) static inline bool
level3_delete(iceberg_table *table, iceberg_value_t key, hash *h, uint64_t tid)
{
  uint64_t block = get_level3_block(h);

  if (likely(table->level3[block].head == NULL)) {
    return false;
  }

  bool ret = false;
  level3_lock_block(table, block);
  iceberg_lv3_node **node = &table->level3[block].head;

  while (*node != NULL) {
    if ((*node)->key == key) {
      iceberg_lv3_node *dead_node = *node;
      *node                       = (*node)->next_node;
      free(dead_node);
      counter_add(&table->num_items_per_level, LEVEL3, -1, tid);
      ret = true;
      goto unlock_out;
    }
    node = &(*node)->next_node;
  }

unlock_out:
  level3_unlock_block(table, block);
  return ret;
}

__attribute__((unused)) static inline bool
level2_delete(iceberg_table *table, iceberg_value_t key, hash *h, uint64_t tid)
{
  for (block_type lvl = LEVEL2_BLOCK1; lvl < NUM_LEVELS; ++lvl) {
    partition_block pb = get_block(table, h, lvl);

#ifdef ENABLE_RESIZE
    // check if there's an active resize and block isn't fixed yet
    if (unlikely(is_lv2_resize_active(table) &&
                 pb.partition == table->num_partitions)) {
      uint64_t        mask      = ~(1ULL << (table->log_num_blocks - 1));
      uint64_t        old_index = h->raw_block[lvl] & mask;
      uint64_t        chunk     = old_index / 8;
      partition_block pb        = decode_raw_chunk(table, chunk);
      if (__atomic_load_n(&table->lv2_resize_marker[pb.partition][pb.block],
                          __ATOMIC_SEQ_CST) == 0) { // not fixed yet
        partition_block old_pb  = decode_raw_chunk(table, old_index);
        fingerprint_t  *sketch  = get_level2_sketch(table, old_pb);
        __mmask32       md_mask = slot_mask_32(sketch, h->fingerprint);
        verbose_print_sketch(sketch, 8);
        verbose_print_mask8(md_mask);
        uint8_t popct = __builtin_popcount(md_mask);

        for (uint8_t i = 0; i < popct; ++i) {
          uint64_t slot = word_select(md_mask, i);

          kv_pair *candidate_kv = get_level1_kv_pair(table, old_pb, slot);
          if (candidate_kv->key == key) {
            verbose_print_location(
              2, old_pb.partition, old_pb.block, slot, candidate_kv);
            candidate_kv->key = 0;
            candidate_kv->val = 0;
            sketch[slot]      = 0;
            counter_add(&table->num_items_per_level, LEVEL2, -1, tid);
            return true;
          }
        }
      } else {
        // wait for the old block to be fixed
        uint64_t dest_chunk = h->raw_block[lvl] / 8;
        pb                  = decode_raw_chunk(table, dest_chunk);
        while (
          __atomic_load_n(&table->lv2_resize_marker[pb.partition][pb.block],
                          __ATOMIC_SEQ_CST) == 0)
          ;
      }
    }
#endif

    fingerprint_t *sketch  = get_level2_sketch(table, pb);
    __mmask32      md_mask = slot_mask_32(sketch, h->fingerprint);
    verbose_print_sketch(sketch, 8);
    verbose_print_mask8(md_mask);
    uint8_t popct = __builtin_popcount(md_mask);

    for (uint8_t i = 0; i < popct; ++i) {
      uint64_t slot = word_select(md_mask, i);

      kv_pair *candidate_kv = get_level2_kv_pair(table, pb, slot);
      if (candidate_kv->key == key) {
        verbose_print_location(2, pb.partition, pb.block, slot, candidate_kv);
        candidate_kv->key = 0;
        candidate_kv->val = 0;
        sketch[slot]      = 0;
        counter_add(&table->num_items_per_level, LEVEL2, -1, tid);
        return true;
      }
    }
  }

  return level3_delete(table, key, h, tid);
}

__attribute__((always_inline)) inline bool
iceberg_remove(iceberg_table *table, iceberg_value_t key, uint64_t tid)
{
  verbose_print_operation("DELETE:", key, 0);
  bool ret = true;

  hash            h  = hash_key(table, &key);
  partition_block pb = get_block(table, &h, LEVEL1_BLOCK);

#ifdef ENABLE_RESIZE
  // check if there's an active resize and block isn't fixed yet
  if (unlikely(is_lv1_resize_active(table) &&
               h.raw_block[LEVEL1_BLOCK] >= (table->nblocks >> 1))) {
    uint64_t        mask      = ~(1ULL << (table->log_num_blocks - 1));
    uint64_t        old_index = h.raw_block[LEVEL1_BLOCK] & mask;
    uint64_t        chunk     = old_index / 8;
    partition_block chunk_pb  = decode_raw_chunk(table, chunk);
    if (__atomic_load_n(
          &table->lv1_resize_marker[chunk_pb.partition][chunk_pb.block],
          __ATOMIC_SEQ_CST) == 0) { // not fixed yet
      partition_block old_pb  = decode_raw_block(table, old_index);
      fingerprint_t  *sketch  = get_level1_sketch(table, old_pb);
      __mmask64       md_mask = slot_mask_64(sketch, h.fingerprint);
      verbose_print_sketch(sketch, 64);
      verbose_print_mask64(md_mask);
      uint8_t popct = __builtin_popcountll(md_mask);

      for (uint8_t i = 0; i < popct; ++i) {
        uint64_t slot = word_select(md_mask, i);

        kv_pair *candidate_kv = get_level1_kv_pair(table, old_pb, slot);
        if (candidate_kv->key == key) {
          verbose_print_location(
            1, old_pb.partition, old_pb.block, slot, candidate_kv);
          candidate_kv->key = 0;
          candidate_kv->val = 0;
          counter_add(&table->num_items_per_level, LEVEL1, -1, tid);
          sketch[slot] = 0;
          verbose_print_sketch(sketch, 64);
          goto out;
        }
      }
    } else {
      // wait for the old block to be fixed
      uint64_t dest_chunk = h.raw_block[LEVEL1_BLOCK] / 8;
      chunk_pb            = decode_raw_chunk(table, dest_chunk);
      while (__atomic_load_n(
               &table->lv1_resize_marker[chunk_pb.partition][chunk_pb.block],
               __ATOMIC_SEQ_CST) == 0)
        ;
    }
  }
#endif

  fingerprint_t *sketch = get_level1_sketch(table, pb);
  lock_block(sketch);
  verbose_print_sketch(sketch, 64);
  __mmask64 md_mask = slot_mask_64(sketch, h.fingerprint);
  verbose_print_mask64(md_mask);
  uint8_t popct = __builtin_popcountll(md_mask);

  for (uint8_t i = 0; i < popct; ++i) {
    uint64_t slot = word_select(md_mask, i);

    kv_pair *candidate_kv = get_level1_kv_pair(table, pb, slot);
    if (candidate_kv->key == key) {
      verbose_print_location(1, pb.partition, pb.block, slot, candidate_kv);
      candidate_kv->key = 0;
      candidate_kv->val = 0;
      counter_add(&table->num_items_per_level, LEVEL1, -1, tid);
      sketch[slot] = 0;
      verbose_print_sketch(sketch, 64);
      goto unlock_out;
    }
  }

  ret = level2_delete(table, key, &h, tid);

unlock_out:
  unlock_block(sketch);
out:
  verbose_end("DELETE", false);
  return ret;
}

static inline bool
iceberg_lv3_get_value_internal(iceberg_table   *table,
                               iceberg_value_t  key,
                               iceberg_value_t *value,
                               hash            *h)
{
  uint64_t block = get_level3_block(h);

  // Unlocked check to see if the node is empty
  if (likely(table->level3[block].head == NULL)) {
    return false;
  }

  bool ret = false;
  level3_lock_block(table, block);
  iceberg_lv3_node *node = table->level3[block].head;
  while (node != NULL) {
    if (node->key == key) {
      *value = node->val;
      ret    = true;
      goto unlock_out;
    }
    node = node->next_node;
  }

unlock_out:
  level3_unlock_block(table, block);
  return ret;
}

static inline bool
iceberg_lv3_get_value(iceberg_table   *table,
                      iceberg_value_t  key,
                      iceberg_value_t *value,
                      hash            *h)
{
  return iceberg_lv3_get_value_internal(table, key, value, h);
}

static inline bool
iceberg_lv2_get_value(iceberg_table   *table,
                      iceberg_value_t  key,
                      iceberg_value_t *value,
                      hash            *h)
{
  for (block_type lvl = LEVEL2_BLOCK1; lvl < NUM_LEVELS; ++lvl) {
    partition_block pb = get_block(table, h, lvl);

#ifdef ENABLE_RESIZE
    // check if there's an active resize and block isn't fixed yet
    if (unlikely(is_lv2_resize_active(table) &&
                 h->raw_block[lvl] >= (table->nblocks >> 1))) {
      uint64_t        mask      = ~(1ULL << (table->log_num_blocks - 1));
      uint64_t        old_index = h->raw_block[lvl] & mask;
      uint64_t        chunk     = old_index / 8;
      partition_block chunk_pb  = decode_raw_chunk(table, chunk);
      if (__atomic_load_n(
            &table->lv2_resize_marker[chunk_pb.partition][chunk_pb.block],
            __ATOMIC_SEQ_CST) == 0) { // not fixed yet
        partition_block old_pb  = decode_raw_block(table, old_index);
        fingerprint_t  *sketch  = get_level2_sketch(table, old_pb);
        __mmask32       md_mask = slot_mask_32(sketch, h->fingerprint);
        verbose_print_sketch(sketch, 8);
        verbose_print_mask8(md_mask);

        while (md_mask != 0) {
          uint32_t slot = __builtin_ctz(md_mask);
          md_mask       = md_mask & ~(1U << slot);

          kv_pair *candidate_kv = get_level2_kv_pair(table, old_pb, slot);
          if (candidate_kv->key == key) {
            verbose_print_location(
              2, old_pb.partition, old_pb.block, slot, candidate_kv);
            *value = candidate_kv->val;
            return true;
          }
        }
      } else {
        // wait for the old block to be fixed
        uint64_t dest_chunk = h->raw_block[lvl] / 8;
        chunk_pb            = decode_raw_chunk(table, dest_chunk);
        while (__atomic_load_n(
                 &table->lv2_resize_marker[chunk_pb.partition][chunk_pb.block],
                 __ATOMIC_SEQ_CST) == 0)
          ;
      }
    }
#endif

    fingerprint_t *sketch  = get_level2_sketch(table, pb);
    __mmask32      md_mask = slot_mask_32(sketch, h->fingerprint);
    verbose_print_sketch(sketch, 8);
    verbose_print_mask8(md_mask);

    while (md_mask != 0) {
      int slot = __builtin_ctz(md_mask);
      md_mask  = md_mask & ~(1U << slot);

      kv_pair *candidate_kv = get_level2_kv_pair(table, pb, slot);
      if (candidate_kv->key == key) {
        verbose_print_location(2, pb.partition, pb.block, slot, candidate_kv);
        *value = candidate_kv->val;
        return true;
      }
    }
  }

  return iceberg_lv3_get_value(table, key, value, h);
}

__attribute__((always_inline)) inline bool
iceberg_get_value_internal(iceberg_table   *table,
                           iceberg_value_t  key,
                           iceberg_value_t *value,
                           hash            *h,
                           uint64_t         tid)
{

#ifdef ENABLE_RESIZE
  // check if there's an active resize and block isn't fixed yet
  if (unlikely(is_lv1_resize_active(table) &&
               h->raw_block[LEVEL1_BLOCK] >= (table->nblocks >> 1))) {
    uint64_t        mask      = ~(1ULL << (table->log_num_blocks - 1));
    uint64_t        old_index = h->raw_block[LEVEL1_BLOCK] & mask;
    uint64_t        chunk     = old_index / 8;
    partition_block chunk_pb  = decode_raw_chunk(table, chunk);
    if (__atomic_load_n(
          &table->lv1_resize_marker[chunk_pb.partition][chunk_pb.block],
          __ATOMIC_SEQ_CST) == 0) { // not fixed yet
      partition_block old_pb  = decode_raw_block(table, old_index);
      fingerprint_t  *sketch  = get_level1_sketch(table, old_pb);
      __mmask64       md_mask = slot_mask_64(sketch, h->fingerprint);
      verbose_print_sketch(sketch, 64);
      verbose_print_mask64(md_mask);

      while (md_mask != 0) {
        int slot = __builtin_ctzll(md_mask);
        md_mask  = md_mask & ~(1ULL << slot);

        kv_pair *candidate_kv = get_level1_kv_pair(table, old_pb, slot);
        if (candidate_kv->key == key) {
          verbose_print_location(
            1, old_pb.partition, old_pb.block, slot, candidate_kv);
          *value = candidate_kv->val;
          return true;
        }
      }
    } else {
      // wait for the old block to be fixed
      uint64_t dest_chunk = h->raw_block[LEVEL1_BLOCK] / 8;
      chunk_pb            = decode_raw_chunk(table, dest_chunk);
      while (__atomic_load_n(
               &table->lv1_resize_marker[chunk_pb.partition][chunk_pb.block],
               __ATOMIC_SEQ_CST) == 0)
        ;
    }
  }
#endif

  partition_block pb     = decode_raw_block(table, h->raw_block[LEVEL1_BLOCK]);
  fingerprint_t  *sketch = get_level1_sketch(table, pb);
  verbose_print_sketch(sketch, 64);
  __mmask64 md_mask = slot_mask_64(sketch, h->fingerprint);
  verbose_print_mask64(md_mask);

  while (md_mask != 0) {
    int slot = __builtin_ctzll(md_mask);
    md_mask  = md_mask & ~(1ULL << slot);

    kv_pair *candidate_kv = get_level1_kv_pair(table, pb, slot);
    if (candidate_kv->key == key) {
      verbose_print_location(1, pb.partition, pb.block, slot, candidate_kv);
      *value = candidate_kv->val;
      return true;
    }
  }

  bool ret = iceberg_lv2_get_value(table, key, value, h);

  return ret;
}


__attribute__((always_inline)) inline bool
iceberg_get_value(iceberg_table   *table,
                  iceberg_value_t  key,
                  iceberg_value_t *value,
                  uint64_t         tid)
{
  verbose_print_operation("QUERY:", key, 0);
  bool ret = true;

  hash h = hash_key(table, &key);
  ret    = iceberg_get_value_internal(table, key, value, &h, tid);
  verbose_end("QUERY", false);
  return ret;
}

#ifdef ENABLE_RESIZE
static inline bool
iceberg_nuke_key(iceberg_table *table,
                 uint64_t       level,
                 uint64_t       index,
                 uint64_t       slot,
                 uint64_t       tid)
{
  partition_block pb = decode_raw_block(table, index);

  if (level == 1) {
    kv_pair *kv           = get_level1_kv_pair(table, pb, slot);
    kv->key               = 0;
    kv->val               = 0;
    fingerprint_t *sketch = get_level1_sketch(table, pb);
    sketch[slot]          = 0;
    counter_add(&table->num_items_per_level, LEVEL1, -1, tid);
  } else if (level == 2) {
    kv_pair *kv           = get_level2_kv_pair(table, pb, slot);
    kv->key               = 0;
    kv->val               = 0;
    fingerprint_t *sketch = get_level2_sketch(table, pb);
    sketch[slot]          = 0;
    counter_add(&table->num_items_per_level, LEVEL2, -1, tid);
  }

  return true;
}

static bool
iceberg_lv1_move_block(iceberg_table *table, uint64_t bnum, uint64_t tid)
{
  // grab a block
  uint64_t bctr =
    __atomic_fetch_sub(&table->lv1_resize_ctr, 1, __ATOMIC_SEQ_CST);
  assert(bctr != 0);

  partition_block pb = decode_raw_block(table, bnum);
  // relocate items in level1
  for (uint64_t j = 0; j < LEVEL1_BLOCK_SIZE; ++j) {
    kv_pair *kv = get_level1_kv_pair(table, pb, j);
    if (kv->key == 0) {
      continue;
    }

    hash h = hash_key(table, &kv->key);

    // move to new location
    if (h.raw_block[LEVEL1_BLOCK] != bnum) {
      partition_block local_pb =
        decode_raw_block(table, h.raw_block[LEVEL1_BLOCK]);
      if (!iceberg_insert_internal(
            table, kv->key, kv->val, &h, local_pb, tid)) {
        printf("Failed insert during resize lv1\n");
        exit(0);
      }
      if (!iceberg_nuke_key(table, 1, bnum, j, tid)) {
        printf("Failed remove during resize lv1. key: %" PRIu64
               ", block: %" PRIu64 "\n",
               kv->key,
               bnum);
        exit(0);
      }
      // iceberg_value_t *val;
      // if (!iceberg_get_value(table, key, &val, tid)) {
      //  printf("Key not found during resize lv1: %ld\n", key);
      // exit(0);
      // }
    }
  }

  return false;
}

static bool
iceberg_lv2_move_block(iceberg_table *table, uint64_t bnum, uint64_t tid)
{
  // grab a block
  uint64_t bctr =
    __atomic_fetch_sub(&table->lv2_resize_ctr, 1, __ATOMIC_SEQ_CST);
  assert(bctr != 0);

  partition_block pb   = decode_raw_block(table, bnum);
  uint64_t        mask = ~(1ULL << (table->log_num_blocks - 1));
  // relocate items in level2
  for (uint64_t j = 0; j < LEVEL2_BLOCK_SIZE; ++j) {
    kv_pair *kv = get_level2_kv_pair(table, pb, j);
    if (kv->key == 0) {
      continue;
    }

    hash h = hash_key(table, &kv->key);

    if ((h.raw_block[LEVEL2_BLOCK1] & mask) == bnum &&
        h.raw_block[LEVEL2_BLOCK1] != bnum) {
      partition_block local_pb =
        decode_raw_block(table, h.raw_block[LEVEL2_BLOCK1]);
      if (!iceberg_lv2_insert_internal(
            table, kv->key, kv->val, &h, local_pb, tid)) {
        if (!iceberg_lv2_insert(table, kv->key, kv->val, &h, tid)) {
          printf("Failed insert during resize lv2\n");
          exit(0);
        }
      }
      if (!iceberg_nuke_key(table, 2, bnum, j, tid)) {
        printf("Failed remove during resize lv2\n");
        exit(0);
      }
    } else if ((h.raw_block[LEVEL2_BLOCK2] & mask) == bnum &&
               h.raw_block[LEVEL2_BLOCK2] != bnum) {
      partition_block local_pb =
        decode_raw_block(table, h.raw_block[LEVEL2_BLOCK2]);
      if (!iceberg_lv2_insert_internal(
            table, kv->key, kv->val, &h, local_pb, tid)) {
        if (!iceberg_lv2_insert(table, kv->key, kv->val, &h, tid)) {
          printf("Failed insert during resize lv2\n");
          exit(0);
        }
      }
      if (!iceberg_nuke_key(table, 2, bnum, j, tid)) {
        printf("Failed remove during resize lv2\n");
        exit(0);
      }
    }
  }
  return false;
}

#endif
