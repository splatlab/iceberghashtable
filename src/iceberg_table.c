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

#define FP_FREE 0
#define LEVEL2_CHOICE1_MASK 0xff
#define LEVEL2_CHOICE2_MASK 0xff00

#define LEVEL1_BLOCKS_PER_RESIZE_CHUNK 8
#define LEVEL1_LOG_BLOCKS_PER_RESIZE_CHUNK 3
#define LEVEL2_BLOCKS_PER_RESIZE_CHUNK 8
#define LEVEL2_LOG_BLOCKS_PER_RESIZE_CHUNK 3

#define KEY_FREE 0ULL
#define VALUE_FREE 0ULL

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
level1_load(iceberg_table *table)
{
  counter *cntr = &table->num_items_per_level;
  counter_sync(cntr, LEVEL1);
  return counter_get(cntr, LEVEL1);
}

static inline uint64_t
level1_load_approx(iceberg_table *table)
{
  counter *cntr = &table->num_items_per_level;
  return counter_get(cntr, LEVEL1);
}

uint64_t
level2_load(iceberg_table *table)
{
  counter *cntr = &table->num_items_per_level;
  counter_sync(cntr, LEVEL2);
  return counter_get(cntr, LEVEL2);
}

static inline uint64_t
level2_load_approx(iceberg_table *table)
{
  counter *cntr = &table->num_items_per_level;
  return counter_get(cntr, LEVEL2);
}

uint64_t
level3_load(iceberg_table *table)
{
  counter *cntr = &table->num_items_per_level;
  counter_sync(cntr, LEVEL3);
  return counter_get(cntr, LEVEL3);
}

static inline uint64_t
level3_load_approx(iceberg_table *table)
{
  counter *cntr = &table->num_items_per_level;
  return counter_get(cntr, LEVEL3);
}

uint64_t
iceberg_load(iceberg_table *table)
{
  return level1_load(table) + level2_load(table) + level3_load(table);
}

uint64_t
iceberg_load_approx(iceberg_table *table)
{
  return level1_load_approx(table) + level2_load_approx(table) +
         level3_load_approx(table);
}

static inline uint64_t
capacity(iceberg_table *table)
{
  return table->num_blocks * (LEVEL1_BLOCK_SIZE + LEVEL2_BLOCK_SIZE);
}

inline double
iceberg_load_factor(iceberg_table *table)
{
  return (double)iceberg_load(table) / (double)capacity(table);
}

#ifdef ENABLE_RESIZE
static inline bool
needs_resize(iceberg_table *table)
{
  return iceberg_load_approx(table) >= table->resize_threshold;
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
hash_key(iceberg_table *table, iceberg_key_t *key)
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
  partition_block pb        = {0};
  uint64_t        high_bits = raw_block >> init_log;
  pb.partition              = 64 - _lzcnt_u64(high_bits);
  uint64_t adj              = 1ULL << pb.partition;
  adj                       = adj >> 1;
  adj                       = adj << init_log;
  pb.block                  = raw_block - adj;
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

#ifdef ENABLE_RESIZE
static inline partition_block
level1_decode_raw_chunk(iceberg_table *table, uint64_t raw_chunk)
{
  return decode_raw_internal(table->log_initial_num_blocks - LEVEL1_LOG_BLOCKS_PER_RESIZE_CHUNK , raw_chunk);
}

static inline partition_block
level2_decode_raw_chunk(iceberg_table *table, uint64_t raw_chunk)
{
  return decode_raw_internal(table->log_initial_num_blocks - LEVEL2_LOG_BLOCKS_PER_RESIZE_CHUNK , raw_chunk);
}
#endif

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
sketch_match_8(uint8_t *sketch, uint8_t fprint)
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

static inline uint32_t
double_sketch_match_8(uint8_t *sketch1, uint8_t *sketch2, uint8_t fprint)
{
  __m256i bcast = _mm256_set1_epi8(fprint);
  __m256i block = _mm256_maskz_loadu_epi64(0x1, (const __m256i *)sketch1);
  block = _mm256_mask_loadu_epi64(block, 0x2, (const __m256i *)(sketch2 - 8));
#if defined __AVX512BW__ && defined __AVX512VL__
  return _mm256_mask_cmpeq_epi8_mask(0xffff, bcast, block);
#else
  assert(0);
  __m256i cmp = _mm256_cmpeq_epi8(bcast, block);
  return _mm256_movemask_epi8(cmp);
#endif
}


#if defined __AVX512F__ && defined __AVX512BW__
static inline uint64_t
sketch_match_64(fingerprint_t *sketch, fingerprint_t fp)
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
sketch_match_32_half(__m256i fprint, __m256i md, __m256i mask)
{
  __m256i masked_fp = _mm256_or_si256(fprint, mask);
  __m256i masked_md = _mm256_or_si256(md, mask);
  __m256i cmp = _mm256_cmpeq_epi8(masked_md, masked_fp);
  return _mm256_movemask_epi8(cmp);
}

static inline uint64_t
sketch_match_64(fingerprint_t *sketch, fingerprint_t fp)
{
  __m256i fprint = _mm256_set1_epi8(fp);

  __m256i md1 = _mm256_load_si256((const __m256i *)(sketch));
  __m256i mask1 = _mm256_load_si256((const __m256i *)(broadcast_mask));
  uint64_t result1 = sketch_match_32_half(fprint, md1, mask1);

  __m256i md2 = _mm256_load_si256((const __m256i *)(&sketch[32]));
  __m256i mask2 = _mm256_load_si256((const __m256i *)(&broadcast_mask[32]));
  uint64_t result2 = sketch_match_32_half(fprint, md2, mask2);

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
  size_t resize_marker_size      = sizeof(uint8_t) * num_blocks / 8;
  table->level1_resize_marker[0] = util_mmap(resize_marker_size);
  table->level2_resize_marker[0] = util_mmap(resize_marker_size);
  table->marker_sizes[0]         = resize_marker_size;
  table->lock                    = 0;
#endif
}

void
iceberg_init(iceberg_table *table, uint64_t log_slots)
{
  assert(table);
  memset(table, 0, sizeof(*table));

  uint64_t num_blocks           = 1 << (log_slots - LEVEL1_LOG_BLOCK_SIZE);
  table->num_blocks                = num_blocks;
  table->log_num_blocks         = log_slots - LEVEL1_LOG_BLOCK_SIZE;
  table->log_initial_num_blocks = log2(num_blocks);

  iceberg_allocate(table, 0, num_blocks);

  counter_init(&table->num_items_per_level);

#ifdef ENABLE_RESIZE
  table->num_partitions    = 0;
  table->level1_resize_counter = 0;
  table->level2_resize_counter = 0;
  table->resize_threshold  = RESIZE_THRESHOLD * capacity(table);
#endif
}

#ifdef ENABLE_RESIZE
static inline bool
level1_resize_active(iceberg_table *table)
{
  return table->level1_resize_counter;
}

static inline bool
level2_resize_active(iceberg_table *table)
{
  return table->level2_resize_counter;
}

static inline bool
is_resize_active(iceberg_table *table)
{
  return level2_resize_active(table) || level1_resize_active(table);
}
#endif

static void
maybe_create_new_partition(iceberg_table *table, uint64_t tid)
{
#ifdef ENABLE_RESIZE
  if (likely(!needs_resize(table))) {
    return;
  }

  // grab write lock
  if (!lock(&table->lock)) {
    return;
  }

  if (unlikely(!needs_resize(table))) {
    unlock(&table->lock);
    return;
  }

  if (is_resize_active(table)) {
    // finish the current resize
    iceberg_end(table, tid);
  }

  // Compute size of new partition
  uint64_t new_partition_num_blocks = table->num_blocks;
  uint64_t new_partition_num        = table->num_partitions + 1;

  // Allocate the partition
  iceberg_allocate(table, new_partition_num, new_partition_num_blocks);

  // Reset the resize markers
  size_t resize_marker_size = sizeof(uint8_t) * new_partition_num_blocks / 8;
  table->marker_sizes[new_partition_num] = resize_marker_size;
  for (uint64_t i = 0; i <= new_partition_num; ++i) {
    memset(table->level1_resize_marker[i], 0, table->marker_sizes[i]);
    memset(table->level2_resize_marker[i], 0, table->marker_sizes[i]);
  }

  uint64_t total_blocks = table->num_blocks * 2;

  // This is where the new partition becomes live
  table->num_partitions += 1;

  // update metadata
  table->num_blocks = total_blocks;
  table->log_num_blocks += 1;
  table->resize_threshold = RESIZE_THRESHOLD * capacity(table);

  // reset the block counter
  table->level1_resize_counter = table->num_blocks / 2;
  table->level2_resize_counter = table->num_blocks / 2;

  /*printf("Setting up finished\n");*/
  unlock(&table->lock);
#  endif
}

#ifdef ENABLE_RESIZE
static inline void
level1_decrement_resize_counter(iceberg_table *table)
{
  uint64_t counter_value_pre =
    __atomic_fetch_sub(&table->level1_resize_counter, 1, __ATOMIC_SEQ_CST);
  assert(counter_value_pre != 0);
}

static inline bool
level1_key_should_move(iceberg_table *table, partition_block key_pb, partition_block new_block_pb)
{
  return key_pb.partition == new_block_pb.partition && key_pb.block == new_block_pb.block;
}

static inline partition_block
compute_new_partition_block(iceberg_table *table, partition_block pb)
{
  uint64_t raw_block_num = (1 << (table->log_initial_num_blocks + pb.partition)) + pb.block;
  partition_block new_pb = { .partition = table->num_partitions -1, .block = raw_block_num };
  return new_pb;
}

static inline partition_block
level1_compute_new_partition_chunk(iceberg_table *table, partition_block chunk_pb)
{
  uint64_t raw_block_num = (1 << (table->log_initial_num_blocks - LEVEL1_LOG_BLOCKS_PER_RESIZE_CHUNK + chunk_pb.partition)) + chunk_pb.block;
  partition_block new_pb = { .partition = table->num_partitions -1, .block = raw_block_num };
  return new_pb;
}

static inline partition_block
level2_compute_new_partition_chunk(iceberg_table *table, partition_block chunk_pb)
{
  uint64_t raw_block_num = (1 << (table->log_initial_num_blocks - LEVEL2_LOG_BLOCKS_PER_RESIZE_CHUNK + chunk_pb.partition)) + chunk_pb.block;
  partition_block new_pb = { .partition = table->num_partitions -1, .block = raw_block_num };
  return new_pb;
}

static inline bool
level1_insert_into_block(iceberg_table  *table,
                       iceberg_key_t   key,
                       iceberg_value_t value,
                       hash           *h,
                       partition_block pb,
                       uint64_t        tid)
{
  fingerprint_t *sketch     = get_level1_sketch(table, pb);
  uint64_t       match_mask = sketch_match_64(sketch, 0);
  verbose_print_sketch(sketch, 64);
  verbose_print_mask_64(match_mask);

  uint8_t popcnt = __builtin_popcountll(match_mask);

  if (unlikely(!popcnt)) {
    return false;
  }

  uint64_t slot = __builtin_ctzll(match_mask);
  counter_increment(&table->num_items_per_level, LEVEL1, tid);
  kv_pair *kv = get_level1_kv_pair(table, pb, slot);
  verbose_print_location(1, pb.partition, pb.block, slot, kv);
  atomic_write_128(key, value, kv);
  sketch[slot] = h->fingerprint;
  verbose_print_sketch(sketch, 64);
  return true;
}

static inline void
delete_from_slot(iceberg_table *table,
                        kv_pair *kv,
                        fingerprint_t *sketch,
                        uint64_t slot_num,
                        level_type lvl,
                        uint64_t tid)
{
  atomic_write_128(KEY_FREE, VALUE_FREE, kv);
  sketch[slot_num] = FP_FREE;
  counter_decrement(&table->num_items_per_level, lvl, tid);
}

static inline void
level1_move_block(iceberg_table *table, partition_block pb, uint64_t tid)
{
  partition_block new_block_pb = compute_new_partition_block(table, pb);
  fingerprint_t *sketch = get_level1_sketch(table, pb);
  for (uint64_t i = 0; i < LEVEL1_BLOCK_SIZE; i++) {
    kv_pair *kv = get_level1_kv_pair(table, pb, i);
    if (kv->key == KEY_FREE) {
      continue;
    }

    hash h = hash_key(table, &kv->key);

    partition_block key_pb = get_block(table, &h, LEVEL1_BLOCK);
    if (level1_key_should_move(table, key_pb, new_block_pb)) {
      bool ret = level1_insert_into_block(table, kv->key, kv->val, &h, key_pb, tid);
      assert(ret);
      delete_from_slot(table, kv, sketch, i, LEVEL1, tid);
    }
  }

  level1_decrement_resize_counter(table);
}

static inline void
level2_decrement_resize_counter(iceberg_table *table)
{
  uint64_t counter_value_pre =
    __atomic_fetch_sub(&table->level2_resize_counter, 1, __ATOMIC_SEQ_CST);
  assert(counter_value_pre != 0);
}

static inline bool
level2_key_should_move(iceberg_table *table, partition_block key_pb, partition_block new_block_pb)
{
  return key_pb.partition == new_block_pb.partition && key_pb.block == new_block_pb.block;
}

static inline bool
level2_insert_into_block_with_mask(iceberg_table  *table,
                                   iceberg_key_t   key,
                                   iceberg_value_t value,
                                   hash           *h,
                                   partition_block pb,
                                   uint8_t        *sketch,
                                   uint32_t        match_mask,
                                   uint64_t        tid)
{
  while (match_mask != 0) {
    uint64_t slot = __builtin_ctzll(match_mask);
    match_mask    = match_mask & ~(1ULL << slot);

    if (__sync_bool_compare_and_swap(&sketch[slot], 0, 1)) {
      counter_increment(&table->num_items_per_level, LEVEL2, tid);
      kv_pair *kv = get_level2_kv_pair(table, pb, slot);
      verbose_print_location(2, pb.partition, pb.block, slot, kv);
      atomic_write_128(key, value, kv);
      sketch[slot] = h->fingerprint;
      verbose_print_sketch(sketch, 8);
      return true;
    }
  }

  return false;
}

static inline bool
level2_insert_into_block(iceberg_table  *table,
                         iceberg_key_t   key,
                         iceberg_value_t value,
                         hash           *h,
                         partition_block pb,
                         uint64_t        tid)
{
  fingerprint_t *sketch     = get_level2_sketch(table, pb);
  uint32_t       match_mask = sketch_match_8(sketch, 0);
  verbose_print_sketch(sketch, 8);
  verbose_print_mask_8(match_mask);

  return level2_insert_into_block_with_mask(
    table, key, value, h, pb, sketch, match_mask, tid);
}

static inline void
level2_move_block(iceberg_table *table, partition_block pb, uint64_t tid)
{
  partition_block new_block_pb = compute_new_partition_block(table, pb);
  fingerprint_t *sketch = get_level2_sketch(table, pb);
  for (uint64_t i = 0; i < LEVEL2_BLOCK_SIZE; i++) {
    kv_pair *kv = get_level2_kv_pair(table, pb, i);
    if (kv->key == KEY_FREE) {
      continue;
    }

    hash h = hash_key(table, &kv->key);

    partition_block key_pb1 = get_block(table, &h, LEVEL2_BLOCK1);
    if (level2_key_should_move(table, key_pb1, new_block_pb)) {
      bool ret = level2_insert_into_block(table, kv->key, kv->val, &h, key_pb1, tid);
      assert(ret);
      delete_from_slot(table, kv, sketch, i, LEVEL2, tid);
      continue;
    }

    partition_block key_pb2 = get_block(table, &h, LEVEL2_BLOCK2);
    if (level2_key_should_move(table, key_pb2, new_block_pb)) {
      bool ret = level2_insert_into_block(table, kv->key, kv->val, &h, key_pb2, tid);
      assert(ret);
      delete_from_slot(table, kv, sketch, i, LEVEL2, tid);
    }
  }

  level2_decrement_resize_counter(table);
}

static inline uint64_t
level1_num_chunks(iceberg_table *table)
{
  return table->num_blocks / LEVEL1_BLOCKS_PER_RESIZE_CHUNK;
}

static inline bool
level1_try_set_marker(iceberg_table *table, partition_block chunk_pb)
{
  uint8_t *marker = &table->level1_resize_marker[chunk_pb.partition][chunk_pb.block];
  return __sync_lock_test_and_set(marker, 1);
}

static inline void
level1_set_marker(iceberg_table *table, partition_block chunk_pb)
{
  uint8_t *marker = &table->level1_resize_marker[chunk_pb.partition][chunk_pb.block];
  bool old_marker = __sync_lock_test_and_set(marker, 1);
  assert(old_marker == 0);
}

static inline partition_block
level1_chunk_to_block(partition_block chunk_pb, uint64_t offset)
{
  partition_block pb = { .partition = chunk_pb.partition, .block = chunk_pb.block * LEVEL1_BLOCKS_PER_RESIZE_CHUNK + offset };
  return pb;
}

static inline void
level1_maybe_move_chunk(iceberg_table *table, partition_block chunk_pb, uint64_t tid)
{
  if (level1_try_set_marker(table, chunk_pb)) {
    for (uint64_t i = 0; i < LEVEL1_BLOCKS_PER_RESIZE_CHUNK; i++) {
      partition_block pb = level1_chunk_to_block(chunk_pb, i);
      level1_move_block(table, pb, tid);
    }
    partition_block new_chunk_pb = level1_compute_new_partition_chunk(table, chunk_pb);
    level1_set_marker(table, new_chunk_pb);
  }
}

static inline uint64_t
level2_num_chunks(iceberg_table *table)
{
  return table->num_blocks / LEVEL2_BLOCKS_PER_RESIZE_CHUNK;
}

static inline bool
level2_try_set_marker(iceberg_table *table, partition_block chunk_pb)
{
  uint8_t *marker = &table->level2_resize_marker[chunk_pb.partition][chunk_pb.block];
  return __sync_lock_test_and_set(marker, 1);
}

static inline void
level2_set_marker(iceberg_table *table, partition_block chunk_pb)
{
  uint8_t *marker = &table->level2_resize_marker[chunk_pb.partition][chunk_pb.block];
  bool old_marker = __sync_lock_test_and_set(marker, 1);
  assert(old_marker == 0);
}

static inline partition_block
level2_chunk_to_block(partition_block chunk_pb, uint64_t offset)
{
  partition_block pb = { .partition = chunk_pb.partition, .block = chunk_pb.block * LEVEL2_BLOCKS_PER_RESIZE_CHUNK + offset };
   return pb;
}

static inline void
level2_maybe_move_chunk(iceberg_table *table, partition_block chunk_pb, uint64_t tid)
{
  if (level2_try_set_marker(table, chunk_pb)) {
    for (uint64_t i = 0; i < LEVEL2_BLOCKS_PER_RESIZE_CHUNK; i++) {
      partition_block pb = level2_chunk_to_block(chunk_pb, i);
      level2_move_block(table, pb, tid);
    }
    partition_block new_chunk_pb = level2_compute_new_partition_chunk(table, chunk_pb);
    level2_set_marker(table, new_chunk_pb);
  }

}

void
iceberg_end(iceberg_table *table, uint64_t tid)
{
  if (level1_resize_active(table)) {
    uint64_t num_chunks = level1_num_chunks(table);
    for (uint64_t chunk_num = 0; chunk_num < num_chunks; chunk_num++) {
      partition_block chunk_pb = level1_decode_raw_chunk(table, chunk_num);
      level1_maybe_move_chunk(table, chunk_pb, tid);
    }
  }

  if (level2_resize_active(table)) {
    uint64_t num_chunks = level2_num_chunks(table);
    for (uint64_t chunk_num = 0; chunk_num < num_chunks; chunk_num++) {
      partition_block chunk_pb = level2_decode_raw_chunk(table, chunk_num);
      level2_maybe_move_chunk(table, chunk_pb, tid);
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
level3_insert(iceberg_table  *table,
              iceberg_key_t   key,
              iceberg_value_t value,
              hash           *h,
              uint64_t        tid)
{
  uint64_t block = get_level3_block(h);
  level3_lock_block(table, block);
  iceberg_level3_node *new_node = malloc(sizeof(iceberg_level3_node));
  assert(new_node);
  verbose_print_location(3, 0, block, 0, new_node);
  new_node->key             = key;
  new_node->val             = value;
  new_node->next_node       = table->level3[block].head;
  table->level3[block].head = new_node;
  counter_increment(&table->num_items_per_level, LEVEL3, tid);
  level3_unlock_block(table, block);
  return true;
}

static inline partition_block
level2_block_to_chunk(partition_block pb)
{
  partition_block chunk_pb = { .partition = pb.partition, .block = pb.block / LEVEL2_BLOCKS_PER_RESIZE_CHUNK };
  return chunk_pb;
}

static inline void
level2_maybe_move(iceberg_table *table, partition_block pb1, partition_block pb2, uint64_t tid)
{
  if (likely(!level2_resize_active(table))) {
    return;
  }

  if (pb1.partition != table->num_partitions - 1) {
     partition_block chunk_pb = level2_block_to_chunk(pb1);
     level2_maybe_move_chunk(table, chunk_pb, tid);
  }

  if (pb2.partition != table->num_partitions - 1) {
     partition_block chunk_pb = level2_block_to_chunk(pb2);
     level2_maybe_move_chunk(table, chunk_pb, tid);
  }
}

static inline bool
level2_insert(iceberg_table  *table,
              iceberg_key_t   key,
              iceberg_value_t value,
              hash           *h,
              uint64_t        tid)
{

  partition_block pb1     = get_block(table, h, LEVEL2_BLOCK1);
  fingerprint_t  *sketch1 = get_level2_sketch(table, pb1);
  verbose_print_sketch(sketch1, LEVEL2_BLOCK_SIZE);

  partition_block pb2     = get_block(table, h, LEVEL2_BLOCK2);
  fingerprint_t  *sketch2 = get_level2_sketch(table, pb2);
  verbose_print_sketch(sketch2, LEVEL2_BLOCK_SIZE);

  level2_maybe_move(table, pb1, pb2, tid);

  uint32_t match_mask = double_sketch_match_8(sketch1, sketch2, FP_FREE);
  verbose_print_double_mask_8(match_mask);
  uint8_t popcnt1 = __builtin_popcount(match_mask & LEVEL2_CHOICE1_MASK);
  uint8_t popcnt2 = __builtin_popcount(match_mask & LEVEL2_CHOICE2_MASK);

  if (popcnt2 > popcnt1) {
    pb1       = pb2;
    sketch1   = sketch2;
    match_mask >>= 8;
    popcnt1 = popcnt2;
  }


  if (level2_insert_into_block_with_mask(
        table, key, value, h, pb1, sketch1, match_mask, tid)) {
    return true;
  }

  return level3_insert(table, key, value, h, tid);
}

static inline bool iceberg_query_internal(iceberg_table   *table,
                                          iceberg_key_t    key,
                                          iceberg_value_t *value,
                                          hash            *h,
                                          uint64_t         tid);


static inline partition_block
level1_block_to_chunk(partition_block pb)
{
  partition_block chunk_pb = { .partition = pb.partition, .block = pb.block / LEVEL1_BLOCKS_PER_RESIZE_CHUNK };
  return chunk_pb;
}

static inline void
level1_maybe_move(iceberg_table *table, partition_block pb, uint64_t tid)
{
  if (likely(!level1_resize_active(table))) {
    return;
  }

  if (pb.partition != table->num_partitions - 1) {
     partition_block chunk_pb = level1_block_to_chunk(pb);
     level1_maybe_move_chunk(table, chunk_pb, tid);
  }
}

__attribute__((always_inline)) inline bool
iceberg_insert(iceberg_table  *table,
               iceberg_key_t   key,
               iceberg_value_t value,
               uint64_t        tid)
{
  verbose_print_operation("INSERT:", key, value);

  maybe_create_new_partition(table, tid);

  hash h = hash_key(table, &key);
  partition_block pb     = get_block(table, &h, LEVEL1_BLOCK);

  level1_maybe_move(table, pb, tid);

  fingerprint_t  *sketch = get_level1_sketch(table, pb);
  lock_block(sketch);
  verbose_print_sketch(sketch, 64);

  iceberg_value_t v;
  verbose_print_operation("INTERNAL QUERY:", key, value);
  bool ret = true;
  if (unlikely(iceberg_query_internal(table, key, &v, &h, tid))) {
    ret = true;
    goto out;
  }
  verbose_end("INTERNAL QUERY", true);

  ret = level1_insert_into_block(table, key, value, &h, pb, tid);
  if (!ret) {
    ret = level2_insert(table, key, value, &h, tid);
  }

out:
  verbose_end("INSERT", false);
  unlock_block(sketch);
  return ret;
}

static bool
level3_delete(iceberg_table *table, iceberg_key_t key, hash *h, uint64_t tid)
{
  uint64_t block = get_level3_block(h);

  if (likely(table->level3[block].head == NULL)) {
    return false;
  }

  bool ret = false;
  level3_lock_block(table, block);
  iceberg_level3_node **node = &table->level3[block].head;

  while (*node != NULL) {
    if ((*node)->key == key) {
      iceberg_level3_node *dead_node = *node;
      *node                          = (*node)->next_node;
      free(dead_node);
      counter_decrement(&table->num_items_per_level, LEVEL3, tid);
      ret = true;
      goto unlock_out;
    }
    node = &(*node)->next_node;
  }

unlock_out:
  level3_unlock_block(table, block);
  return ret;
}

static inline bool
level2_delete_from_block(iceberg_table *table, iceberg_key_t key, hash *h, partition_block pb, uint64_t tid)
{
   fingerprint_t *sketch     = get_level2_sketch(table, pb);
   uint32_t       match_mask = sketch_match_8(sketch, h->fingerprint);
   verbose_print_sketch(sketch, 8);
   verbose_print_mask_8(match_mask);
   uint8_t popcnt = __builtin_popcount(match_mask);

   for (uint8_t i = 0; i < popcnt; ++i) {
      uint64_t slot_num = word_select(match_mask, i);

      kv_pair *candidate_kv = get_level2_kv_pair(table, pb, slot_num);
      if (candidate_kv->key == key) {
         delete_from_slot(table, candidate_kv, sketch, slot_num, LEVEL2, tid);
         return true;
      }
   }

   return false;
}

static inline bool
level2_delete(iceberg_table *table, iceberg_key_t key, hash *h, uint64_t tid)
{
  partition_block pb1 = get_block(table, h, LEVEL2_BLOCK1);
  partition_block pb2 = get_block(table, h, LEVEL2_BLOCK2);
  level2_maybe_move(table, pb1, pb2, tid);

  bool ret = level2_delete_from_block(table, key, h, pb1, tid);
  if (ret) {
    return ret;
  }

  ret = level2_delete_from_block(table, key, h, pb2, tid);
  if (ret) {
    return ret;
  }

  return level3_delete(table, key, h, tid);
}

__attribute__((always_inline)) inline bool
iceberg_delete(iceberg_table *table, iceberg_key_t key, uint64_t tid)
{
  verbose_print_operation("DELETE:", key, 0);
  bool ret = true;

  hash            h  = hash_key(table, &key);
  partition_block pb = get_block(table, &h, LEVEL1_BLOCK);

  level1_maybe_move(table, pb, tid);

  fingerprint_t *sketch = get_level1_sketch(table, pb);
  lock_block(sketch);
  verbose_print_sketch(sketch, 64);
  uint64_t match_mask = sketch_match_64(sketch, h.fingerprint);
  verbose_print_mask_64(match_mask);
  uint8_t popcnt = __builtin_popcountll(match_mask);

  for (uint8_t i = 0; i < popcnt; ++i) {
    uint64_t slot = word_select(match_mask, i);

    kv_pair *candidate_kv = get_level1_kv_pair(table, pb, slot);
    if (candidate_kv->key == key) {
      verbose_print_location(1, pb.partition, pb.block, slot, candidate_kv);
      candidate_kv->key = 0;
      candidate_kv->val = 0;
      counter_decrement(&table->num_items_per_level, LEVEL1, tid);
      sketch[slot] = 0;
      verbose_print_sketch(sketch, 64);
      goto unlock_out;
    }
  }

  ret = level2_delete(table, key, &h, tid);

unlock_out:
  unlock_block(sketch);
  goto out;
out:
  verbose_end("DELETE", false);
  return ret;
}

static inline bool
level3_query(iceberg_table   *table,
                      iceberg_key_t    key,
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
  iceberg_level3_node *node = table->level3[block].head;
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
level2_query_block(iceberg_table *table, iceberg_key_t key, iceberg_value_t *value, hash *h, partition_block pb)
{
    fingerprint_t *sketch     = get_level2_sketch(table, pb);
    uint32_t       match_mask = sketch_match_8(sketch, h->fingerprint);
    verbose_print_sketch(sketch, 8);
    verbose_print_mask_8(match_mask);

    while (match_mask != 0) {
      int slot   = __builtin_ctz(match_mask);
      match_mask = match_mask & ~(1U << slot);

      kv_pair *candidate_kv = get_level2_kv_pair(table, pb, slot);
      if (candidate_kv->key == key) {
        verbose_print_location(2, pb.partition, pb.block, slot, candidate_kv);
        *value = candidate_kv->val;
        return true;
      }
    }
    return false;
}

static inline bool
block_is_new(iceberg_table *table, partition_block pb)
{
  return pb.partition == table->num_partitions - 1;
}

static inline partition_block
get_old_block(iceberg_table *table, partition_block pb)
{
  return decode_raw_block(table, pb.block);
}

static inline bool
level2_maybe_query_old_block(iceberg_table *table, iceberg_key_t key, iceberg_value_t *value, hash *h, partition_block pb)
{
  if (likely(!level2_resize_active(table) || !block_is_new(table, pb))) {
    return false;
  }
  partition_block old_pb = get_old_block(table, pb);
  return level2_query_block(table, key, value, h, old_pb);
}

static inline bool
level2_query(iceberg_table   *table,
             iceberg_key_t  key,
             iceberg_value_t *value,
             hash            *h)
{
  partition_block pb1 = get_block(table, h, LEVEL2_BLOCK1);
  bool ret = level2_maybe_query_old_block(table, key, value, h, pb1);
  if (ret) {
    return ret;
  }
  ret = level2_query_block(table, key, value, h, pb1);
  if (ret) {
     return ret;
  }

  partition_block pb2 = get_block(table, h, LEVEL2_BLOCK2);
  ret = level2_maybe_query_old_block(table, key, value, h, pb1);
  if (ret) {
    return ret;
  }
  ret = level2_query_block(table, key, value, h, pb2);
  if (ret) {
     return ret;
  }

  return level3_query(table, key, value, h);
}

static inline bool
level1_query_block(iceberg_table *table, iceberg_key_t key, iceberg_value_t *value, hash *h, partition_block pb)
{
  fingerprint_t  *sketch = get_level1_sketch(table, pb);
  verbose_print_sketch(sketch, 64);
  uint64_t match_mask = sketch_match_64(sketch, h->fingerprint);
  verbose_print_mask_64(match_mask);

  while (match_mask != 0) {
    int slot   = __builtin_ctzll(match_mask);
    match_mask = match_mask & ~(1ULL << slot);

    kv_pair *candidate_kv = get_level1_kv_pair(table, pb, slot);
    if (candidate_kv->key == key) {
      verbose_print_location(1, pb.partition, pb.block, slot, candidate_kv);
      *value = candidate_kv->val;
      return true;
    }
  }
  return false;
}

static inline bool
level1_maybe_query_old_block(iceberg_table *table, iceberg_key_t key, iceberg_value_t *value, hash *h, partition_block pb)
{
  if (likely(!level1_resize_active(table) || !block_is_new(table, pb))) {
    return false;
  }
  partition_block old_pb = get_old_block(table, pb);
  return level1_query_block(table, key, value, h, old_pb);
}

__attribute__((always_inline)) inline bool
iceberg_query_internal(iceberg_table   *table,
                       iceberg_key_t    key,
                       iceberg_value_t *value,
                       hash            *h,
                       uint64_t         tid)
{
  partition_block pb     = get_block(table, h, LEVEL1_BLOCK);

  level1_maybe_query_old_block(table, key, value, h, pb);

  bool ret = level1_query_block(table, key, value, h, pb);
  if (ret) {
    goto out;
  }

  ret = level2_query(table, key, value, h);

out:
  return ret;
}

__attribute__((always_inline)) inline bool
iceberg_query(iceberg_table   *table,
              iceberg_key_t    key,
              iceberg_value_t *value,
              uint64_t         tid)
{
  verbose_print_operation("QUERY:", key, 0);
  bool ret = true;

  hash h = hash_key(table, &key);
  ret    = iceberg_query_internal(table, key, value, &h, tid);
  verbose_end("QUERY", false);
  return ret;
}

