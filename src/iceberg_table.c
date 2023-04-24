#define _GNU_SOURCE
#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <stdlib.h>
#include <pthread.h>
#include <immintrin.h>
#include <tmmintrin.h>
#include <sys/mman.h>
#include <math.h>

#include "xxhash.h"
#include "iceberg_precompute.h"
#include "iceberg_table.h"
#include "verbose.h"

#define likely(x)   __builtin_expect((x),1)
#define unlikely(x) __builtin_expect((x),0)

#define RESIZE_THRESHOLD 0.96
/*#define RESIZE_THRESHOLD 0.85 // For YCSB*/

#define MAX_PROCS 64

#if __linux__
#include <linux/version.h>
#if LINUX_VERSION_CODE > KERNEL_VERSION(2,6,22)
#define _MAP_POPULATE_AVAILABLE
#endif
#endif

#ifdef _MAP_POPULATE_AVAILABLE
#ifdef _MAP_HUGETLB_AVAILABLE
#define MMAP_FLAGS (MAP_PRIVATE | MAP_ANONYMOUS | MAP_POPULATE | MAP_HUGE_TLB)
#else // _MAP_HUGETLB_AVAILABLE
#define MMAP_FLAGS (MAP_PRIVATE | MAP_ANONYMOUS | MAP_POPULATE)
#endif // _MAP_HUGETLB_AVAILABLE
#else // _MAP_POPULATE_AVAILABLE
#define MMAP_FLAGS (MAP_PRIVATE | MAP_ANONYMOUS)
#endif

uint64_t seed = 12351327692179052ll;

static inline uint8_t word_select(uint64_t val, int rank) {
  val = _pdep_u64(one[rank], val);
  return _tzcnt_u64(val);
}

uint64_t lv1_balls(iceberg_table * table) {
  pc_sync(&table->metadata.lv1_balls);
  return *(table->metadata.lv1_balls.global_counter);
}

static inline uint64_t lv1_balls_aprox(iceberg_table * table) {
  return *(table->metadata.lv1_balls.global_counter);
}

uint64_t lv2_balls(iceberg_table * table) {
  pc_sync(&table->metadata.lv2_balls);
  return *(table->metadata.lv2_balls.global_counter);
}

static inline uint64_t lv2_balls_aprox(iceberg_table * table) {
  return *(table->metadata.lv2_balls.global_counter);
}

uint64_t lv3_balls(iceberg_table * table) {
  pc_sync(&table->metadata.lv3_balls);
  return *(table->metadata.lv3_balls.global_counter);
}

static inline uint64_t lv3_balls_aprox(iceberg_table * table) {
  return *(table->metadata.lv3_balls.global_counter);
}

uint64_t tot_balls(iceberg_table * table) {
  return lv1_balls(table) + lv2_balls(table) + lv3_balls(table);
}

uint64_t tot_balls_aprox(iceberg_table * table) {
  return lv1_balls_aprox(table) + lv2_balls_aprox(table) + lv3_balls_aprox(table);
}

static inline uint64_t
level1_slots_per_block()
{
  return 1ULL << LEVEL1_BLOCK_WIDTH;
}

static inline uint64_t
level2_slots_per_block()
{
  return 1ULL << LEVEL2_BLOCK_WIDTH;
}

static inline uint64_t total_capacity(iceberg_table * table) {
  return table->metadata.nblocks * (level1_slots_per_block() + level2_slots_per_block());
}

inline double iceberg_load_factor(iceberg_table * table) {
  return (double)tot_balls(table) / (double)total_capacity(table);
}

#ifdef ENABLE_RESIZE
static inline bool
need_resize(iceberg_table * table) {
  return tot_balls_aprox(table) >= table->metadata.resize_threshold;
}
#endif

typedef struct __attribute__ (( packed )) {
  uint64_t level1_raw_block  : 40;
  uint64_t level2_raw_block1 : 40;
  uint64_t level2_raw_block2 : 40;
  uint8_t  fingerprint   : 8;
} raw_hash;

_Static_assert(sizeof(raw_hash) == 16, "hash not 16B\n");

typedef enum {
  LEVEL1 = 0,
  LEVEL2_BLOCK1,
  LEVEL2_BLOCK2,
  NUM_LEVELS,
} level;

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
  return raw_block & ((1 << table->metadata.block_bits) - 1);
}

static inline hash
hash_key(iceberg_table *table, KeyType *key) {
  hash h = { 0 };
  XXH128_hash_t *raw = (XXH128_hash_t *)&h.raw;
  *raw = XXH128(key, sizeof(*key), seed);
  h.fingerprint = ensure_nonzero_fingerprint(&h.raw);
  h.raw_block[LEVEL1] = truncate_to_current_num_raw_blocks(table, h.raw.level1_raw_block);
  h.raw_block[LEVEL2_BLOCK1] = truncate_to_current_num_raw_blocks(table, h.raw.level2_raw_block1);
  h.raw_block[LEVEL2_BLOCK2] = truncate_to_current_num_raw_blocks(table, h.raw.level2_raw_block2);
  return h;
}

typedef struct {
  uint64_t partition;
  uint64_t block;
} partition_block;

static inline partition_block
decode_raw_internal(uint64_t init_log, uint64_t raw_block)
{
  partition_block pb = { 0 };
  uint64_t shf = raw_block >> init_log;
  pb.partition = 64 - _lzcnt_u64(shf);
  uint64_t adj = 1ULL << pb.partition;
  adj = adj >> 1;
  adj = adj << init_log;
  pb.block = raw_block - adj;
  return pb;
}

static inline partition_block
decode_raw_block(iceberg_table *table, uint64_t raw_block)
{
   return decode_raw_internal(table->metadata.log_init_size, raw_block);
}

static inline partition_block
get_block(iceberg_table *table, hash *h, level lvl)
{
  return decode_raw_block(table, h->raw_block[lvl]);
}

static inline uint64_t
get_level3_block(hash *h)
{
  return h->raw_block[LEVEL1] % LEVEL3_BLOCKS;
}

static inline partition_block
decode_raw_chunk(iceberg_table *table, uint64_t raw_chunk)
{
   return decode_raw_internal(table->metadata.log_init_size - 3, raw_chunk);
}

#define LOCK_MASK 1ULL
#define UNLOCK_MASK ~1ULL

static inline void lock_block(uint64_t * metadata)
{
#ifdef ENABLE_BLOCK_LOCKING
  uint64_t *data = metadata + 7;
  while ((__sync_fetch_and_or(data, LOCK_MASK) & 1) != 0) {
    _mm_pause();
  }
#endif
}

static inline void unlock_block(uint64_t * metadata)
{
#ifdef ENABLE_BLOCK_LOCKING
  uint64_t *data = metadata + 7;
   *data = *data & UNLOCK_MASK;
#endif
}

static inline uint32_t slot_mask_32(uint8_t * metadata, uint8_t fprint) {
  __m256i bcast = _mm256_set1_epi8(fprint);
  __m256i block = _mm256_loadu_si256((const __m256i *)(metadata));
#if defined __AVX512BW__ && defined __AVX512VL__
  return _mm256_cmp_epi8_mask(bcast, block, _MM_CMPINT_EQ);
#else
  __m256i cmp = _mm256_cmpeq_epi8(bcast, block);
  return _mm256_movemask_epi8(cmp);
#endif
}

#if defined __AVX512F__ && defined __AVX512BW__
static inline uint64_t slot_mask_64(uint8_t * metadata, uint8_t fprint) {
  __m512i mask = _mm512_loadu_si512((const __m512i *)(broadcast_mask));
  __m512i bcast = _mm512_set1_epi8(fprint);
  bcast = _mm512_or_epi64(bcast, mask);
  __m512i block = _mm512_loadu_si512((const __m512i *)(metadata));
  block = _mm512_or_epi64(block, mask);
  return _mm512_cmp_epi8_mask(bcast, block, _MM_CMPINT_EQ);
}
#else /* ! (defined __AVX512F__ && defined __AVX512BW__) */
static inline uint32_t slot_mask_64_half(__m256i fprint, __m256i md, __m256i mask)
{
  __m256i masked_fp = _mm256_or_si256(fprint, mask);
  __m256i masked_md = _mm256_or_si256(md, mask);
  __m256i cmp       = _mm256_cmpeq_epi8(masked_md, masked_fp);
  return _mm256_movemask_epi8(cmp);
}

static inline uint64_t slot_mask_64(uint8_t * metadata, uint8_t fp) {
  __m256i fprint   = _mm256_set1_epi8(fp);

  __m256i  md1     = _mm256_loadu_si256((const __m256i *)(metadata));
  __m256i  mask1   = _mm256_loadu_si256((const __m256i *)(broadcast_mask));
  uint64_t result1 = slot_mask_64_half(fprint, md1, mask1);

  __m256i  md2     = _mm256_loadu_si256((const __m256i *)(&metadata[32]));
  __m256i  mask2   = _mm256_loadu_si256((const __m256i *)(&broadcast_mask[32]));
  uint64_t result2 = slot_mask_64_half(fprint, md2, mask2);

  return ((uint64_t)result2 << 32) | result1;
}
#endif /* ! (defined __AVX512F__ && defined __AVX512BW__) */


static inline void
atomic_write_128(uint64_t key, uint64_t val, kv_pair *kv) {
  uint64_t arr[2] = {key, val};
  __m128d a =  _mm_load_pd((double *)arr);
  _mm_store_pd((double*)kv, a);
}

static inline uint64_t
kv_pair_offset(partition_block pb, uint64_t block_size, uint64_t slot_in_block)
{
  return block_size * pb.block + slot_in_block;
}

static inline kv_pair *
level1_kv_pair(iceberg_table *table, partition_block pb, uint64_t slot_in_block)
{
   uint64_t slot = kv_pair_offset(pb, level1_slots_per_block(), slot_in_block);
   return &table->level1[pb.partition][slot];
}

static inline kv_pair *
level2_kv_pair(iceberg_table *table, partition_block pb, uint64_t slot_in_block)
{
   uint64_t slot = kv_pair_offset(pb, level2_slots_per_block(), slot_in_block);
   return &table->level2[pb.partition][slot];
}

static inline uint64_t level1_blocks_to_size(uint64_t blocks)
{
   return blocks * level1_slots_per_block() * sizeof(kv_pair);
}

static inline uint64_t level2_blocks_to_size(uint64_t blocks)
{
   return blocks * level2_slots_per_block() * sizeof(kv_pair);
}

int iceberg_init(iceberg_table *table, uint64_t log_slots) {
  memset(table, 0, sizeof(*table));

  uint64_t total_blocks = 1 << (log_slots - LEVEL1_BLOCK_WIDTH);
  uint64_t level1_size = level1_blocks_to_size(total_blocks);
  uint64_t level2_size = level2_blocks_to_size(total_blocks);
  uint64_t total_size_in_bytes = level1_size + level2_size + (sizeof(iceberg_lv1_block_md) + sizeof(iceberg_lv2_block_md)) * total_blocks;

  assert(table);

  table->level1[0] = (kv_pair *)mmap(NULL, level1_size, PROT_READ | PROT_WRITE, MMAP_FLAGS, 0, 0);
  if (!table->level1[0]) {
    perror("level1 malloc failed");
    exit(1);
  }
  table->level2[0] = (kv_pair *)mmap(NULL, level2_size, PROT_READ | PROT_WRITE, MMAP_FLAGS, 0, 0);
  if (!table->level2[0]) {
    perror("level2 malloc failed");
    exit(1);
  }
  size_t level3_size = sizeof(iceberg_lv3_list) * LEVEL3_BLOCKS;
  table->level3 = (iceberg_lv3_list *)mmap(NULL, level3_size, PROT_READ | PROT_WRITE, MMAP_FLAGS, 0, 0);
  if (!table->level3) {
    perror("level3 malloc failed");
    exit(1);
  }

  table->metadata.total_size_in_bytes = total_size_in_bytes;
  table->metadata.nslots = 1 << log_slots;
  table->metadata.nblocks = total_blocks;
  table->metadata.block_bits = log_slots - LEVEL1_BLOCK_WIDTH;
  table->metadata.init_size = total_blocks;
  table->metadata.log_init_size = log2(total_blocks);
  table->metadata.nblocks_parts[0] = total_blocks;
  table->metadata.resize_threshold = RESIZE_THRESHOLD * total_capacity(table);

  pc_init(&table->metadata.lv1_balls, &table->metadata.lv1_ctr, MAX_PROCS, 1000);
  pc_init(&table->metadata.lv2_balls, &table->metadata.lv2_ctr, MAX_PROCS, 1000);
  pc_init(&table->metadata.lv3_balls, &table->metadata.lv3_ctr, MAX_PROCS, 1000);

  size_t lv1_md_size = sizeof(iceberg_lv1_block_md) * total_blocks + 64;
  table->metadata.lv1_md[0] = (iceberg_lv1_block_md *)mmap(NULL, lv1_md_size, PROT_READ | PROT_WRITE, MMAP_FLAGS, 0, 0);
  if (!table->metadata.lv1_md[0]) {
    perror("lv1_md malloc failed");
    exit(1);
  }
  size_t lv2_md_size = sizeof(iceberg_lv2_block_md) * total_blocks + 32;
  table->metadata.lv2_md[0] = (iceberg_lv2_block_md *)mmap(NULL, lv2_md_size, PROT_READ | PROT_WRITE, MMAP_FLAGS, 0, 0);
  if (!table->metadata.lv2_md[0]) {
    perror("lv2_md malloc failed");
    exit(1);
  }
  table->metadata.lv3_sizes = (uint64_t *)mmap(NULL, sizeof(uint64_t) * LEVEL3_BLOCKS, PROT_READ | PROT_WRITE, MMAP_FLAGS, 0, 0);
  if (!table->metadata.lv3_sizes) {
    perror("lv3_sizes malloc failed");
    exit(1);
  }
  table->metadata.lv3_locks = (uint8_t *)mmap(NULL, sizeof(uint8_t) * LEVEL3_BLOCKS, PROT_READ | PROT_WRITE, MMAP_FLAGS, 0, 0);
  if (!table->metadata.lv3_locks) {
    perror("lv3_locks malloc failed");
    exit(1);
  }

#ifdef ENABLE_RESIZE
  table->metadata.resize_cnt = 0;
  table->metadata.lv1_resize_ctr = 0;
  table->metadata.lv2_resize_ctr = 0;

  // create one marker for 8 blocks.
  size_t resize_marker_size = sizeof(uint8_t) * total_blocks / 8;
  table->metadata.lv1_resize_marker[0] = (uint8_t *)mmap(NULL, resize_marker_size, PROT_READ | PROT_WRITE, MMAP_FLAGS, 0, 0);
  if (!table->metadata.lv1_resize_marker[0]) {
    perror("level1 resize ctr malloc failed");
    exit(1);
  }
  table->metadata.lv2_resize_marker[0] = (uint8_t *)mmap(NULL, resize_marker_size, PROT_READ | PROT_WRITE, MMAP_FLAGS, 0, 0);
  if (!table->metadata.lv2_resize_marker[0]) {
    perror("level2 resize ctr malloc failed");
    exit(1);
  }

  table->metadata.marker_sizes[0] = resize_marker_size;
  table->metadata.lock = 0;
#endif

  return 0;
}

#ifdef ENABLE_RESIZE
static inline bool is_lv1_resize_active(iceberg_table * table) {
  return __atomic_load_n(&table->metadata.lv1_resize_ctr, __ATOMIC_SEQ_CST);
}

static inline bool is_lv2_resize_active(iceberg_table * table) {
  return __atomic_load_n(&table->metadata.lv2_resize_ctr, __ATOMIC_SEQ_CST);
}

static bool is_resize_active(iceberg_table * table) {
  return is_lv2_resize_active(table) || is_lv1_resize_active(table);
}

static bool iceberg_setup_resize(iceberg_table * table) {
  // grab write lock
  if (!lock(&table->metadata.lock, TRY_ONCE_LOCK))
    return false;

  if (unlikely(!need_resize(table))) {
    unlock(&table->metadata.lock);
    return false;
  }
  if (is_resize_active(table)) {
    // finish the current resize
    iceberg_end(table);
    unlock(&table->metadata.lock);
    return false;
  }

  /*printf("Setting up resize\n");*/
  /*printf("Current stats: \n");*/

  /*printf("Load factor: %f\n", iceberg_load_factor(table));*/
  /*printf("Number level 1 inserts: %ld\n", lv1_balls(table));*/
  /*printf("Number level 2 inserts: %ld\n", lv2_balls(table));*/
  /*printf("Number level 3 inserts: %ld\n", lv3_balls(table));*/
  /*printf("Total inserts: %ld\n", tot_balls(table));*/

  // compute new sizes
  uint64_t cur_blocks = table->metadata.nblocks;
  uint64_t resize_cnt = table->metadata.resize_cnt + 1;

  // Allocate new table and metadata
  // alloc level1
  size_t level1_size = level1_blocks_to_size(cur_blocks);
  table->level1[resize_cnt] = (kv_pair *)mmap(NULL, level1_size, PROT_READ | PROT_WRITE, MMAP_FLAGS, 0, 0);
  if (table->level1[resize_cnt] == (void *)-1) {
    perror("level1 resize failed");
    exit(1);
  }

  // alloc level2
  size_t level2_size = level2_blocks_to_size(cur_blocks);
  table->level2[resize_cnt] = (kv_pair *)mmap(NULL, level2_size, PROT_READ | PROT_WRITE, MMAP_FLAGS, 0, 0);
  if (table->level2[resize_cnt] == (void *)-1) {
    perror("level2 resize failed");
    exit(1);
  }

#endif

  // alloc level1 metadata
  size_t lv1_md_size = sizeof(iceberg_lv1_block_md) * cur_blocks + 64;
  table->metadata.lv1_md[resize_cnt] = (iceberg_lv1_block_md *)mmap(NULL, lv1_md_size, PROT_READ | PROT_WRITE, MMAP_FLAGS, 0, 0);
  if (table->metadata.lv1_md[resize_cnt] == (void *)-1) {
    perror("lv1_md resize failed");
    exit(1);
  }

  // alloc level2 metadata
  size_t lv2_md_size = sizeof(iceberg_lv2_block_md) * cur_blocks + 32;
  table->metadata.lv2_md[resize_cnt] = (iceberg_lv2_block_md *)mmap(NULL, lv2_md_size, PROT_READ | PROT_WRITE, MMAP_FLAGS, 0, 0);
  if (table->metadata.lv2_md[resize_cnt] == (void *)-1) {
    perror("lv2_md resize failed");
    exit(1);
  }

  // alloc resize markers
  // resize_marker_size
  size_t resize_marker_size = sizeof(uint8_t) * cur_blocks / 8;
  table->metadata.lv1_resize_marker[resize_cnt] = (uint8_t *)mmap(NULL, resize_marker_size, PROT_READ | PROT_WRITE, MMAP_FLAGS, 0, 0);
  if (table->metadata.lv1_resize_marker[resize_cnt] == (void *)-1) {
    perror("level1 resize failed");
    exit(1);
  }

  table->metadata.lv2_resize_marker[resize_cnt] = (uint8_t *)mmap(NULL, resize_marker_size, PROT_READ | PROT_WRITE, MMAP_FLAGS, 0, 0);
  if (table->metadata.lv2_resize_marker[resize_cnt] == (void *)-1) {
    perror("level1 resize failed");
    exit(1);
  }

  table->metadata.marker_sizes[resize_cnt] = resize_marker_size;
  // resetting the resize markers.
  for (uint64_t i = 0;  i <= resize_cnt; ++i) {
    memset(table->metadata.lv1_resize_marker[i], 0, table->metadata.marker_sizes[i]);
    memset(table->metadata.lv2_resize_marker[i], 0, table->metadata.marker_sizes[i]);
  }

  uint64_t total_blocks = table->metadata.nblocks * 2;
  uint64_t total_size_in_bytes = (level1_size + level2_size + sizeof(iceberg_lv1_block_md) + sizeof(iceberg_lv2_block_md)) * total_blocks;

  // increment resize cnt
  table->metadata.resize_cnt += 1;

  // update metadata
  table->metadata.total_size_in_bytes = total_size_in_bytes;
  table->metadata.nslots *= 2;
  table->metadata.nblocks = total_blocks;
  table->metadata.block_bits += 1;
  table->metadata.nblocks_parts[resize_cnt] = total_blocks;
  table->metadata.resize_threshold = RESIZE_THRESHOLD * total_capacity(table);

  // reset the block ctr
  table->metadata.lv1_resize_ctr = table->metadata.nblocks / 2;
  table->metadata.lv2_resize_ctr = table->metadata.nblocks / 2;

  /*printf("Setting up finished\n");*/
  unlock(&table->metadata.lock);
  return true;
}

static bool iceberg_lv1_move_block(iceberg_table * table, uint64_t bnum, uint8_t thread_id);
static bool iceberg_lv2_move_block(iceberg_table * table, uint64_t bnum, uint8_t thread_id);

// finish moving blocks that are left during the last resize.
void iceberg_end(iceberg_table * table) {
  if (is_lv1_resize_active(table)) {
    for (uint64_t chunk = 0; chunk < table->metadata.nblocks / 8; ++chunk) {
      partition_block pb = decode_raw_chunk(table, chunk);
      // if fixing is needed set the marker
      if (!__sync_lock_test_and_set(&table->metadata.lv1_resize_marker[pb.partition][pb.block], 1)) {
        for (uint8_t i = 0; i < 8; ++i) {
          uint64_t idx = chunk * 8 + i;
          iceberg_lv1_move_block(table, idx, 0);
        }
        // set the marker for the dest block
        uint64_t dest_chunk = chunk + table->metadata.nblocks / 8 / 2;
        pb = decode_raw_chunk(table, dest_chunk);
        __sync_lock_test_and_set(&table->metadata.lv1_resize_marker[pb.partition][pb.block], 1);
      }
    }
  }
  if (is_lv2_resize_active(table)) {
    for (uint64_t chunk = 0; chunk < table->metadata.nblocks / 8; ++chunk) {
      partition_block pb = decode_raw_chunk(table, chunk);
      // if fixing is needed set the marker
      if (!__sync_lock_test_and_set(&table->metadata.lv2_resize_marker[pb.partition][pb.block], 1)) {
        for (uint8_t i = 0; i < 8; ++i) {
          uint64_t idx = chunk * 8 + i;
          iceberg_lv2_move_block(table, idx, 0);
        }
        // set the marker for the dest block
        uint64_t dest_chunk = chunk + table->metadata.nblocks / 8 / 2;
        pb = decode_raw_chunk(table, dest_chunk);
        __sync_lock_test_and_set(&table->metadata.lv2_resize_marker[pb.partition][pb.block], 1);
      }
    }
  }
}

static inline bool
iceberg_lv3_insert(iceberg_table * table, KeyType key, ValueType value, hash *h, uint8_t thread_id) {
  uint64_t block = get_level3_block(h);
  iceberg_metadata * metadata = &table->metadata;

  while(__sync_lock_test_and_set(&metadata->lv3_locks[block], 1)) {
    _mm_pause();
  }

  iceberg_lv3_node *new_node = (iceberg_lv3_node *)malloc(sizeof(iceberg_lv3_node));

  new_node->key = key;
  new_node->val = value;
  new_node->next_node = table->level3[block].head;
  table->level3[block].head = new_node;

  metadata->lv3_sizes[block]++;
  pc_add(&metadata->lv3_balls, 1, thread_id);
  metadata->lv3_locks[block] = 0;

  return true;
}

static inline bool
iceberg_lv2_insert_internal(iceberg_table * table, KeyType key, ValueType value, hash *h, partition_block pb, uint8_t thread_id) {
  iceberg_metadata * metadata = &table->metadata;

start: ;
  __mmask32 md_mask = slot_mask_32(metadata->lv2_md[pb.partition][pb.block].block_md, 0) & ((1 << level2_slots_per_block()) - 1);
  verbose_print_sketch(metadata->lv2_md[pb.partition][pb.block].block_md, 8);
  verbose_print_mask8(md_mask);
  uint8_t popct = __builtin_popcountll(md_mask);

  if (unlikely(!popct))
    return false;

  uint64_t start = 0;
  uint64_t slot = word_select(md_mask, start);

  if(__sync_bool_compare_and_swap(metadata->lv2_md[pb.partition][pb.block].block_md + slot, 0, 1)) {
    pc_add(&metadata->lv2_balls, 1, thread_id);
    kv_pair *kv = level2_kv_pair(table, pb, slot);
    verbose_print_location(2, pb.partition, pb.block, slot, kv);
    atomic_write_128(key, value, kv);
    metadata->lv2_md[pb.partition][pb.block].block_md[slot] = h->fingerprint;
    return true;
  }
  goto start;

  return false;
}

static inline bool
iceberg_lv2_insert(iceberg_table * table, KeyType key, ValueType value, hash *h, uint8_t thread_id) {

  iceberg_metadata * metadata = &table->metadata;

  if (metadata->lv2_ctr == (int64_t)(6 * metadata->nblocks)) {
    return iceberg_lv3_insert(table, key, value, h, thread_id);
  }

  partition_block pb1 = get_block(table, h, LEVEL2_BLOCK1);
  partition_block pb2 = get_block(table, h, LEVEL2_BLOCK2);

  __mmask32 md_mask1 = slot_mask_32(metadata->lv2_md[pb1.partition][pb1.block].block_md, 0) & ((1 << level2_slots_per_block()) - 1);
  __mmask32 md_mask2 = slot_mask_32(metadata->lv2_md[pb2.partition][pb2.block].block_md, 0) & ((1 << level2_slots_per_block()) - 1);
    verbose_print_sketch(metadata->lv2_md[pb1.partition][pb1.block].block_md, 8);
    verbose_print_mask8(md_mask1);
    verbose_print_sketch(metadata->lv2_md[pb2.partition][pb2.block].block_md, 8);
    verbose_print_mask8(md_mask2);

  uint8_t popct1 = __builtin_popcountll(md_mask1);
  uint8_t popct2 = __builtin_popcountll(md_mask2);

  uint64_t raw_block = h->raw_block[LEVEL2_BLOCK1];
  if(popct2 > popct1) {
    pb1 = pb2;
    raw_block = h->raw_block[LEVEL2_BLOCK2];
  }

#ifdef ENABLE_RESIZE
  // move blocks if resize is active and not already moved.
  if (unlikely(is_lv2_resize_active(table) && raw_block < (table->metadata.nblocks >> 1))) {
    uint64_t chunk = raw_block / 8;
    partition_block pb = decode_raw_chunk(table, chunk);
    // if fixing is needed set the marker
    if (!__sync_lock_test_and_set(&table->metadata.lv2_resize_marker[pb.partition][pb.block], 1)) {
      for (uint8_t i = 0; i < 8; ++i) {
        uint64_t idx = chunk * 8 + i;
        /*printf("LV2 Before: Moving block: %ld load: %f\n", idx, iceberg_block_load(table, idx, 2));*/
        iceberg_lv2_move_block(table, idx, thread_id);
        /*printf("LV2 After: Moving block: %ld load: %f\n", idx, iceberg_block_load(table, idx, 2));*/
      }
      // set the marker for the dest block
      uint64_t dest_chunk = chunk + table->metadata.nblocks / 8 / 2;
      pb = decode_raw_chunk(table, dest_chunk);
      __sync_lock_test_and_set(&table->metadata.lv2_resize_marker[pb.partition][pb.block], 1);
    }
  }
#endif

  if (iceberg_lv2_insert_internal(table, key, value, h, pb1, thread_id)) {
    return true;
  }

  return iceberg_lv3_insert(table, key, value, h, thread_id);
}

static inline bool
iceberg_insert_internal(iceberg_table * table, KeyType key, ValueType value, hash *h, partition_block pb, uint8_t thread_id) {
  iceberg_metadata * metadata = &table->metadata;

start: ;
  __mmask64 md_mask = slot_mask_64(metadata->lv1_md[pb.partition][pb.block].block_md, 0);
  verbose_print_sketch(metadata->lv1_md[pb.partition][pb.block].block_md, 64);
  verbose_print_mask64(md_mask);

  uint8_t popct = __builtin_popcountll(md_mask);

  if (unlikely(!popct)) {
    return false;
  }

  uint64_t start = 0;
  uint64_t slot = word_select(md_mask, start);

  pc_add(&metadata->lv1_balls, 1, thread_id);
  kv_pair *kv = level1_kv_pair(table, pb, slot);
  verbose_print_location(1, pb.partition, pb.block, slot, kv);
  atomic_write_128(key, value, kv);
  metadata->lv1_md[pb.partition][pb.block].block_md[slot] = h->fingerprint;
  return true;

  goto start;
}

static inline bool
iceberg_get_value_internal(iceberg_table * table, KeyType key, ValueType *value, hash *h, uint8_t thread_id);

__attribute__ ((always_inline)) inline bool
iceberg_insert(iceberg_table * table, KeyType key, ValueType value, uint8_t thread_id) {
  verbose_print_operation("INSERT", key, value);

#ifdef ENABLE_RESIZE
  if (unlikely(need_resize(table))) {
    iceberg_setup_resize(table);
  }
#endif

  iceberg_metadata * metadata = &table->metadata;
  hash h = hash_key(table, &key);

#ifdef ENABLE_RESIZE
  // move blocks if resize is active and not already moved.
  if (unlikely(is_lv1_resize_active(table) && h.raw_block[LEVEL1] < (table->metadata.nblocks >> 1))) {
    uint64_t chunk = h.raw_block[LEVEL1] / 8;
    partition_block pb = decode_raw_chunk(table, chunk);
    // if fixing is needed set the marker
    if (!__sync_lock_test_and_set(&table->metadata.lv1_resize_marker[pb.partition][pb.block], 1)) {
      for (uint8_t i = 0; i < 8; ++i) {
        uint64_t idx = chunk * 8 + i;
        /*printf("LV1 Before: Moving block: %ld load: %f\n", idx, iceberg_block_load(table, idx, 1));*/
        iceberg_lv1_move_block(table, idx, thread_id);
        /*printf("LV1 After: Moving block: %ld load: %f\n", idx, iceberg_block_load(table, idx, 1));*/
      }
      // set the marker for the dest block
      uint64_t dest_chunk = chunk + table->metadata.nblocks / 8 / 2;
      pb = decode_raw_chunk(table, dest_chunk);
      __sync_lock_test_and_set(&table->metadata.lv1_resize_marker[pb.partition][pb.block], 1);
    }
  }
#endif

  partition_block pb = get_block(table, &h, LEVEL1);
  lock_block((uint64_t *)&metadata->lv1_md[pb.partition][pb.block].block_md);
  ValueType v;
  verbose_print_operation("INTERNAL QUERY", key, value);
  bool ret = true;
  if (unlikely(iceberg_get_value_internal(table, key, &v, &h, thread_id))) {
    ret = true;
    goto out;
  }

  ret = iceberg_insert_internal(table, key, value, &h, pb, thread_id);
  if (!ret) {
    ret = iceberg_lv2_insert(table, key, value, &h, thread_id);
  }

out:
  verbose_end();
  unlock_block((uint64_t *)&metadata->lv1_md[pb.partition][pb.block].block_md);
  return ret;
}

static inline bool iceberg_lv3_remove_internal(iceberg_table * table, KeyType key, hash *h, uint8_t thread_id) {
  uint64_t block = get_level3_block(h);

  iceberg_metadata * metadata = &table->metadata;
  iceberg_lv3_list * lists = table->level3;

  while(__sync_lock_test_and_set(metadata->lv3_locks + block, 1));

  if(metadata->lv3_sizes[block] == 0) {
     return false;
  }

  iceberg_lv3_node *head = lists[block].head;

  if(head->key == key) {
    iceberg_lv3_node * old_head = lists[block].head;
    lists[block].head = lists[block].head->next_node;
    free(old_head);

    metadata->lv3_sizes[block]--;
    pc_add(&metadata->lv3_balls, -1, thread_id);
    metadata->lv3_locks[block] = 0;

    return true;
  }

  iceberg_lv3_node * current_node = head;

  for(uint64_t i = 0; i < metadata->lv3_sizes[block] - 1; ++i) {
    iceberg_lv3_node *next_node = current_node->next_node;

    if(next_node->key == key) {
      iceberg_lv3_node * old_node = current_node->next_node;
      current_node->next_node = current_node->next_node->next_node;
      free(old_node);

      metadata->lv3_sizes[block]--;
      pc_add(&metadata->lv3_balls, -1, thread_id);
      metadata->lv3_locks[block] = 0;

      return true;
    }

    current_node = next_node;
  }

  metadata->lv3_locks[block] = 0;
  return false;
}

static inline bool iceberg_lv3_remove(iceberg_table * table, KeyType key, hash *h, uint8_t thread_id) {
  return iceberg_lv3_remove_internal(table, key, h, thread_id);
}

static inline bool iceberg_lv2_remove(iceberg_table * table, KeyType key, hash *h, uint8_t thread_id) {
  iceberg_metadata * metadata = &table->metadata;

  for(level lvl = LEVEL2_BLOCK1; lvl < NUM_LEVELS; ++lvl) {
    partition_block pb = get_block(table, h, lvl);

#ifdef ENABLE_RESIZE
    // check if there's an active resize and block isn't fixed yet
    if (unlikely(is_lv2_resize_active(table) && h->raw_block[lvl] >= (table->metadata.nblocks >> 1))) {
      uint64_t mask = ~(1ULL << (table->metadata.block_bits - 1));
      uint64_t old_index = h->raw_block[lvl] & mask;
      uint64_t chunk = old_index / 8;
      partition_block pb = decode_raw_chunk(table, chunk);
      if (__atomic_load_n(&table->metadata.lv2_resize_marker[pb.partition][pb.block], __ATOMIC_SEQ_CST) == 0) { // not fixed yet
        partition_block old_pb = decode_raw_chunk(table, old_index);
        __mmask32 md_mask = slot_mask_32(metadata->lv2_md[old_pb.partition][old_pb.block].block_md, h->fingerprint) & ((1 << level2_slots_per_block()) - 1);
        verbose_print_sketch(metadata->lv2_md[old_pb.partition][old_pb.block].block_md, 8);
        verbose_print_mask8(md_mask);
        uint8_t popct = __builtin_popcount(md_mask);

        for(uint8_t i = 0; i < popct; ++i) {
          uint64_t slot = word_select(md_mask, i);

          kv_pair *candidate_kv = level1_kv_pair(table, old_pb, slot);
          if (candidate_kv->key == key) {
            verbose_print_location(2, old_pb.partition, old_pb.block, slot, candidate_kv);
            candidate_kv->key = 0;
            candidate_kv->val = 0;
            metadata->lv2_md[old_pb.partition][old_pb.block].block_md[slot] = 0;
            pc_add(&metadata->lv2_balls, -1, thread_id);
            return true;
          }
        }
      } else {
        // wait for the old block to be fixed
        uint64_t dest_chunk = h->raw_block[lvl] / 8;
        pb = decode_raw_chunk(table, dest_chunk);
        while (__atomic_load_n(&table->metadata.lv2_resize_marker[pb.partition][pb.block], __ATOMIC_SEQ_CST) == 0)
          ;
      }
    }
#endif

    __mmask32 md_mask = slot_mask_32(metadata->lv2_md[pb.partition][pb.block].block_md, h->fingerprint) & ((1 << level2_slots_per_block()) - 1);
    verbose_print_sketch(metadata->lv2_md[pb.partition][pb.block].block_md, 8);
    verbose_print_mask8(md_mask);
    uint8_t popct = __builtin_popcount(md_mask);

    for(uint8_t i = 0; i < popct; ++i) {
      uint64_t slot = word_select(md_mask, i);

      kv_pair *candidate_kv = level2_kv_pair(table, pb, slot);
      if (candidate_kv->key == key) {
        verbose_print_location(2, pb.partition, pb.block, slot, candidate_kv);
        candidate_kv->key = 0;
        candidate_kv->val = 0;
        metadata->lv2_md[pb.partition][pb.block].block_md[slot] = 0;
        pc_add(&metadata->lv2_balls, -1, thread_id);
        return true;
      }
    }
  }

  return iceberg_lv3_remove(table, key, h, thread_id);
}

bool iceberg_remove(iceberg_table * table, KeyType key, uint8_t thread_id) {
  verbose_print_operation("DELETE", key, 0);
  bool ret = true;

  iceberg_metadata * metadata = &table->metadata;
  hash h = hash_key(table, &key);
  partition_block pb = get_block(table, &h, LEVEL1);

#ifdef ENABLE_RESIZE
  // check if there's an active resize and block isn't fixed yet
  if (unlikely(is_lv1_resize_active(table) && h.raw_block[LEVEL1] >= (table->metadata.nblocks >> 1))) {
    uint64_t mask = ~(1ULL << (table->metadata.block_bits - 1));
    uint64_t old_index = h.raw_block[LEVEL1] & mask;
    uint64_t chunk = old_index / 8;
    partition_block chunk_pb = decode_raw_chunk(table, chunk);
    if (__atomic_load_n(&table->metadata.lv1_resize_marker[chunk_pb.partition][chunk_pb.block], __ATOMIC_SEQ_CST) == 0) { // not fixed yet
      partition_block old_pb = decode_raw_block(table, old_index);
      __mmask64 md_mask = slot_mask_64(metadata->lv1_md[old_pb.partition][old_pb.block].block_md, h.fingerprint);
      verbose_print_sketch(metadata->lv1_md[old_pb.partition][old_pb.block].block_md, 64);
      verbose_print_mask64(md_mask);
      uint8_t popct = __builtin_popcountll(md_mask);

      for(uint8_t i = 0; i < popct; ++i) {
        uint64_t slot = word_select(md_mask, i);

        kv_pair *candidate_kv = level1_kv_pair(table, old_pb, slot);
        if (candidate_kv->key == key) {
          metadata->lv1_md[old_pb.partition][old_pb.block].block_md[slot] = 0;
          verbose_print_location(1, old_pb.partition, old_pb.block, slot, candidate_kv);
          candidate_kv->key = 0;
          candidate_kv->val = 0;
          pc_add(&metadata->lv1_balls, -1, thread_id);
          goto out;
        }
      }
    } else {
      // wait for the old block to be fixed
      uint64_t dest_chunk = h.raw_block[LEVEL1] / 8;
      chunk_pb = decode_raw_chunk(table, dest_chunk);
      while (__atomic_load_n(&table->metadata.lv1_resize_marker[chunk_pb.partition][chunk_pb.block], __ATOMIC_SEQ_CST) == 0)
        ;
    }
  }
#endif

  lock_block((uint64_t *)&metadata->lv1_md[pb.partition][pb.block].block_md);
  __mmask64 md_mask = slot_mask_64(metadata->lv1_md[pb.partition][pb.block].block_md, h.fingerprint);
  verbose_print_sketch(metadata->lv1_md[pb.partition][pb.block].block_md, 64);
  verbose_print_mask64(md_mask);
  uint8_t popct = __builtin_popcountll(md_mask);

  for(uint8_t i = 0; i < popct; ++i) {
    uint64_t slot = word_select(md_mask, i);

    kv_pair *candidate_kv = level1_kv_pair(table, pb, slot);
    if (candidate_kv->key == key) {
      verbose_print_location(1, pb.partition, pb.block, slot, candidate_kv);
      metadata->lv1_md[pb.partition][pb.block].block_md[slot] = 0;
      candidate_kv->key = 0;
      candidate_kv->val = 0;
      pc_add(&metadata->lv1_balls, -1, thread_id);
      goto unlock_out;
    }
  }

  ret = iceberg_lv2_remove(table, key, &h, thread_id);

unlock_out:
  unlock_block((uint64_t *)&metadata->lv1_md[pb.partition][pb.block].block_md);
out:
  verbose_end();
  return ret;
}

static inline bool iceberg_lv3_get_value_internal(iceberg_table * table, KeyType key, ValueType *value, hash *h) {
  uint64_t block = get_level3_block(h);

  iceberg_metadata * metadata = &table->metadata;
  iceberg_lv3_list * lists = table->level3;

  if(likely(!metadata->lv3_sizes[block])) {
    return false;
  }

  while(__sync_lock_test_and_set(metadata->lv3_locks + block, 1));

  iceberg_lv3_node * current_node = lists[block].head;

  for(uint8_t i = 0; i < metadata->lv3_sizes[block]; ++i) {
    if(current_node->key == key) {
      *value = current_node->val;
      metadata->lv3_locks[block] = 0;
      return true;
    }
    current_node = current_node->next_node;
  }

  metadata->lv3_locks[block] = 0;

  return false;
}

static inline bool iceberg_lv3_get_value(iceberg_table * table, KeyType key, ValueType *value, hash *h) {
  return iceberg_lv3_get_value_internal(table, key, value, h);
}

static inline bool iceberg_lv2_get_value(iceberg_table * table, KeyType key, ValueType *value, hash *h) {

  iceberg_metadata * metadata = &table->metadata;

  for(level lvl = LEVEL2_BLOCK1; lvl < NUM_LEVELS; ++lvl) {
    partition_block pb = get_block(table, h, lvl);

#ifdef ENABLE_RESIZE
    // check if there's an active resize and block isn't fixed yet
    if (unlikely(is_lv2_resize_active(table) && h->raw_block[lvl] >= (table->metadata.nblocks >> 1))) {
      uint64_t mask = ~(1ULL << (table->metadata.block_bits - 1));
      uint64_t old_index = h->raw_block[lvl] & mask;
      uint64_t chunk = old_index / 8;
      partition_block chunk_pb = decode_raw_chunk(table, chunk);
      if (__atomic_load_n(&table->metadata.lv2_resize_marker[chunk_pb.partition][chunk_pb.block], __ATOMIC_SEQ_CST) == 0) { // not fixed yet
        partition_block old_pb = decode_raw_block(table, old_index);
        __mmask32 md_mask = slot_mask_32(metadata->lv2_md[old_pb.partition][old_pb.block].block_md, h->fingerprint) & ((1 << level2_slots_per_block()) - 1);
        verbose_print_sketch(metadata->lv2_md[old_pb.partition][old_pb.block].block_md, 8);
        verbose_print_mask8(md_mask);

        while (md_mask != 0) {
          int slot = __builtin_ctz(md_mask);
          md_mask = md_mask & ~(1U << slot);

          kv_pair *candidate_kv = level2_kv_pair(table, old_pb, slot);
          if (candidate_kv->key == key) {
            verbose_print_location(2, old_pb.partition, old_pb.block, slot, candidate_kv);
            *value = candidate_kv->val;
            return true;
          }
        }
      } else {
        // wait for the old block to be fixed
        uint64_t dest_chunk = h->raw_block[lvl] / 8;
        chunk_pb = decode_raw_chunk(table, dest_chunk);
        while (__atomic_load_n(&table->metadata.lv2_resize_marker[chunk_pb.partition][chunk_pb.block], __ATOMIC_SEQ_CST) == 0)
          ;
      }
    }
#endif

    __mmask32 md_mask =
      slot_mask_32(metadata->lv2_md[pb.partition][pb.block].block_md,
          h->fingerprint) & ((1 << level2_slots_per_block()) - 1);
    verbose_print_sketch(metadata->lv2_md[pb.partition][pb.block].block_md, 8);
    verbose_print_mask8(md_mask);

    while (md_mask != 0) {
      int slot = __builtin_ctz(md_mask);
      md_mask = md_mask & ~(1U << slot);

      kv_pair *candidate_kv = level2_kv_pair(table, pb, slot);
      if (candidate_kv->key == key) {
        verbose_print_location(2, pb.partition, pb.block, slot, candidate_kv);
        *value = candidate_kv->val;
        return true;
      }
    }

  }

  return iceberg_lv3_get_value(table, key, value, h);
}

__attribute__ ((always_inline)) inline bool
iceberg_get_value_internal(iceberg_table * table, KeyType key, ValueType *value, hash *h, uint8_t thread_id) {
  iceberg_metadata * metadata = &table->metadata;

#ifdef ENABLE_RESIZE
  // check if there's an active resize and block isn't fixed yet
  if (unlikely(is_lv1_resize_active(table) && h->raw_block[LEVEL1] >= (table->metadata.nblocks >> 1))) {
    uint64_t mask = ~(1ULL << (table->metadata.block_bits - 1));
    uint64_t old_index = h->raw_block[LEVEL1] & mask;
    uint64_t chunk = old_index / 8;
    partition_block chunk_pb = decode_raw_chunk(table, chunk);
    if (__atomic_load_n(&table->metadata.lv1_resize_marker[chunk_pb.partition][chunk_pb.block], __ATOMIC_SEQ_CST) == 0) { // not fixed yet
      partition_block old_pb = decode_raw_block(table, old_index);
      __mmask64 md_mask = slot_mask_64(metadata->lv1_md[old_pb.partition][old_pb.block].block_md, h->fingerprint);
      verbose_print_sketch(metadata->lv1_md[old_pb.partition][old_pb.block].block_md, 64);
      verbose_print_mask64(md_mask);

      while (md_mask != 0) {
        int slot = __builtin_ctzll(md_mask);
        md_mask = md_mask & ~(1ULL << slot);

        kv_pair *candidate_kv = level1_kv_pair(table, old_pb, slot);
        if (candidate_kv->key == key) {
          verbose_print_location(1, old_pb.partition, old_pb.block, slot, candidate_kv);
          *value = candidate_kv->val;
          return true;
        }
      }
    } else {
      // wait for the old block to be fixed
      uint64_t dest_chunk = h->raw_block[LEVEL1] / 8;
      chunk_pb = decode_raw_chunk(table, dest_chunk);
      while (__atomic_load_n(&table->metadata.lv1_resize_marker[chunk_pb.partition][chunk_pb.block], __ATOMIC_SEQ_CST) == 0)
        ;
    }
  }
#endif

  partition_block pb = decode_raw_block(table, h->raw_block[LEVEL1]);
  __mmask64 md_mask = slot_mask_64(metadata->lv1_md[pb.partition][pb.block].block_md, h->fingerprint);
  verbose_print_sketch(metadata->lv1_md[pb.partition][pb.block].block_md, 64);
  verbose_print_mask64(md_mask);

  while (md_mask != 0) {
    int slot = __builtin_ctzll(md_mask);
    md_mask = md_mask & ~(1ULL << slot);

    kv_pair *candidate_kv = level1_kv_pair(table, pb, slot);
    if (candidate_kv->key == key) {
      verbose_print_location(1, pb.partition, pb.block, slot, candidate_kv);
      *value = candidate_kv->val;
      return true;
    }
  }

  bool ret = iceberg_lv2_get_value(table, key, value, h);

  return ret;
}


__attribute__ ((always_inline)) inline bool
iceberg_get_value(iceberg_table * table, KeyType key, ValueType *value, uint8_t thread_id) {
  verbose_print_operation("QUERY", key, 0);
  bool ret = true;

  hash h = hash_key(table, &key);
  ret = iceberg_get_value_internal(table, key, value, &h, thread_id);
  verbose_end();
  return ret;
}

#ifdef ENABLE_RESIZE
static bool iceberg_nuke_key(iceberg_table * table, uint64_t level, uint64_t index, uint64_t slot, uint64_t thread_id) {
  partition_block pb = decode_raw_block(table, index);
  iceberg_metadata * metadata = &table->metadata;

  if (level == 1) {
    kv_pair *kv = level1_kv_pair(table, pb, slot);
    kv->key = 0;
    kv->val = 0;
    metadata->lv1_md[pb.partition][pb.block].block_md[slot] = 0;
    pc_add(&metadata->lv1_balls, -1, thread_id);
  } else if (level == 2) {
    kv_pair *kv = level2_kv_pair(table, pb, slot);
    kv->key = 0;
    kv->val = 0;
    metadata->lv2_md[pb.partition][pb.block].block_md[slot] = 0;
    pc_add(&metadata->lv2_balls, -1, thread_id);
  }

  return true;
}

static bool iceberg_lv1_move_block(iceberg_table * table, uint64_t bnum, uint8_t thread_id) {
  // grab a block
  uint64_t bctr = __atomic_fetch_sub(&table->metadata.lv1_resize_ctr, 1, __ATOMIC_SEQ_CST);
  assert(bctr !=  0);

  partition_block pb = decode_raw_block(table, bnum);
  // relocate items in level1
  for (uint64_t j = 0; j < level1_slots_per_block(); ++j) {
    kv_pair *kv = level1_kv_pair(table, pb, j);
    if (kv->key == 0) {
      continue;
    }

    hash h = hash_key(table, &kv->key);

    // move to new location
    if (h.raw_block[LEVEL1] != bnum) {
      partition_block local_pb = decode_raw_block(table, h.raw_block[LEVEL1]);
      if (!iceberg_insert_internal(table, kv->key, kv->val, &h, local_pb, thread_id)) {
        printf("Failed insert during resize lv1\n");
        exit(0);
      }
      if (!iceberg_nuke_key(table, 1, bnum, j, thread_id)) {
        printf("Failed remove during resize lv1. key: %" PRIu64 ", block: %" PRIu64"\n", kv->key, bnum);
        exit(0);
      }
      //ValueType *val;
      //if (!iceberg_get_value(table, key, &val, thread_id)) {
      // printf("Key not found during resize lv1: %ld\n", key);
      //exit(0);
      //}
    }
  }

  return false;
}

static bool iceberg_lv2_move_block(iceberg_table * table, uint64_t bnum, uint8_t thread_id) {
  // grab a block
  uint64_t bctr = __atomic_fetch_sub(&table->metadata.lv2_resize_ctr, 1, __ATOMIC_SEQ_CST);
  assert(bctr != 0);

  partition_block pb = decode_raw_block(table, bnum);
  uint64_t mask = ~(1ULL << (table->metadata.block_bits - 1));
  // relocate items in level2
  for (uint64_t j = 0; j < level2_slots_per_block(); ++j) {
    kv_pair *kv = level2_kv_pair(table, pb, j);
    if (kv->key == 0) {
      continue;
    }

    hash h = hash_key(table, &kv->key);

    if ((h.raw_block[LEVEL2_BLOCK1] & mask) == bnum && h.raw_block[LEVEL2_BLOCK1] != bnum) {
      partition_block local_pb = decode_raw_block(table, h.raw_block[LEVEL2_BLOCK1]);
      if (!iceberg_lv2_insert_internal(table, kv->key, kv->val, &h, local_pb, thread_id)) {
        if (!iceberg_lv2_insert(table, kv->key, kv->val, &h, thread_id)) {
          printf("Failed insert during resize lv2\n");
          exit(0);
        }
      }
      if (!iceberg_nuke_key(table, 2, bnum, j, thread_id)) {
        printf("Failed remove during resize lv2\n");
        exit(0);
      }
    } else if ((h.raw_block[LEVEL2_BLOCK2] & mask) == bnum && h.raw_block[LEVEL2_BLOCK2] != bnum) {
      partition_block local_pb = decode_raw_block(table, h.raw_block[LEVEL2_BLOCK2]);
      if (!iceberg_lv2_insert_internal(table, kv->key, kv->val, &h, local_pb, thread_id)) {
        if (!iceberg_lv2_insert(table, kv->key, kv->val, &h, thread_id)) {
          printf("Failed insert during resize lv2\n");
          exit(0);
        }
      }
      if (!iceberg_nuke_key(table, 2, bnum, j, thread_id)) {
        printf("Failed remove during resize lv2\n");
        exit(0);
      }
    }
  }
  return false;
}

#endif
