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

#ifdef PMEM
#include <fcntl.h>
#include <libpmem.h>
#endif

#include "xxhash.h"
#include "iceberg_precompute.h"
#include "iceberg_table.h"

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

#ifdef PMEM
#define PMEM_PATH "/mnt/pmem1"
#define FILENAME_LEN 1024
#define NUM_LEVEL3_NODES 2048
#define FILE_SIZE (10ULL * 1024 * 1024 * 1024)
#endif

uint64_t seed[5] = { 12351327692179052ll, 23246347347385899ll, 35236262354132235ll, 13604702930934770ll, 57439820692984798ll };

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

static inline uint64_t total_capacity(iceberg_table * table) {
  return table->metadata.nblocks * ((1 << SLOT_BITS) + C_LV2 + MAX_LG_LG_N / D_CHOICES);
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

#ifdef PMEM
static inline uint8_t get_slot_choice(KeyType key)
{
  uint64_t hash = MurmurHash64A(&key, sizeof(KeyType), seed[0]);
  return hash & ((1 << FPRINT_BITS) - 1);
}
#endif

typedef struct __attribute__ (( packed )) {
  uint64_t level1_raw_block  : 40;
  uint64_t level2_raw_block1 : 40;
  uint64_t level2_raw_block2 : 40;
  uint8_t  fingerprint   : 8;
} raw_hash;

_Static_assert(sizeof(raw_hash) == 16, "hash not 16B\n");

typedef struct {
  raw_hash raw;
  uint64_t level1_raw_block;
  uint64_t level2_raw_block1;
  uint64_t level2_raw_block2;
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
  *raw = XXH128(key, sizeof(*key), seed[0]);
  h.fingerprint = ensure_nonzero_fingerprint(&h.raw);
  h.level1_raw_block = truncate_to_current_num_raw_blocks(table, h.raw.level1_raw_block);
  h.level2_raw_block1 = truncate_to_current_num_raw_blocks(table, h.raw.level2_raw_block1);
  h.level2_raw_block2 = truncate_to_current_num_raw_blocks(table, h.raw.level2_raw_block2);
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
  while ((__sync_fetch_and_or(data, LOCK_MASK) & 1) != 0) { _mm_pause(); }
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


static inline void atomic_write_128(uint64_t key, uint64_t val, uint64_t *slot) {
  uint64_t arr[2] = {key, val};
  __m128d a =  _mm_load_pd((double *)arr);
  _mm_store_pd ((double*)slot, a);
}

int iceberg_init(iceberg_table *table, uint64_t log_slots) {
  memset(table, 0, sizeof(*table));

  uint64_t total_blocks = 1 << (log_slots - SLOT_BITS);
  uint64_t total_size_in_bytes = (sizeof(iceberg_lv1_block) + sizeof(iceberg_lv2_block) + sizeof(iceberg_lv1_block_md) + sizeof(iceberg_lv2_block_md)) * total_blocks;

  assert(table);

#if PMEM
  size_t mapped_len;
  int is_pmem;

  char level1_filename[FILENAME_LEN];
  sprintf(level1_filename, "%s/level1", PMEM_PATH);
  if ((table->level1[0] = (iceberg_lv1_block *)pmem_map_file(level1_filename, FILE_SIZE, PMEM_FILE_CREATE | PMEM_FILE_SPARSE, 0666, &mapped_len, &is_pmem)) == NULL) {
    perror("pmem_map_file");
    exit(1);
  }
  assert(is_pmem);
  assert(mapped_len == FILE_SIZE);

  char level2_filename[FILENAME_LEN];
  sprintf(level2_filename, "%s/level2", PMEM_PATH);
  if ((table->level2[0] = (iceberg_lv2_block *)pmem_map_file(level2_filename, FILE_SIZE, PMEM_FILE_CREATE | PMEM_FILE_SPARSE, 0666, &mapped_len, &is_pmem)) == NULL) {
    perror("pmem_map_file");
  }
  assert(is_pmem);
  assert(mapped_len == FILE_SIZE);

  char level3_filename[FILENAME_LEN];
  sprintf(level3_filename, "%s/level3", PMEM_PATH);
  if ((table->level3[0] = (iceberg_lv3_list *)pmem_map_file(level3_filename, FILE_SIZE, PMEM_FILE_CREATE | PMEM_FILE_SPARSE, 0666, &mapped_len, &is_pmem)) == NULL) {
    perror("pmem_map_file");
    exit(1);
  }
  assert(is_pmem);
  assert(mapped_len == FILE_SIZE);

  size_t level3_nodes_size = NUM_LEVEL3_NODES * sizeof(iceberg_lv3_node);
  char level3_nodes_filename[FILENAME_LEN];
  sprintf(level3_nodes_filename, "%s/level3_data", PMEM_PATH);
  table->level3_nodes =
    (iceberg_lv3_node *)pmem_map_file(level3_nodes_filename,
        level3_nodes_size, PMEM_FILE_CREATE | PMEM_FILE_SPARSE, 0666,
        &mapped_len, &is_pmem);
  if (table->level3_nodes == NULL) {
    perror("pmem_map_file");
    exit(1);
  }
  assert(is_pmem);
  assert(mapped_len == level3_nodes_size);
#else
  size_t level1_size = sizeof(iceberg_lv1_block) * total_blocks;
  //table->level1 = (iceberg_lv1_block *)malloc(level1_size);
  table->level1[0] = (iceberg_lv1_block *)mmap(NULL, level1_size, PROT_READ | PROT_WRITE, MMAP_FLAGS, 0, 0);
  if (!table->level1[0]) {
    perror("level1 malloc failed");
    exit(1);
  }
  size_t level2_size = sizeof(iceberg_lv2_block) * total_blocks;
  //table->level2 = (iceberg_lv2_block *)malloc(level2_size);
  table->level2[0] = (iceberg_lv2_block *)mmap(NULL, level2_size, PROT_READ | PROT_WRITE, MMAP_FLAGS, 0, 0);
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
#endif

  table->metadata.total_size_in_bytes = total_size_in_bytes;
  table->metadata.nslots = 1 << log_slots;
  table->metadata.nblocks = total_blocks;
  table->metadata.block_bits = log_slots - SLOT_BITS;
  table->metadata.init_size = total_blocks;
  table->metadata.log_init_size = log2(total_blocks);
  table->metadata.nblocks_parts[0] = total_blocks;
  table->metadata.resize_threshold = RESIZE_THRESHOLD * total_capacity(table);

  pc_init(&table->metadata.lv1_balls, &table->metadata.lv1_ctr, MAX_PROCS, 1000);
  pc_init(&table->metadata.lv2_balls, &table->metadata.lv2_ctr, MAX_PROCS, 1000);
  pc_init(&table->metadata.lv3_balls, &table->metadata.lv3_ctr, MAX_PROCS, 1000);

  size_t lv1_md_size = sizeof(iceberg_lv1_block_md) * total_blocks + 64;
  //table->metadata.lv1_md = (iceberg_lv1_block_md *)malloc(sizeof(iceberg_lv1_block_md) * total_blocks);
  table->metadata.lv1_md[0] = (iceberg_lv1_block_md *)mmap(NULL, lv1_md_size, PROT_READ | PROT_WRITE, MMAP_FLAGS, 0, 0);
  if (!table->metadata.lv1_md[0]) {
    perror("lv1_md malloc failed");
    exit(1);
  }
  //table->metadata.lv2_md = (iceberg_lv2_block_md *)malloc(sizeof(iceberg_lv2_block_md) * total_blocks);
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
  table->metadata.lv1_resize_ctr = total_blocks;
  table->metadata.lv2_resize_ctr = total_blocks;

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
  memset(table->metadata.lv1_resize_marker[0], 1, resize_marker_size);
  memset(table->metadata.lv2_resize_marker[0], 1, resize_marker_size);

  table->metadata.marker_sizes[0] = resize_marker_size;
  table->metadata.lock = 0;
#endif

#if PMEM
  size_t level1_size = sizeof(iceberg_lv1_block) * total_blocks;
  size_t level2_size = sizeof(iceberg_lv2_block) * total_blocks;
  pmem_memset_persist(table->level1[0], 0, level1_size);
  pmem_memset_persist(table->level2[0], 0, level2_size);

  for (uint64_t i = 0; i < total_blocks; i++) {
    table->level3[0][i].head_idx = -1;
  }
  size_t level3_size = sizeof(iceberg_lv3_list) * total_blocks;
  pmem_persist(table->level3[0], level3_size);
  pmem_memset_persist(table->level3_nodes, 0, level3_nodes_size);
#else
  for (uint64_t i = 0; i < total_blocks; ++i) {
    for (uint64_t j = 0; j < (1 << SLOT_BITS); ++j) {
      table->level1[0][i].slots[j].key = table->level1[0][i].slots[j].val = 0;
    }

    for (uint64_t j = 0; j < C_LV2 + MAX_LG_LG_N / D_CHOICES; ++j) {
      table->level2[0][i].slots[j].key = table->level2[0][i].slots[j].val = 0;
    }
    table->level3->head = NULL;
  }
#endif

  memset((char *)table->metadata.lv1_md[0], 0, lv1_md_size);
  memset((char *)table->metadata.lv2_md[0], 0, lv2_md_size);
  memset(table->metadata.lv3_sizes, 0, LEVEL3_BLOCKS * sizeof(uint64_t));
  memset(table->metadata.lv3_locks, 0, LEVEL3_BLOCKS * sizeof(uint8_t));

  return 0;
}

#if PMEM
uint64_t iceberg_dismount(iceberg_table * table) {
  iceberg_end(table);
  for (uint64_t i = 0; i < table->metadata.resize_cnt; ++i) {
    uint64_t total_blocks = table->metadata.nblocks_parts[i];
    size_t lv1_md_size = sizeof(iceberg_lv1_block_md) * total_blocks + 64;
    munmap(table->metadata.lv1_md[i], lv1_md_size);
    size_t lv2_md_size = sizeof(iceberg_lv2_block_md) * total_blocks + 32;
    munmap(table->metadata.lv2_md[i], lv2_md_size);
    munmap(table->metadata.lv3_sizes, sizeof(uint64_t) * LEVEL3_BLOCKS);
    munmap(table->metadata.lv3_locks, sizeof(uint8_t) * LEVEL3_BLOCKS);
  }

  pc_destructor(&table->metadata.lv1_balls);
  pc_destructor(&table->metadata.lv2_balls);
  pc_destructor(&table->metadata.lv3_balls);

  pmem_unmap(table->level1[0], FILE_SIZE);
  pmem_unmap(table->level2[0], FILE_SIZE);
  pmem_unmap(table->level3[0], FILE_SIZE);

  size_t level3_nodes_size = NUM_LEVEL3_NODES * sizeof(iceberg_lv3_node);
  pmem_unmap(table->level3_nodes, level3_nodes_size);

#ifdef ENABLE_RESIZE
  for (uint64_t i = 0;  i <= table->metadata.resize_cnt; ++i) {
    memset(table->metadata.lv1_resize_marker[i], 0, table->metadata.marker_sizes[i]);
    memset(table->metadata.lv2_resize_marker[i], 0, table->metadata.marker_sizes[i]);
    memset(table->metadata.lv3_resize_marker[i], 0, table->metadata.marker_sizes[i]);
  }
#endif

  return table->metadata.block_bits + SLOT_BITS;
}

int iceberg_mount(iceberg_table *table, uint64_t log_slots, uint64_t resize_cnt) {
  memset(table, 0, sizeof(*table));

  uint64_t init_log_slots = log_slots;
  log_slots += resize_cnt;
  uint64_t total_blocks = 1 << (log_slots - SLOT_BITS);
  uint64_t total_size_in_bytes = (sizeof(iceberg_lv1_block) + sizeof(iceberg_lv2_block) + sizeof(iceberg_lv1_block_md) + sizeof(iceberg_lv2_block_md)) * total_blocks;

  assert(table);

  size_t mapped_len;
  int is_pmem;

  /* map the file for first partition */
  char level1_filename[FILENAME_LEN];
  sprintf(level1_filename, "%s/level1", PMEM_PATH);
  if ((table->level1[0] = (iceberg_lv1_block *)pmem_map_file(level1_filename, 0, 0, 0666, &mapped_len, &is_pmem)) == NULL) {
    perror("pmem_map_file");
    exit(1);
  }
  assert(is_pmem);
  assert(mapped_len == FILE_SIZE);

  char level2_filename[FILENAME_LEN];
  sprintf(level2_filename, "%s/level2", PMEM_PATH);
  if ((table->level2[0] = (iceberg_lv2_block *)pmem_map_file(level2_filename, 0, 0, 0666, &mapped_len, &is_pmem)) == NULL) {
    perror("pmem_map_file");
    exit(1);
  }
  assert(is_pmem);
  assert(mapped_len == FILE_SIZE);

  char level3_filename[FILENAME_LEN];
  sprintf(level3_filename, "%s/level3", PMEM_PATH);
  if ((table->level3[0] = (iceberg_lv3_list *)pmem_map_file(level3_filename, 0, 0, 0666, &mapped_len, &is_pmem)) == NULL) {
    perror("pmem_map_file");
    exit(1);
  }
  assert(is_pmem);
  assert(mapped_len == FILE_SIZE);

  size_t level3_nodes_size = NUM_LEVEL3_NODES * sizeof(iceberg_lv3_node);
  char level3_nodes_filename[FILENAME_LEN];
  sprintf(level3_nodes_filename, "%s/level3_data", PMEM_PATH);
  table->level3_nodes = (iceberg_lv3_node *)pmem_map_file(level3_nodes_filename, 0, 0, 0666, &mapped_len, &is_pmem);
  if (table->level3_nodes == NULL) {
    perror("pmem_map_file");
    exit(1);
  }
  assert(is_pmem);
  assert(mapped_len == level3_nodes_size);

  /* init metadata for first partition */
  table->metadata.total_size_in_bytes = total_size_in_bytes;
  table->metadata.nslots = 1 << log_slots;
  table->metadata.nblocks = total_blocks;
  table->metadata.block_bits = log_slots - SLOT_BITS;
  table->metadata.resize_cnt = resize_cnt;
  table->metadata.log_init_size = init_log_slots - SLOT_BITS;
  table->metadata.init_size = 1 << (init_log_slots - SLOT_BITS);
  table->metadata.nblocks_parts[0] = table->metadata.init_size;

  /* init counters */
  uint32_t MAX_PROCS = get_nprocs();
  pc_init(&table->metadata.lv1_balls, &table->metadata.lv1_ctr, MAX_PROCS, 1000);
  pc_init(&table->metadata.lv2_balls, &table->metadata.lv2_ctr, MAX_PROCS, 1000);
  pc_init(&table->metadata.lv3_balls, &table->metadata.lv3_ctr, MAX_PROCS, 1000);

  /* init fingerprint metadata for first partition */
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
  table->metadata.lv3_sizes[0] = (uint64_t *)mmap(NULL, sizeof(uint64_t) * total_blocks, PROT_READ | PROT_WRITE, MMAP_FLAGS, 0, 0);
  if (!table->metadata.lv3_sizes[0]) {
    perror("lv3_sizes malloc failed");
    exit(1);
  }
  table->metadata.lv3_locks[0] = (uint8_t *)mmap(NULL, sizeof(uint8_t) * total_blocks, PROT_READ | PROT_WRITE, MMAP_FLAGS, 0, 0);
  if (!table->metadata.lv3_locks[0]) {
    perror("lv3_locks malloc failed");
    exit(1);
  }

  // Init table/metadata for rest of the partitions
  for (uint64_t i = 1; i <= resize_cnt; ++i) {
    uint64_t initial_total_blocks = 1 << table->metadata.log_init_size;
    uint64_t nblocks = initial_total_blocks * pow(2, i-1);
    table->metadata.nblocks_parts[i] = nblocks;

    table->level1[i] = table->level1[0] + nblocks;
    table->level2[i] = table->level2[0] + nblocks;
    table->level3[i] = table->level3[0] + nblocks;

    size_t lv1_md_size = sizeof(iceberg_lv1_block_md) * nblocks + 64;
    table->metadata.lv1_md[i] = (iceberg_lv1_block_md *)mmap(NULL, lv1_md_size, PROT_READ | PROT_WRITE, MMAP_FLAGS, 0, 0);
    if (!table->metadata.lv1_md[i]) {
      perror("lv1_md malloc failed");
      exit(1);
    }
    size_t lv2_md_size = sizeof(iceberg_lv2_block_md) * nblocks + 32;
    table->metadata.lv2_md[i] = (iceberg_lv2_block_md *)mmap(NULL, lv2_md_size, PROT_READ | PROT_WRITE, MMAP_FLAGS, 0, 0);
    if (!table->metadata.lv2_md[i]) {
      perror("lv2_md malloc failed");
      exit(1);
    }
    table->metadata.lv3_sizes[i] = (uint64_t *)mmap(NULL, sizeof(uint64_t) * nblocks, PROT_READ | PROT_WRITE, MMAP_FLAGS, 0, 0);
    if (!table->metadata.lv3_sizes[i]) {
      perror("lv3_sizes malloc failed");
      exit(1);
    }
    table->metadata.lv3_locks[i] = (uint8_t *)mmap(NULL, sizeof(uint8_t) * nblocks, PROT_READ | PROT_WRITE, MMAP_FLAGS, 0, 0);
    if (!table->metadata.lv3_locks[i]) {
      perror("lv3_locks malloc failed");
      exit(1);
    }
  }

  for (uint64_t p = 0; p <= resize_cnt; ++p) {
    uint64_t total_blocks = table->metadata.nblocks_parts[p];

    // recover the metadata
    // level 1
    for (uint64_t i = 0; i < total_blocks; i++) {
      iceberg_lv1_block *block = &table->level1[p][i];
      uint8_t *block_md = table->metadata.lv1_md[p][i].block_md;
      for (uint64_t slot = 0; slot < 1 << SLOT_BITS; slot++) {
        KeyType key = block->slots[slot].key;
        if (key != 0) {
          hash h = hash_key(table, key);
          partition_block pb = decode_raw_block(table, h.level1_raw_block);
          assert(pb.partition == p);
          assert(pb.block == i);
          block_md[slot] = fprint;
          pc_add(&table->metadata.lv1_balls, 1, 0);
        }
      }
    }

    // level 2
    for (uint64_t i = 0; i < total_blocks; i++) {
      iceberg_lv2_block *block = &table->level2[p][i];
      uint8_t *block_md = table->metadata.lv2_md[p][i].block_md;
      for (uint64_t slot = 0; slot < C_LV2 + MAX_LG_LG_N / D_CHOICES; slot++) {
        KeyType key = block->slots[slot].key;
        if (key != 0) {
          hash h = hash_key(table, key);
          partition_block pb = decode_raw_block(table, h.level2_raw_block1);
          if (pb.partition != p || pb.block != i) {
            pb = decode_raw_block(table, h.level2_raw_block2);
          }
          assert(pb.partition == p);
          assert(pb.block == i);
          block_md[slot] = fprint;
          pc_add(&table->metadata.lv2_balls, 1, 0);
        }
      }
    }

    // level 3
    for (uint64_t i = 0; i < total_blocks; i++) {
      ptrdiff_t idx = table->level3[p][i].head_idx;
      while (idx != -1) {
        idx = table->level3_nodes[idx].next_idx;
        table->metadata.lv3_sizes[p][i]++;
        pc_add(&table->metadata.lv3_balls, 1, 0);
      }
    }
  }

#ifdef ENABLE_RESIZE
  table->metadata.lv1_resize_ctr = total_blocks;
  table->metadata.lv2_resize_ctr = total_blocks;

  // Create marker metadata for rest of the partitions
  for (uint64_t i = 0; i <= resize_cnt; ++i) {
    uint64_t nblocks = table->metadata.nblocks_parts[i];

    size_t resize_marker_size = sizeof(uint8_t) * nblocks / 8;
    table->metadata.marker_sizes[i] = resize_marker_size;

    table->metadata.lv1_resize_marker[i] = (uint8_t *)mmap(NULL, resize_marker_size, PROT_READ | PROT_WRITE, MMAP_FLAGS, 0, 0);
    if (!table->metadata.lv1_resize_marker[i]) {
      perror("level1 resize ctr malloc failed");
      exit(1);
    }
    table->metadata.lv2_resize_marker[i] = (uint8_t *)mmap(NULL, resize_marker_size, PROT_READ | PROT_WRITE, MMAP_FLAGS, 0, 0);
    if (!table->metadata.lv2_resize_marker[i]) {
      perror("level2 resize ctr malloc failed");
      exit(1);
    }

    memset(table->metadata.lv1_resize_marker[i], 1, resize_marker_size);
    memset(table->metadata.lv2_resize_marker[i], 1, resize_marker_size);
  }
#endif

  return 0;
}
#endif

#ifdef ENABLE_RESIZE
static inline bool is_lv1_resize_active(iceberg_table * table) {
  uint64_t half_mark = table->metadata.nblocks >> 1;
  uint64_t lv1_ctr = __atomic_load_n(&table->metadata.lv1_resize_ctr, __ATOMIC_SEQ_CST);
  return lv1_ctr < half_mark;
}

static inline bool is_lv2_resize_active(iceberg_table * table) {
  uint64_t half_mark = table->metadata.nblocks >> 1;
  uint64_t lv2_ctr = __atomic_load_n(&table->metadata.lv2_resize_ctr, __ATOMIC_SEQ_CST);
  return lv2_ctr < half_mark;
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
#if PMEM
  table->level1[resize_cnt] = table->level1[0] + cur_blocks;
  table->level2[resize_cnt] = table->level2[0] + cur_blocks;
  table->level3[resize_cnt] = table->level3[0] + cur_blocks;
  for (uint64_t i = 0; i < cur_blocks; i++) {
    table->level3[resize_cnt][i].head_idx = -1;
  }
#else
  // alloc level1
  size_t level1_size = sizeof(iceberg_lv1_block) * cur_blocks;
  table->level1[resize_cnt] = (iceberg_lv1_block *)mmap(NULL, level1_size, PROT_READ | PROT_WRITE, MMAP_FLAGS, 0, 0);
  if (table->level1[resize_cnt] == (void *)-1) {
    perror("level1 resize failed");
    exit(1);
  }

  // alloc level2
  size_t level2_size = sizeof(iceberg_lv2_block) * cur_blocks;
  table->level2[resize_cnt] = (iceberg_lv2_block *)mmap(NULL, level2_size, PROT_READ | PROT_WRITE, MMAP_FLAGS, 0, 0);
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
  uint64_t total_size_in_bytes = (sizeof(iceberg_lv1_block) + sizeof(iceberg_lv2_block) + sizeof(iceberg_lv1_block_md) + sizeof(iceberg_lv2_block_md)) * total_blocks;

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
  table->metadata.lv1_resize_ctr = 0;
  table->metadata.lv2_resize_ctr = 0;

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

  /*printf("Final resize done.\n");*/
  /*printf("Final resize done. Table load: %ld\n", iceberg_table_load(table));*/
}
#endif

static inline bool
iceberg_lv3_insert(iceberg_table * table, KeyType key, ValueType value, hash h, uint8_t thread_id) {
  uint64_t block = h.level1_raw_block % LEVEL3_BLOCKS;
  iceberg_metadata * metadata = &table->metadata;

  while(__sync_lock_test_and_set(&metadata->lv3_locks[block], 1)) {
    _mm_pause();
  }

#if PMEM
  iceberg_lv3_node * level3_nodes = table->level3_nodes;
  iceberg_lv3_node *new_node = NULL;
  ptrdiff_t new_node_idx = -1;
  ptrdiff_t start = h.level1_raw_block % NUM_LEVEL3_NODES;
  for (ptrdiff_t i = start; i != NUM_LEVEL3_NODES + start; i++) {
    ptrdiff_t j = i % NUM_LEVEL3_NODES;
    if (__sync_bool_compare_and_swap(&level3_nodes[j].in_use, 0, 1)) {
      new_node = &level3_nodes[j];
      new_node_idx = j;
      break;
    }
  }
  if (new_node == NULL) {
    metadata->lv3_locks[pb.partition][pb.block] = 0;
    return false;
  }
#else
  iceberg_lv3_node *new_node = (iceberg_lv3_node *)malloc(sizeof(iceberg_lv3_node));
#endif

  new_node->key = key;
  new_node->val = value;
#if PMEM
  new_node->next_idx = table->level3[block].head_idx;
  pmem_persist(new_node, sizeof(*new_node));
  table->level3[block].head_idx = new_node_idx;
  pmem_persist(&table->level3[block], sizeof(table->level3[block]));
#else
  new_node->next_node = table->level3[block].head;
  table->level3[block].head = new_node;
#endif

  metadata->lv3_sizes[block]++;
  pc_add(&metadata->lv3_balls, 1, thread_id);
  metadata->lv3_locks[block] = 0;

  return true;
}

static inline bool
iceberg_lv2_insert_internal(iceberg_table * table, KeyType key, ValueType value, hash h, partition_block pb, uint8_t thread_id) {
  iceberg_metadata * metadata = &table->metadata;
  iceberg_lv2_block * blocks = table->level2[pb.partition];

start: ;
  __mmask32 md_mask = slot_mask_32(metadata->lv2_md[pb.partition][pb.block].block_md, 0) & ((1 << (C_LV2 + MAX_LG_LG_N / D_CHOICES)) - 1);
  uint8_t popct = __builtin_popcountll(md_mask);

  if (unlikely(!popct))
    return false;

#if PMEM
  uint64_t slot_choice = get_slot_choice(key);
  uint64_t start = popct == 0 ? 0 : slot_choice % popct;
#else
  uint64_t start = 0;
#endif
  /*for(uint8_t i = start; i < start + popct; ++i) {*/
#if PMEM
    uint64_t slot = word_select(md_mask, start % popct);
#else
    uint64_t slot = word_select(md_mask, start);
#endif

    if(__sync_bool_compare_and_swap(metadata->lv2_md[pb.partition][pb.block].block_md + slot, 0, 1)) {
      pc_add(&metadata->lv2_balls, 1, thread_id);
      /*blocks[pb.block].slots[slot].key = key;*/
      /*blocks[pb.block].slots[slot].val = value;*/
      atomic_write_128(key, value, (uint64_t*)&blocks[pb.block].slots[slot]);
#if PMEM
      pmem_persist(&blocks[pb.block].slots[slot], sizeof(kv_pair));
#endif
      metadata->lv2_md[pb.partition][pb.block].block_md[slot] = h.fingerprint;
      return true;
    }
    goto start;
  /*}*/

  return false;
}

static inline bool
iceberg_lv2_insert(iceberg_table * table, KeyType key, ValueType value, hash h, uint8_t thread_id) {

  iceberg_metadata * metadata = &table->metadata;

  if (metadata->lv2_ctr == (int64_t)(C_LV2 * metadata->nblocks)) {
    return iceberg_lv3_insert(table, key, value, h, thread_id);
  }

  partition_block pb1 = decode_raw_block(table, h.level2_raw_block1);
  partition_block pb2 = decode_raw_block(table, h.level2_raw_block2);

  __mmask32 md_mask1 = slot_mask_32(metadata->lv2_md[pb1.partition][pb1.block].block_md, 0) & ((1 << (C_LV2 + MAX_LG_LG_N / D_CHOICES)) - 1);
  __mmask32 md_mask2 = slot_mask_32(metadata->lv2_md[pb2.partition][pb2.block].block_md, 0) & ((1 << (C_LV2 + MAX_LG_LG_N / D_CHOICES)) - 1);

  uint8_t popct1 = __builtin_popcountll(md_mask1);
  uint8_t popct2 = __builtin_popcountll(md_mask2);

  uint64_t raw_block = h.level2_raw_block1;
  if(popct2 > popct1) {
    pb1 = pb2;
    raw_block = h.level2_raw_block2;
  }

#ifdef ENABLE_RESIZE
  // move blocks if resize is active and not already moved.
  if (unlikely(raw_block < (table->metadata.nblocks >> 1) && is_lv2_resize_active(table))) {
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

  if (iceberg_lv2_insert_internal(table, key, value, h, pb1, thread_id))
    return true;

  return iceberg_lv3_insert(table, key, value, h, thread_id);
}

static inline bool
iceberg_insert_internal(iceberg_table * table, KeyType key, ValueType value, hash h, partition_block pb, uint8_t thread_id) {
  iceberg_metadata * metadata = &table->metadata;
  iceberg_lv1_block * blocks = table->level1[pb.partition];	
start: ;
  __mmask64 md_mask = slot_mask_64(metadata->lv1_md[pb.partition][pb.block].block_md, 0);

  uint8_t popct = __builtin_popcountll(md_mask);

  if (unlikely(!popct))
    return false;

#if PMEM
  uint64_t slot_choice = get_slot_choice(key);
  uint64_t start = popct == 0 ? 0 : slot_choice % popct;
#else
  uint64_t start = 0;
#endif
  /*for(uint8_t i = start; i < start + popct; ++i) {*/
#if PMEM
    uint64_t slot = word_select(md_mask, start % popct);
#else
    uint64_t slot = word_select(md_mask, start);
#endif

    /*if(__sync_bool_compare_and_swap(metadata->lv1_md[pb.partition][pb.block].block_md + slot, 0, 1)) {*/
      pc_add(&metadata->lv1_balls, 1, thread_id);
      /*blocks[pb.block].slots[slot].key = key;*/
      /*blocks[pb.block].slots[slot].val = value;*/
      atomic_write_128(key, value, (uint64_t*)&blocks[pb.block].slots[slot]);
#if PMEM
      pmem_persist(&blocks[pb.block].slots[slot], sizeof(kv_pair));
#endif
      metadata->lv1_md[pb.partition][pb.block].block_md[slot] = h.fingerprint;
      return true;
    /*}*/
  goto start;
  /*}*/

  return false;
}

inline bool
iceberg_get_value_internal(iceberg_table * table, KeyType key, ValueType *value, hash h, uint8_t thread_id);

__attribute__ ((always_inline)) inline bool
iceberg_insert(iceberg_table * table, KeyType key, ValueType value, uint8_t thread_id) {
#ifdef ENABLE_RESIZE
  if (unlikely(need_resize(table))) {
    iceberg_setup_resize(table);
  }
#endif

  iceberg_metadata * metadata = &table->metadata;
  hash h = hash_key(table, &key);

#ifdef ENABLE_RESIZE
  // move blocks if resize is active and not already moved.
  if (unlikely(h.level1_raw_block < (table->metadata.nblocks >> 1) && is_lv1_resize_active(table))) {
    uint64_t chunk = h.level1_raw_block / 8;
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

  partition_block pb = decode_raw_block(table, h.level1_raw_block);
  lock_block((uint64_t *)&metadata->lv1_md[pb.partition][pb.block].block_md);
  ValueType v;
  if (unlikely(iceberg_get_value_internal(table, key, &v, h, thread_id))) {
    /*printf("Found!\n");*/
    unlock_block((uint64_t *)&metadata->lv1_md[pb.partition][pb.block].block_md);
    return true;
  }

  bool ret = iceberg_insert_internal(table, key, value, h, pb, thread_id);
  if (!ret) {
    ret = iceberg_lv2_insert(table, key, value, h, thread_id);
  }

  unlock_block((uint64_t *)&metadata->lv1_md[pb.partition][pb.block].block_md);
  return ret;
}

static inline bool iceberg_lv3_remove_internal(iceberg_table * table, KeyType key, uint64_t level1_raw_block, uint8_t thread_id) {
  uint64_t block = level1_raw_block % LEVEL3_BLOCKS;

  iceberg_metadata * metadata = &table->metadata;
  iceberg_lv3_list * lists = table->level3;

  while(__sync_lock_test_and_set(metadata->lv3_locks + block, 1));

  if(metadata->lv3_sizes[block] == 0) {
     return false;
  }

#if PMEM
  iceberg_lv3_node * lv3_nodes = table->level3_nodes;
  assert(lists[block].head_idx != -1);
  iceberg_lv3_node *head = &lv3_nodes[lists[block].head_idx];
#else
  iceberg_lv3_node *head = lists[block].head;
#endif

  if(head->key == key) {
#if PMEM
    lists[block].head_idx = head->next_idx;
    pmem_memset_persist(head, 0, sizeof(*head));
#else
    iceberg_lv3_node * old_head = lists[block].head;
    lists[block].head = lists[block].head->next_node;
    free(old_head);
#endif

    metadata->lv3_sizes[block]--;
    pc_add(&metadata->lv3_balls, -1, thread_id);
    metadata->lv3_locks[block] = 0;

    return true;
  }

  iceberg_lv3_node * current_node = head;

  for(uint64_t i = 0; i < metadata->lv3_sizes[block] - 1; ++i) {
#if PMEM
    assert(current_node->next_idx != -1);
    iceberg_lv3_node *next_node = &lv3_nodes[current_node->next_idx];
#else
    iceberg_lv3_node *next_node = current_node->next_node;
#endif

    if(next_node->key == key) {
#if PMEM
      current_node->next_idx = next_node->next_idx;
      pmem_memset_persist(next_node, 0, sizeof(*next_node));
#else
      iceberg_lv3_node * old_node = current_node->next_node;
      current_node->next_node = current_node->next_node->next_node;
      free(old_node);
#endif

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

static inline bool iceberg_lv3_remove(iceberg_table * table, KeyType key, hash h, uint8_t thread_id) {
  return iceberg_lv3_remove_internal(table, key, h.level1_raw_block, thread_id);
}

static inline bool iceberg_lv2_remove(iceberg_table * table, KeyType key, hash h, uint8_t thread_id) {
  iceberg_metadata * metadata = &table->metadata;

  for(int i = 0; i < D_CHOICES; ++i) {
    uint64_t raw_block = i == 0 ? h.level2_raw_block1 : h.level2_raw_block2;
    partition_block pb = decode_raw_block(table, raw_block);
    iceberg_lv2_block * blocks = table->level2[pb.partition];

#ifdef ENABLE_RESIZE
    // check if there's an active resize and block isn't fixed yet
    if (unlikely(is_lv2_resize_active(table) && raw_block >= (table->metadata.nblocks >> 1))) {
      uint64_t mask = ~(1ULL << (table->metadata.block_bits - 1));
      uint64_t old_index = raw_block & mask;
      uint64_t chunk = old_index / 8;
      partition_block pb = decode_raw_chunk(table, chunk);
      if (__atomic_load_n(&table->metadata.lv2_resize_marker[pb.partition][pb.block], __ATOMIC_SEQ_CST) == 0) { // not fixed yet
        partition_block old_pb = decode_raw_chunk(table, old_index);
        __mmask32 md_mask = slot_mask_32(metadata->lv2_md[old_pb.partition][old_pb.block].block_md, h.fingerprint) & ((1 << (C_LV2 + MAX_LG_LG_N / D_CHOICES)) - 1);
        uint8_t popct = __builtin_popcount(md_mask);
        iceberg_lv2_block * blocks = table->level2[old_pb.partition];
        for(uint8_t i = 0; i < popct; ++i) {
          uint64_t slot = word_select(md_mask, i);

          if (blocks[old_pb.block].slots[slot].key == key) {
            metadata->lv2_md[old_pb.partition][old_pb.block].block_md[slot] = 0;
            blocks[old_pb.block].slots[slot].key = blocks[old_pb.block].slots[slot].val = 0;
#if PMEM
            pmem_persist(&blocks[old_pb.block].slots[slot], sizeof(kv_pair));
#endif
            pc_add(&metadata->lv2_balls, -1, thread_id);
            return true;
          }
        }
      } else {
        // wait for the old block to be fixed
        uint64_t dest_chunk = raw_block / 8;
        pb = decode_raw_chunk(table, dest_chunk);
        while (__atomic_load_n(&table->metadata.lv2_resize_marker[pb.partition][pb.block], __ATOMIC_SEQ_CST) == 0)
          ;
      }
    }
#endif

    __mmask32 md_mask = slot_mask_32(metadata->lv2_md[pb.partition][pb.block].block_md, h.fingerprint) & ((1 << (C_LV2 + MAX_LG_LG_N / D_CHOICES)) - 1);
    uint8_t popct = __builtin_popcount(md_mask);

    for(uint8_t i = 0; i < popct; ++i) {
      uint64_t slot = word_select(md_mask, i);

      if (blocks[pb.block].slots[slot].key == key) {
        metadata->lv2_md[pb.partition][pb.block].block_md[slot] = 0;
        blocks[pb.block].slots[slot].key = blocks[pb.block].slots[slot].val = 0;
#if PMEM
        pmem_persist(&blocks[pb.block].slots[slot], sizeof(kv_pair));
#endif
        pc_add(&metadata->lv2_balls, -1, thread_id);
        return true;
      }
    }
  }

  return iceberg_lv3_remove(table, key, h, thread_id);
}

bool iceberg_remove(iceberg_table * table, KeyType key, uint8_t thread_id) {
  iceberg_metadata * metadata = &table->metadata;
  hash h = hash_key(table, &key);
  partition_block pb = decode_raw_block(table, h.level1_raw_block);
  iceberg_lv1_block * blocks = table->level1[pb.partition];

#ifdef ENABLE_RESIZE
  // check if there's an active resize and block isn't fixed yet
  if (unlikely(is_lv1_resize_active(table) && h.level1_raw_block >= (table->metadata.nblocks >> 1))) {
    uint64_t mask = ~(1ULL << (table->metadata.block_bits - 1));
    uint64_t old_index = h.level1_raw_block & mask;
    uint64_t chunk = old_index / 8;
    partition_block chunk_pb = decode_raw_chunk(table, chunk);
    if (__atomic_load_n(&table->metadata.lv1_resize_marker[chunk_pb.partition][chunk_pb.block], __ATOMIC_SEQ_CST) == 0) { // not fixed yet
      partition_block old_pb = decode_raw_block(table, old_index);
      __mmask64 md_mask = slot_mask_64(metadata->lv1_md[old_pb.partition][old_pb.block].block_md, h.fingerprint);
      uint8_t popct = __builtin_popcountll(md_mask);

      iceberg_lv1_block * blocks = table->level1[old_pb.partition];
      for(uint8_t i = 0; i < popct; ++i) {
        uint64_t slot = word_select(md_mask, i);

        if (blocks[old_index].slots[slot].key == key) {
          metadata->lv1_md[old_pb.partition][old_pb.block].block_md[slot] = 0;
          blocks[old_pb.block].slots[slot].key = blocks[old_pb.block].slots[slot].val = 0;
#if PMEM
          pmem_persist(&blocks[old_pb.block].slots[slot], sizeof(kv_pair));
#endif
          pc_add(&metadata->lv1_balls, -1, thread_id);
          return true;
        }
      }
    } else {
      // wait for the old block to be fixed
      uint64_t dest_chunk = h.level1_raw_block / 8;
      chunk_pb = decode_raw_chunk(table, dest_chunk);
      while (__atomic_load_n(&table->metadata.lv1_resize_marker[chunk_pb.partition][chunk_pb.block], __ATOMIC_SEQ_CST) == 0)
        ;
    }
  }
#endif

  lock_block((uint64_t *)&metadata->lv1_md[pb.partition][pb.block].block_md);
  __mmask64 md_mask = slot_mask_64(metadata->lv1_md[pb.partition][pb.block].block_md, h.fingerprint);
  uint8_t popct = __builtin_popcountll(md_mask);

  for(uint8_t i = 0; i < popct; ++i) {
    uint64_t slot = word_select(md_mask, i);

    if (blocks[pb.block].slots[slot].key == key) {
      metadata->lv1_md[pb.partition][pb.block].block_md[slot] = 0;
      blocks[pb.block].slots[slot].key = blocks[pb.block].slots[slot].val = 0;
#if PMEM
      pmem_persist(&blocks[pb.block].slots[slot], sizeof(kv_pair));
#endif
      pc_add(&metadata->lv1_balls, -1, thread_id);
      unlock_block((uint64_t *)&metadata->lv1_md[pb.partition][pb.block].block_md);
      return true;
    }
  }

  bool ret = iceberg_lv2_remove(table, key, h, thread_id);

  unlock_block((uint64_t *)&metadata->lv1_md[pb.partition][pb.block].block_md);
  return ret;
}

static inline bool iceberg_lv3_get_value_internal(iceberg_table * table, KeyType key, ValueType *value, uint64_t level1_raw_block) {
  uint64_t block = level1_raw_block % LEVEL3_BLOCKS;

  iceberg_metadata * metadata = &table->metadata;
  iceberg_lv3_list * lists = table->level3;

  if(likely(!metadata->lv3_sizes[block])) {
    return false;
  }

  while(__sync_lock_test_and_set(metadata->lv3_locks + block, 1));

#if PMEM
  iceberg_lv3_node * lv3_nodes = table->level3_nodes;
  assert(lists[block].head_idx != -1);
  iceberg_lv3_node * current_node = &lv3_nodes[lists[block].head_idx];
#else
  iceberg_lv3_node * current_node = lists[block].head;
#endif

  for(uint8_t i = 0; i < metadata->lv3_sizes[block]; ++i) {
    if(current_node->key == key) {
      *value = current_node->val;
      metadata->lv3_locks[block] = 0;
      return true;
    }
#if PMEM
    current_node = &lv3_nodes[current_node->next_idx];
#else
    current_node = current_node->next_node;
#endif
  }

  metadata->lv3_locks[block] = 0;

  return false;
}

static inline bool iceberg_lv3_get_value(iceberg_table * table, KeyType key, ValueType *value, uint64_t level1_raw_block) {
  return iceberg_lv3_get_value_internal(table, key, value, level1_raw_block);
}

static inline bool iceberg_lv2_get_value(iceberg_table * table, KeyType key, ValueType *value, hash h) {

  iceberg_metadata * metadata = &table->metadata;

  for(uint8_t i = 0; i < D_CHOICES; ++i) {
    uint64_t raw_block = i == 0 ? h.level2_raw_block1 : h.level2_raw_block2;
    partition_block pb = decode_raw_block(table, raw_block);
    iceberg_lv2_block * blocks = table->level2[pb.partition];

#ifdef ENABLE_RESIZE
    // check if there's an active resize and block isn't fixed yet
    if (unlikely(is_lv2_resize_active(table) && raw_block >= (table->metadata.nblocks >> 1))) {
      uint64_t mask = ~(1ULL << (table->metadata.block_bits - 1));
      uint64_t old_index = raw_block & mask;
      uint64_t chunk = old_index / 8;
      partition_block chunk_pb = decode_raw_chunk(table, chunk);
      if (__atomic_load_n(&table->metadata.lv2_resize_marker[chunk_pb.partition][chunk_pb.block], __ATOMIC_SEQ_CST) == 0) { // not fixed yet
        partition_block old_pb = decode_raw_block(table, old_index);
        __mmask32 md_mask = slot_mask_32(metadata->lv2_md[old_pb.partition][old_pb.block].block_md, h.fingerprint) & ((1 << (C_LV2 + MAX_LG_LG_N / D_CHOICES)) - 1);

        iceberg_lv2_block * blocks = table->level2[old_pb.partition];
        while (md_mask != 0) {
          int slot = __builtin_ctz(md_mask);
          md_mask = md_mask & ~(1U << slot);

          if (blocks[old_pb.block].slots[slot].key == key) {
            *value = blocks[old_pb.block].slots[slot].val;
            return true;
          }
        }
      } else {
        // wait for the old block to be fixed
        uint64_t dest_chunk = raw_block / 8;
        chunk_pb = decode_raw_chunk(table, dest_chunk);
        while (__atomic_load_n(&table->metadata.lv2_resize_marker[chunk_pb.partition][chunk_pb.block], __ATOMIC_SEQ_CST) == 0)
          ;
      }
    }
#endif

    __mmask32 md_mask = slot_mask_32(metadata->lv2_md[pb.partition][pb.block].block_md, h.fingerprint) & ((1 << (C_LV2 + MAX_LG_LG_N / D_CHOICES)) - 1);

    while (md_mask != 0) {
      int slot = __builtin_ctz(md_mask);
      md_mask = md_mask & ~(1U << slot);

      if (blocks[pb.block].slots[slot].key == key) {
        *value = blocks[pb.block].slots[slot].val;
        return true;
      }
    }

  }

  return iceberg_lv3_get_value(table, key, value, h.level1_raw_block);
}

__attribute__ ((always_inline)) inline bool
iceberg_get_value_internal(iceberg_table * table, KeyType key, ValueType *value, hash h, uint8_t thread_id) {
  iceberg_metadata * metadata = &table->metadata;

#ifdef ENABLE_RESIZE
  // check if there's an active resize and block isn't fixed yet
  if (unlikely(is_lv1_resize_active(table) && h.level1_raw_block >= (table->metadata.nblocks >> 1))) {
    uint64_t mask = ~(1ULL << (table->metadata.block_bits - 1));
    uint64_t old_index = h.level1_raw_block & mask;
    uint64_t chunk = old_index / 8;
    partition_block chunk_pb = decode_raw_chunk(table, chunk);
    if (__atomic_load_n(&table->metadata.lv1_resize_marker[chunk_pb.partition][chunk_pb.block], __ATOMIC_SEQ_CST) == 0) { // not fixed yet
      partition_block old_pb = decode_raw_block(table, old_index);
      __mmask64 md_mask = slot_mask_64(metadata->lv1_md[old_pb.partition][old_pb.block].block_md, h.fingerprint);

      iceberg_lv1_block * blocks = table->level1[old_pb.partition];
      while (md_mask != 0) {
        int slot = __builtin_ctzll(md_mask);
        md_mask = md_mask & ~(1ULL << slot);

        if (blocks[old_pb.block].slots[slot].key == key) {
          *value = blocks[old_pb.block].slots[slot].val;
          return true;
        }
      }
    } else {
      // wait for the old block to be fixed
      uint64_t dest_chunk = h.level1_raw_block / 8;
      chunk_pb = decode_raw_chunk(table, dest_chunk);
      while (__atomic_load_n(&table->metadata.lv1_resize_marker[chunk_pb.partition][chunk_pb.block], __ATOMIC_SEQ_CST) == 0)
        ;
    }
  }
#endif

  partition_block pb = decode_raw_block(table, h.level1_raw_block);
  iceberg_lv1_block * blocks = table->level1[pb.partition];
  __mmask64 md_mask = slot_mask_64(metadata->lv1_md[pb.partition][pb.block].block_md, h.fingerprint);

  while (md_mask != 0) {
    int slot = __builtin_ctzll(md_mask);
    md_mask = md_mask & ~(1ULL << slot);

    if (blocks[pb.block].slots[slot].key == key) {
      *value = blocks[pb.block].slots[slot].val;
      return true;
    }
  }

  bool ret = iceberg_lv2_get_value(table, key, value, h);

  return ret;
}


__attribute__ ((always_inline)) inline bool
iceberg_get_value(iceberg_table * table, KeyType key, ValueType *value, uint8_t thread_id) {
  hash h = hash_key(table, &key);

  return iceberg_get_value_internal(table, key, value, h, thread_id);
}

#ifdef ENABLE_RESIZE
static bool iceberg_nuke_key(iceberg_table * table, uint64_t level, uint64_t index, uint64_t slot, uint64_t thread_id) {
  partition_block pb = decode_raw_block(table, index);
  iceberg_metadata * metadata = &table->metadata;

  if (level == 1) {
    iceberg_lv1_block * blocks = table->level1[pb.partition];
    metadata->lv1_md[pb.partition][pb.block].block_md[slot] = 0;
    blocks[pb.block].slots[slot].key = blocks[pb.block].slots[slot].val = 0;
#if PMEM
    pmem_persist(&blocks[pb.block].slots[slot], sizeof(kv_pair));
#endif
    pc_add(&metadata->lv1_balls, -1, thread_id);
  } else if (level == 2) {
    iceberg_lv2_block * blocks = table->level2[pb.partition];
    metadata->lv2_md[pb.partition][pb.block].block_md[slot] = 0;
    blocks[pb.block].slots[slot].key = blocks[pb.block].slots[slot].val = 0;
#if PMEM
    pmem_persist(&blocks[pb.block].slots[slot], sizeof(kv_pair));
#endif
    pc_add(&metadata->lv2_balls, -1, thread_id);
  }

  return true;
}

static bool iceberg_lv1_move_block(iceberg_table * table, uint64_t bnum, uint8_t thread_id) {
  // grab a block
  uint64_t bctr = __atomic_fetch_add(&table->metadata.lv1_resize_ctr, 1, __ATOMIC_SEQ_CST);
  if (bctr >= (table->metadata.nblocks >> 1))
    return true;

  partition_block pb = decode_raw_block(table, bnum);
  // relocate items in level1
  for (uint64_t j = 0; j < (1 << SLOT_BITS); ++j) {
    KeyType key = table->level1[pb.partition][pb.block].slots[j].key;
    if (key == 0)
      continue;
    ValueType value = table->level1[pb.partition][pb.block].slots[j].val;

    hash h = hash_key(table, &key);

    // move to new location
    if (h.level1_raw_block != bnum) {
      partition_block local_pb = decode_raw_block(table, h.level1_raw_block);
      if (!iceberg_insert_internal(table, key, value, h, local_pb, thread_id)) {
        printf("Failed insert during resize lv1\n");
        exit(0);
      }
      if (!iceberg_nuke_key(table, 1, bnum, j, thread_id)) {
        printf("Failed remove during resize lv1. key: %" PRIu64 ", block: %" PRIu64"\n", key, bnum);
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
  uint64_t bctr = __atomic_fetch_add(&table->metadata.lv2_resize_ctr, 1, __ATOMIC_SEQ_CST);
  if (bctr >= (table->metadata.nblocks >> 1))
    return true;

  partition_block pb = decode_raw_block(table, bnum);
  uint64_t mask = ~(1ULL << (table->metadata.block_bits - 1));
  // relocate items in level2
  for (uint64_t j = 0; j < C_LV2 + MAX_LG_LG_N / D_CHOICES; ++j) {
    KeyType key = table->level2[pb.partition][pb.block].slots[j].key;
    if (key == 0)
      continue;
    ValueType value = table->level2[pb.partition][pb.block].slots[j].val;

    hash h = hash_key(table, &key);

    if ((h.level2_raw_block1 & mask) == bnum && h.level2_raw_block1 !=bnum) {
      partition_block local_pb = decode_raw_block(table, h.level2_raw_block1);
      if (!iceberg_lv2_insert_internal(table, key, value, h, local_pb, thread_id)) {
        if (!iceberg_lv2_insert(table, key, value, h, thread_id)) {
          printf("Failed insert during resize lv2\n");
          exit(0);
        }
      }
      if (!iceberg_nuke_key(table, 2, bnum, j, thread_id)) {
        printf("Failed remove during resize lv2\n");
        exit(0);
      }
    } else if ((h.level2_raw_block1 & mask) == bnum && h.level2_raw_block1 !=bnum) {
      partition_block local_pb = decode_raw_block(table, h.level2_raw_block2);
      if (!iceberg_lv2_insert_internal(table, key, value, h, local_pb, thread_id)) {
        if (!iceberg_lv2_insert(table, key, value, h, thread_id)) {
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
