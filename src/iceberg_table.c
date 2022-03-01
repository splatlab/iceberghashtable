#define _GNU_SOURCE
#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <stdlib.h>
#include <pthread.h>
#include <immintrin.h>
#include <tmmintrin.h>
#include <sys/mman.h>
#include <sys/sysinfo.h>
#include <math.h>

#include "hashutil.h"
#include "iceberg_precompute.h"
#include "iceberg_table.h"

#define likely(x)   __builtin_expect((x),1)
#define unlikely(x) __builtin_expect((x),0)

uint64_t seed[5] = { 12351327692179052ll, 23246347347385899ll, 35236262354132235ll, 13604702930934770ll, 57439820692984798ll };

static inline uint64_t nonzero_fprint(uint64_t hash) {
  return hash & ((1 << FPRINT_BITS) - 2) ? hash : hash | 2;
}

static inline uint64_t lv1_hash(KeyType key) {
  return nonzero_fprint(MurmurHash64A(&key, FPRINT_BITS, seed[0]));
}

static inline uint64_t lv2_hash(KeyType key, uint8_t i) {
  return nonzero_fprint(MurmurHash64A(&key, FPRINT_BITS, seed[i + 1]));
}

static inline uint8_t word_select(uint64_t val, int rank) {
  val = _pdep_u64(one[rank], val);
  return _tzcnt_u64(val);
}

uint64_t lv1_balls(iceberg_table * table) {
  pc_sync(&table->metadata.lv1_balls);
  return *(table->metadata.lv1_balls.global_counter);
}

uint64_t lv1_balls_aprox(iceberg_table * table) {
  return *(table->metadata.lv1_balls.global_counter);
}

uint64_t lv2_balls(iceberg_table * table) {
  pc_sync(&table->metadata.lv2_balls);
  return *(table->metadata.lv2_balls.global_counter);
}

uint64_t lv2_balls_aprox(iceberg_table * table) {
  return *(table->metadata.lv2_balls.global_counter);
}

uint64_t lv3_balls(iceberg_table * table) {
  pc_sync(&table->metadata.lv3_balls);
  return *(table->metadata.lv3_balls.global_counter);
}

uint64_t lv3_balls_aprox(iceberg_table * table) {
  return *(table->metadata.lv3_balls.global_counter);
}

uint64_t tot_balls(iceberg_table * table) {
  return lv1_balls(table) + lv2_balls(table) + lv3_balls(table);
}

uint64_t tot_balls_aprox(iceberg_table * table) {
  return lv1_balls_aprox(table) + lv2_balls_aprox(table) + lv3_balls_aprox(table);
}

static inline uint64_t total_capacity(iceberg_table * table) {
  return lv3_balls(table) + table->metadata.nblocks * ((1 << SLOT_BITS) + C_LV2 + MAX_LG_LG_N / D_CHOICES);
}

static inline uint64_t total_capacity_aprox(iceberg_table * table) {
  return lv3_balls_aprox(table) + table->metadata.nblocks * ((1 << SLOT_BITS) + C_LV2 + MAX_LG_LG_N / D_CHOICES);
}

inline double iceberg_load_factor(iceberg_table * table) {
  return (double)tot_balls(table) / (double)total_capacity(table);
}

static inline double iceberg_load_factor_aprox(iceberg_table * table) {
  return (double)tot_balls_aprox(table) / (double)total_capacity_aprox(table);
}

#ifdef ENABLE_RESIZE
bool need_resize(iceberg_table * table) {
  double lf = iceberg_load_factor_aprox(table);
  if (lf >= 0.85)
    return true;
  return false;
}
#endif

unsigned highestPowerof2(unsigned x)
{
    // check for the set bits
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;

    // Then we remove all but the top bit by xor'ing the
    // string of 1's with that string of 1's shifted one to
    // the left, and we end up with just the one top bit
    // followed by 0's.
    return x ^ (x >> 1);
}

const int tab64[64] = {
    63,  0, 58,  1, 59, 47, 53,  2,
    60, 39, 48, 27, 54, 33, 42,  3,
    61, 51, 37, 40, 49, 18, 28, 20,
    55, 30, 34, 11, 43, 14, 22,  4,
    62, 57, 46, 52, 38, 26, 32, 41,
    50, 36, 17, 19, 29, 10, 13, 21,
    56, 45, 25, 31, 35, 16,  9, 12,
    44, 24, 15,  8, 23,  7,  6,  5};

int log2_64 (uint64_t value)
{
    value |= value >> 1;
    value |= value >> 2;
    value |= value >> 4;
    value |= value >> 8;
    value |= value >> 16;
    value |= value >> 32;
    return tab64[((uint64_t)((value - (value >> 1))*0x07EDD5E59A4E28C2)) >> 58];
}

static inline void get_block_index_offset(iceberg_table * table, uint64_t index, uint64_t *bindex, uint64_t *boffset) {
  if (unlikely(index < table->metadata.init_size)) {
    *bindex = 0;
    *boffset = index;
    return;
  }
  *bindex = log2_64(index) - table->metadata.log_init_size + 1;
  *boffset = index - highestPowerof2(index);
}

static inline void get_marker_index_offset(iceberg_table * table, uint64_t index, uint64_t *mindex, uint64_t *moffset) {
  if (unlikely(index < table->metadata.init_size / 8)) {
    *mindex = 0;
    *moffset = index;
    return;
  }
  *mindex = log2_64(index) - (table->metadata.log_init_size - 3) + 1;
  *moffset = index - highestPowerof2(index);
}

static inline void split_hash(uint64_t hash, uint8_t *fprint, uint64_t *index, iceberg_metadata * metadata) {	
  *fprint = hash & ((1 << FPRINT_BITS) - 1);
  *index = (hash >> FPRINT_BITS) & ((1 << metadata->block_bits) - 1);
}

static inline uint32_t slot_mask_32(uint8_t * metadata, uint8_t fprint) {
  __m256i bcast = _mm256_set1_epi8(fprint);
  __m256i block = _mm256_loadu_si256((const __m256i *)(metadata));
  return _mm256_cmp_epi8_mask(bcast, block, _MM_CMPINT_EQ);
}

static inline uint64_t slot_mask_64(uint8_t * metadata, uint8_t fprint) {
  __m512i bcast = _mm512_set1_epi8(fprint);
  __m512i block = _mm512_loadu_si512((const __m512i *)(metadata));
  return _mm512_cmp_epi8_mask(bcast, block, _MM_CMPINT_EQ);
}

static uint64_t iceberg_block_load(iceberg_table * table, uint64_t index, uint8_t level) {
  uint64_t bindex, boffset;
  get_block_index_offset(table, index, &bindex, &boffset);
  if (level == 1) {
      __mmask64 mask64 = slot_mask_64(table->metadata.lv1_md[bindex][boffset].block_md, 0);
      return (1ULL << SLOT_BITS) - __builtin_popcountll(mask64); 
  } else if (level == 2) {
      __mmask32 mask32 = slot_mask_32(table->metadata.lv2_md[bindex][boffset].block_md, 0) & ((1 << (C_LV2 + MAX_LG_LG_N / D_CHOICES)) - 1);
      return (C_LV2 + MAX_LG_LG_N / D_CHOICES) - __builtin_popcountll(mask32);
  } else
      return table->metadata.lv3_sizes[bindex][boffset];
}

static uint64_t iceberg_table_load(iceberg_table * table) {
  uint64_t total = 0;

  for (uint8_t i = 1; i <= 3; ++i) {
    for (uint64_t j = 0; j < table->metadata.nblocks; ++j) {
      total += iceberg_block_load(table, j, i); 
    }
  }

  return total;
}

static double iceberg_block_load_factor(iceberg_table * table, uint64_t index, uint8_t level) {
  if (level == 1)
    return iceberg_block_load(table, index, level) / (double)(1ULL << SLOT_BITS);
  else if (level == 2)
    return iceberg_block_load(table, index, level) / (double)(C_LV2 + MAX_LG_LG_N / D_CHOICES);
  else
    return iceberg_block_load(table, index, level);
}

static inline size_t round_up(size_t n, size_t k) {
  size_t rem = n % k;
  if (rem == 0) {
    return n;
  }
  n += k - rem;
  return n;
}

int iceberg_init(iceberg_table *table, uint64_t log_slots) {
  memset(table, 0, sizeof(*table));

  uint64_t total_blocks = 1 << (log_slots - SLOT_BITS);
  uint64_t total_size_in_bytes = (sizeof(iceberg_lv1_block) + sizeof(iceberg_lv2_block) + sizeof(iceberg_lv1_block_md) + sizeof(iceberg_lv2_block_md)) * total_blocks;

  assert(table);

#if defined(HUGE_TLB)
  int mmap_flags = MAP_PRIVATE | MAP_ANONYMOUS | MAP_POPULATE | MAP_HUGETLB;
#else
  int mmap_flags = MAP_PRIVATE | MAP_ANONYMOUS | MAP_POPULATE;
#endif
  size_t level1_size = sizeof(iceberg_lv1_block) * total_blocks;
  //table->level1 = (iceberg_lv1_block *)malloc(level1_size);
  table->level1[0] = (iceberg_lv1_block *)mmap(NULL, level1_size, PROT_READ | PROT_WRITE, mmap_flags, 0, 0);
  if (!table->level1[0]) {
    perror("level1 malloc failed");
    exit(1);
  }
  size_t level2_size = sizeof(iceberg_lv2_block) * total_blocks;
  //table->level2 = (iceberg_lv2_block *)malloc(level2_size);
  table->level2[0] = (iceberg_lv2_block *)mmap(NULL, level2_size, PROT_READ | PROT_WRITE, mmap_flags, 0, 0);
  if (!table->level2[0]) {
    perror("level2 malloc failed");
    exit(1);
  }
  size_t level3_size = sizeof(iceberg_lv3_list) * total_blocks;
  table->level3[0] = (iceberg_lv3_list *)mmap(NULL, level3_size, PROT_READ | PROT_WRITE, mmap_flags, 0, 0);
  if (!table->level3[0]) {
    perror("level3 malloc failed");
    exit(1);
  }

  table->metadata.total_size_in_bytes = total_size_in_bytes;
  table->metadata.nslots = 1 << log_slots;
  table->metadata.nblocks = total_blocks;
  table->metadata.block_bits = log_slots - SLOT_BITS;
  table->metadata.init_size = total_blocks;
  table->metadata.log_init_size = log2(total_blocks);

  uint32_t procs = get_nprocs();
  pc_init(&table->metadata.lv1_balls, &table->metadata.lv1_ctr, procs, 1000);
  pc_init(&table->metadata.lv2_balls, &table->metadata.lv2_ctr, procs, 1000);
  pc_init(&table->metadata.lv3_balls, &table->metadata.lv3_ctr, procs, 1000);

  size_t lv1_md_size = sizeof(iceberg_lv1_block_md) * total_blocks + 64;
  //table->metadata.lv1_md = (iceberg_lv1_block_md *)malloc(sizeof(iceberg_lv1_block_md) * total_blocks);
  table->metadata.lv1_md[0] = (iceberg_lv1_block_md *)mmap(NULL, lv1_md_size, PROT_READ | PROT_WRITE, mmap_flags, 0, 0);
  if (!table->metadata.lv1_md[0]) {
    perror("lv1_md malloc failed");
    exit(1);
  }
  //table->metadata.lv2_md = (iceberg_lv2_block_md *)malloc(sizeof(iceberg_lv2_block_md) * total_blocks);
  size_t lv2_md_size = sizeof(iceberg_lv2_block_md) * total_blocks + 32;
  table->metadata.lv2_md[0] = (iceberg_lv2_block_md *)mmap(NULL, lv2_md_size, PROT_READ | PROT_WRITE, mmap_flags, 0, 0);
  if (!table->metadata.lv2_md[0]) {
    perror("lv2_md malloc failed");
    exit(1);
  }
  table->metadata.lv3_sizes[0] = (uint64_t *)mmap(NULL, sizeof(uint64_t) * total_blocks, PROT_READ | PROT_WRITE, mmap_flags, 0, 0);
  if (!table->metadata.lv3_sizes[0]) {
    perror("lv3_sizes malloc failed");
    exit(1);
  }
  table->metadata.lv3_locks[0] = (uint8_t *)mmap(NULL, sizeof(uint8_t) * total_blocks, PROT_READ | PROT_WRITE, mmap_flags, 0, 0);
  if (!table->metadata.lv3_locks[0]) {
    perror("lv3_locks malloc failed");
    exit(1);
  }

#ifdef ENABLE_RESIZE
  table->metadata.resize_cnt = 0;
  table->metadata.lv1_resize_ctr = total_blocks;
  table->metadata.lv2_resize_ctr = total_blocks;
  table->metadata.lv3_resize_ctr = total_blocks;

  // create one marker for 8 blocks.
  size_t resize_marker_size = sizeof(uint8_t) * total_blocks / 8;
  table->metadata.lv1_resize_marker[0] = (uint8_t *)mmap(NULL, resize_marker_size, PROT_READ | PROT_WRITE, mmap_flags, 0, 0);
  if (!table->metadata.lv1_resize_marker[0]) {
    perror("level1 resize ctr malloc failed");
    exit(1);
  }
  table->metadata.lv2_resize_marker[0] = (uint8_t *)mmap(NULL, resize_marker_size, PROT_READ | PROT_WRITE, mmap_flags, 0, 0);
  if (!table->metadata.lv2_resize_marker[0]) {
    perror("level2 resize ctr malloc failed");
    exit(1);
  }
  table->metadata.lv3_resize_marker[0] = (uint8_t *)mmap(NULL, resize_marker_size, PROT_READ | PROT_WRITE, mmap_flags, 0, 0);
  if (!table->metadata.lv3_resize_marker[0]) {
    perror("level3 resize ctr malloc failed");
    exit(1);
  }
  memset(table->metadata.lv1_resize_marker[0], 1, resize_marker_size);
  memset(table->metadata.lv2_resize_marker[0], 1, resize_marker_size);
  memset(table->metadata.lv3_resize_marker[0], 1, resize_marker_size);

  table->metadata.marker_sizes[0] = resize_marker_size;

  rw_lock_init(&table->metadata.rw_lock);
#endif

  for (uint64_t i = 0; i < total_blocks; ++i) {
    for (uint64_t j = 0; j < (1 << SLOT_BITS); ++j) {
      table->metadata.lv1_md[0][i].block_md[j] = 0;
      table->level1[0][i].slots[j].key = table->level1[0][i].slots[j].val = 0;
    }

    for (uint64_t j = 0; j < C_LV2 + MAX_LG_LG_N / D_CHOICES; ++j) {
      table->metadata.lv2_md[0][i].block_md[j] = 0;
      table->level2[0][i].slots[j].key = table->level2[0][i].slots[j].val = 0;
    }

    table->level3[0]->head = NULL;
    table->metadata.lv3_sizes[0][i] = table->metadata.lv3_locks[0][i] = 0;
  }

  return 0;
}

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

static inline bool is_lv3_resize_active(iceberg_table * table) {
  uint64_t half_mark = table->metadata.nblocks >> 1;
  uint64_t lv3_ctr = __atomic_load_n(&table->metadata.lv3_resize_ctr, __ATOMIC_SEQ_CST);
  return lv3_ctr < half_mark;
}

static bool is_resize_active(iceberg_table * table) {
  return is_lv3_resize_active(table) || is_lv2_resize_active(table) || is_lv1_resize_active(table); 
}

static bool iceberg_setup_resize(iceberg_table * table) {
  // grab write lock
  if (!write_lock(&table->metadata.rw_lock, TRY_ONCE_LOCK))
    return false;

  if (unlikely(!need_resize(table))) {
    write_unlock(&table->metadata.rw_lock);
    return false;
  }
  if (is_resize_active(table)) {
    // finish the current resize
    iceberg_end(table);
    write_unlock(&table->metadata.rw_lock);
    return false;
  }

  printf("Setting up resize\n");
  /*printf("Current stats: \n");*/
  
  /*printf("Load factor: %f\n", iceberg_load_factor(table));*/
  /*printf("Number level 1 inserts: %ld\n", lv1_balls(table));*/
  /*printf("Number level 2 inserts: %ld\n", lv2_balls(table));*/
  /*printf("Number level 3 inserts: %ld\n", lv3_balls(table));*/
  /*printf("Total inserts: %ld\n", tot_balls(table));*/

#if defined(HUGE_TLB)
  int mmap_flags = MAP_PRIVATE | MAP_ANONYMOUS | MAP_POPULATE | MAP_HUGETLB;
#else
  int mmap_flags = MAP_PRIVATE | MAP_ANONYMOUS | MAP_POPULATE;
#endif

  // compute new sizes
  uint64_t cur_blocks = table->metadata.nblocks;
  uint64_t resize_cnt = table->metadata.resize_cnt + 1;

  // Allocate new table and metadata

  // alloc level1
  size_t level1_size = sizeof(iceberg_lv1_block) * cur_blocks;
  table->level1[resize_cnt] = (iceberg_lv1_block *)mmap(NULL, level1_size, PROT_READ | PROT_WRITE, mmap_flags, 0, 0);
  if (table->level1[resize_cnt] == (void *)-1) {
    perror("level1 resize failed");
    exit(1);
  }

  // alloc level2
  size_t level2_size = sizeof(iceberg_lv2_block) * cur_blocks;
  table->level2[resize_cnt] = (iceberg_lv2_block *)mmap(NULL, level2_size, PROT_READ | PROT_WRITE, mmap_flags, 0, 0);
  if (table->level2[resize_cnt] == (void *)-1) {
    perror("level2 resize failed");
    exit(1);
  }

  // alloc level3
  size_t level3_size = sizeof(iceberg_lv3_list) * cur_blocks;
  table->level3[resize_cnt] = (iceberg_lv3_list *)mmap(NULL, level3_size, PROT_READ | PROT_WRITE, mmap_flags, 0, 0);
  if (table->level3[resize_cnt] == (void *)-1) {
    perror("level3 resize failed");
    exit(1);
  }

  // alloc level1 metadata
  size_t lv1_md_size = sizeof(iceberg_lv1_block_md) * cur_blocks + 64;
  table->metadata.lv1_md[resize_cnt] = (iceberg_lv1_block_md *)mmap(NULL, lv1_md_size, PROT_READ | PROT_WRITE, mmap_flags, 0, 0);
  if (table->metadata.lv1_md[resize_cnt] == (void *)-1) {
    perror("lv1_md resize failed");
    exit(1);
  }

  // alloc level2 metadata
  size_t lv2_md_size = sizeof(iceberg_lv2_block_md) * cur_blocks + 32;
  table->metadata.lv2_md[resize_cnt] = (iceberg_lv2_block_md *)mmap(NULL, lv2_md_size, PROT_READ | PROT_WRITE, mmap_flags, 0, 0);
  if (table->metadata.lv2_md[resize_cnt] == (void *)-1) {
    perror("lv2_md resize failed");
    exit(1);
  }

  // alloc level3 metadata (sizes, locks)
  size_t lv3_sizes_size = sizeof(uint64_t) * cur_blocks;
  table->metadata.lv3_sizes[resize_cnt] = (uint64_t *)mmap(NULL, lv3_sizes_size, PROT_READ | PROT_WRITE, mmap_flags, 0, 0);
  if (table->metadata.lv3_sizes[resize_cnt] == (void *)-1) {
    perror("lv3_sizes resize failed");
    exit(1);
  }

  size_t lv3_locks_size = sizeof(uint8_t) * cur_blocks;
  table->metadata.lv3_locks[resize_cnt] = (uint8_t *)mmap(NULL, lv3_locks_size, PROT_READ | PROT_WRITE, mmap_flags, 0, 0);
  if (table->metadata.lv3_locks[resize_cnt] == (void *)-1) {
    perror("lv3_locks remap failed");
    exit(1);
  }

  // alloc resize markers
  // resize_marker_size
  size_t resize_marker_size = sizeof(uint8_t) * cur_blocks / 8;
  table->metadata.lv1_resize_marker[resize_cnt] = (uint8_t *)mmap(NULL, resize_marker_size, PROT_READ | PROT_WRITE, mmap_flags, 0, 0);
  if (table->metadata.lv1_resize_marker[resize_cnt] == (void *)-1) {
    perror("level1 resize failed");
    exit(1);
  }

  table->metadata.lv2_resize_marker[resize_cnt] = (uint8_t *)mmap(NULL, resize_marker_size, PROT_READ | PROT_WRITE, mmap_flags, 0, 0);
  if (table->metadata.lv2_resize_marker[resize_cnt] == (void *)-1) {
    perror("level1 resize failed");
    exit(1);
  }

  table->metadata.lv3_resize_marker[resize_cnt] = (uint8_t *)mmap(NULL, resize_marker_size, PROT_READ | PROT_WRITE, mmap_flags, 0, 0);
  if (table->metadata.lv3_resize_marker[resize_cnt] == (void *)-1) {
    perror("level1 resize failed");
    exit(1);
  }

  table->metadata.marker_sizes[resize_cnt] = resize_marker_size;
  // resetting the resize markers.
  for (uint64_t i = 0;  i <= resize_cnt; ++i) {
    memset(table->metadata.lv1_resize_marker[i], 0, table->metadata.marker_sizes[i]);
    memset(table->metadata.lv2_resize_marker[i], 0, table->metadata.marker_sizes[i]);
    memset(table->metadata.lv3_resize_marker[i], 0, table->metadata.marker_sizes[i]);
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

  // reset the block ctr 
  table->metadata.lv1_resize_ctr = 0;
  table->metadata.lv2_resize_ctr = 0;
  table->metadata.lv3_resize_ctr = 0;

  printf("Setting up finished\n");
  write_unlock(&table->metadata.rw_lock);
  return true;
}

static bool iceberg_lv1_move_block(iceberg_table * table, uint64_t bnum, uint8_t thread_id);
static bool iceberg_lv2_move_block(iceberg_table * table, uint64_t bnum, uint8_t thread_id);
static bool iceberg_lv3_move_block(iceberg_table * table, uint64_t bnum, uint8_t thread_id);

// finish moving blocks that are left during the last resize.
void iceberg_end(iceberg_table * table) {
  if (is_lv1_resize_active(table)) {
    for (uint64_t j = 0; j < table->metadata.nblocks / 8; ++j) {
      uint64_t chunk_idx = j;
      uint64_t mindex, moffset;
      get_marker_index_offset(table, chunk_idx, &mindex, &moffset);
      // if fixing is needed set the marker
      if (!__sync_lock_test_and_set(&table->metadata.lv1_resize_marker[mindex][moffset], 1)) {
        for (uint8_t i = 0; i < 8; ++i) {
          uint64_t idx = chunk_idx * 8 + i;
          iceberg_lv1_move_block(table, idx, 0);
        }
        // set the marker for the dest block
        uint64_t dest_chunk_idx = chunk_idx + table->metadata.nblocks / 8 / 2;
        uint64_t mindex, moffset;
        get_marker_index_offset(table, dest_chunk_idx, &mindex, &moffset);
        __sync_lock_test_and_set(&table->metadata.lv1_resize_marker[mindex][moffset], 1);
      }
    }
  }
  if (is_lv2_resize_active(table)) {
    for (uint64_t j = 0; j < table->metadata.nblocks / 8; ++j) {
      uint64_t chunk_idx = j;
      uint64_t mindex, moffset;
      get_marker_index_offset(table, chunk_idx, &mindex, &moffset);
      // if fixing is needed set the marker
      if (!__sync_lock_test_and_set(&table->metadata.lv2_resize_marker[mindex][moffset], 1)) {
        for (uint8_t i = 0; i < 8; ++i) {
          uint64_t idx = chunk_idx * 8 + i;
          iceberg_lv2_move_block(table, idx, 0);
        }
        // set the marker for the dest block
        uint64_t dest_chunk_idx = chunk_idx + table->metadata.nblocks / 8 / 2;
        uint64_t mindex, moffset;
        get_marker_index_offset(table, dest_chunk_idx, &mindex, &moffset);
        __sync_lock_test_and_set(&table->metadata.lv2_resize_marker[mindex][moffset], 1);
      }
    }
  }
  if (is_lv3_resize_active(table)) {
    for (uint64_t j = 0; j < table->metadata.nblocks / 8; ++j) {
      uint64_t chunk_idx = j;
      uint64_t mindex, moffset;
      get_marker_index_offset(table, chunk_idx, &mindex, &moffset);
      // if fixing is needed set the marker
      if (!__sync_lock_test_and_set(&table->metadata.lv3_resize_marker[mindex][moffset], 1)) {
        for (uint8_t i = 0; i < 8; ++i) {
          uint64_t idx = chunk_idx * 8 + i;
          iceberg_lv3_move_block(table, idx, 0);
        }
        // set the marker for the dest block
        uint64_t dest_chunk_idx = chunk_idx + table->metadata.nblocks / 8 / 2;
        uint64_t mindex, moffset;
        get_marker_index_offset(table, dest_chunk_idx, &mindex, &moffset);
        __sync_lock_test_and_set(&table->metadata.lv3_resize_marker[mindex][moffset], 1);
      }
    }
  }

  /*printf("Final resize done.\n");*/
  /*printf("Final resize done. Table load: %ld\n", iceberg_table_load(table));*/
}
#endif

static inline bool iceberg_lv3_insert(iceberg_table * table, KeyType key, ValueType value, uint64_t lv3_index, uint8_t thread_id) {

#ifdef ENABLE_RESIZE
  if (unlikely(lv3_index < (table->metadata.nblocks >> 1) && is_lv3_resize_active(table))) {
    uint64_t chunk_idx = lv3_index / 8;
    uint64_t mindex, moffset;
    get_marker_index_offset(table, chunk_idx, &mindex, &moffset);
    // if fixing is needed set the marker
    if (!__sync_lock_test_and_set(&table->metadata.lv3_resize_marker[mindex][moffset], 1)) {
      for (uint8_t i = 0; i < 8; ++i) {
        uint64_t idx = chunk_idx * 8 + i;
        /*printf("LV3 Before: Moving block: %ld load: %f\n", idx, iceberg_block_load(table, idx, 3));*/
        iceberg_lv3_move_block(table, idx, thread_id);
        /*printf("LV3 After: Moving block: %ld load: %f\n", idx, iceberg_block_load(table, idx, 3));*/
      }
      // set the marker for the dest block
      uint64_t dest_chunk_idx = chunk_idx + table->metadata.nblocks / 8 / 2;
      uint64_t mindex, moffset;
      get_marker_index_offset(table, dest_chunk_idx, &mindex, &moffset);
      __sync_lock_test_and_set(&table->metadata.lv3_resize_marker[mindex][moffset], 1);
    }
  }
#endif

  uint64_t bindex, boffset;
  get_block_index_offset(table, lv3_index, &bindex, &boffset);
  iceberg_metadata * metadata = &table->metadata;
  iceberg_lv3_list * lists = table->level3[bindex];

  while(__sync_lock_test_and_set(metadata->lv3_locks[bindex] + boffset, 1));

  iceberg_lv3_node * new_node = (iceberg_lv3_node *)malloc(sizeof(iceberg_lv3_node));
  new_node->key = key;
  new_node->val = value;
  new_node->next_node = lists[boffset].head;
  lists[boffset].head = new_node;

  metadata->lv3_sizes[bindex][boffset]++;
  pc_add(&metadata->lv3_balls, 1, thread_id);
  metadata->lv3_locks[bindex][boffset] = 0;

  return true;
}

static inline bool iceberg_lv2_insert_internal(iceberg_table * table, KeyType key, ValueType value, uint8_t fprint, uint64_t index, uint8_t thread_id) {
  uint64_t bindex, boffset;
  get_block_index_offset(table, index, &bindex, &boffset);

  iceberg_metadata * metadata = &table->metadata;
  iceberg_lv2_block * blocks = table->level2[bindex];

  __mmask32 md_mask = slot_mask_32(metadata->lv2_md[bindex][boffset].block_md, 0) & ((1 << (C_LV2 + MAX_LG_LG_N / D_CHOICES)) - 1);
  uint8_t popct = __builtin_popcountll(md_mask);

  for(uint8_t i = 0; i < popct; ++i) {
    uint8_t slot = word_select(md_mask, i);

    if(__sync_bool_compare_and_swap(metadata->lv2_md[bindex][boffset].block_md + slot, 0, 1)) {
      pc_add(&metadata->lv2_balls, 1, thread_id);
      blocks[boffset].slots[slot].key = key;
      blocks[boffset].slots[slot].val = value;

      metadata->lv2_md[bindex][boffset].block_md[slot] = fprint;
      return true;
    }
  }

  return false;
}

static inline bool iceberg_lv2_insert(iceberg_table * table, KeyType key, ValueType value, uint64_t lv3_index, uint8_t thread_id) {

  iceberg_metadata * metadata = &table->metadata;

  if (metadata->lv2_ctr == (int64_t)(C_LV2 * metadata->nblocks))
    return iceberg_lv3_insert(table, key, value, lv3_index, thread_id);

  uint8_t fprint1, fprint2;
  uint64_t index1, index2;

  split_hash(lv2_hash(key, 0), &fprint1, &index1, metadata);
  split_hash(lv2_hash(key, 1), &fprint2, &index2, metadata);

  uint64_t bindex1, boffset1, bindex2, boffset2;
  get_block_index_offset(table, index1, &bindex1, &boffset1);
  get_block_index_offset(table, index2, &bindex2, &boffset2);

  __mmask32 md_mask1 = slot_mask_32(metadata->lv2_md[bindex1][boffset1].block_md, 0) & ((1 << (C_LV2 + MAX_LG_LG_N / D_CHOICES)) - 1);
  __mmask32 md_mask2 = slot_mask_32(metadata->lv2_md[bindex2][boffset2].block_md, 0) & ((1 << (C_LV2 + MAX_LG_LG_N / D_CHOICES)) - 1);

  uint8_t popct1 = __builtin_popcountll(md_mask1);
  uint8_t popct2 = __builtin_popcountll(md_mask2);

  if(popct2 > popct1) {
    fprint1 = fprint2;
    index1 = index2;
    bindex1 = bindex2;
    boffset1 = boffset2;
    md_mask1 = md_mask2;
    popct1 = popct2;
  }
  
#ifdef ENABLE_RESIZE
  // move blocks if resize is active and not already moved.
  if (unlikely(index1 < (table->metadata.nblocks >> 1) && is_lv2_resize_active(table))) {
    uint64_t chunk_idx = index1 / 8;
    uint64_t mindex, moffset;
    get_marker_index_offset(table, chunk_idx, &mindex, &moffset);
    // if fixing is needed set the marker
    if (!__sync_lock_test_and_set(&table->metadata.lv2_resize_marker[mindex][moffset], 1)) {
      for (uint8_t i = 0; i < 8; ++i) {
        uint64_t idx = chunk_idx * 8 + i;
        /*printf("LV2 Before: Moving block: %ld load: %f\n", idx, iceberg_block_load(table, idx, 2));*/
        iceberg_lv2_move_block(table, idx, thread_id);
        /*printf("LV2 After: Moving block: %ld load: %f\n", idx, iceberg_block_load(table, idx, 2));*/
      }
      // set the marker for the dest block
      uint64_t dest_chunk_idx = chunk_idx + table->metadata.nblocks / 8 / 2;
      uint64_t mindex, moffset;
      get_marker_index_offset(table, dest_chunk_idx, &mindex, &moffset);
      __sync_lock_test_and_set(&table->metadata.lv2_resize_marker[mindex][moffset], 1);
    }
  }
#endif

  if (iceberg_lv2_insert_internal(table, key, value, fprint1, index1, thread_id))
    return true;

  return iceberg_lv3_insert(table, key, value, lv3_index, thread_id);
}

static bool iceberg_insert_internal(iceberg_table * table, KeyType key, ValueType value, uint8_t fprint, uint64_t index, uint8_t thread_id) {
  uint64_t bindex, boffset;
  get_block_index_offset(table, index, &bindex, &boffset);
  
  iceberg_metadata * metadata = &table->metadata;
  iceberg_lv1_block * blocks = table->level1[bindex];	

  __mmask64 md_mask = slot_mask_64(metadata->lv1_md[bindex][boffset].block_md, 0);

  uint8_t popct = __builtin_popcountll(md_mask);

  for(uint8_t i = 0; i < popct; ++i) {
    uint8_t slot = word_select(md_mask, i);

    if(__sync_bool_compare_and_swap(metadata->lv1_md[bindex][boffset].block_md + slot, 0, 1)) {
      pc_add(&metadata->lv1_balls, 1, thread_id);
      blocks[boffset].slots[slot].key = key;
      blocks[boffset].slots[slot].val = value;

      metadata->lv1_md[bindex][boffset].block_md[slot] = fprint;
      return true;
    }
  }

  return false;
}

bool iceberg_insert(iceberg_table * table, KeyType key, ValueType value, uint8_t thread_id) {

#ifdef ENABLE_RESIZE
  if (unlikely(need_resize(table))) {
    iceberg_setup_resize(table);
  }

  /*if (unlikely(!read_lock(&table->metadata.rw_lock, WAIT_FOR_LOCK, thread_id)))*/
    /*return false;*/
#endif

  iceberg_metadata * metadata = &table->metadata;
  uint8_t fprint;
  uint64_t index;

  split_hash(lv1_hash(key), &fprint, &index, metadata);

#ifdef ENABLE_RESIZE
  // move blocks if resize is active and not already moved.
  if (unlikely(index < (table->metadata.nblocks >> 1) && is_lv1_resize_active(table))) {
    uint64_t chunk_idx = index / 8;
    uint64_t mindex, moffset;
    get_marker_index_offset(table, chunk_idx, &mindex, &moffset);
    // if fixing is needed set the marker
    if (!__sync_lock_test_and_set(&table->metadata.lv1_resize_marker[mindex][moffset], 1)) {
      for (uint8_t i = 0; i < 8; ++i) {
        uint64_t idx = chunk_idx * 8 + i;
        /*printf("LV1 Before: Moving block: %ld load: %f\n", idx, iceberg_block_load(table, idx, 1));*/
        iceberg_lv1_move_block(table, idx, thread_id);
        /*printf("LV1 After: Moving block: %ld load: %f\n", idx, iceberg_block_load(table, idx, 1));*/
      }
      // set the marker for the dest block
      uint64_t dest_chunk_idx = chunk_idx + table->metadata.nblocks / 8 / 2;
      uint64_t mindex, moffset;
      get_marker_index_offset(table, dest_chunk_idx, &mindex, &moffset);
      __sync_lock_test_and_set(&table->metadata.lv1_resize_marker[mindex][moffset], 1);
    }
  }
#endif

  bool ret = iceberg_insert_internal(table, key, value, fprint, index, thread_id);
  if (!ret)
    ret = iceberg_lv2_insert(table, key, value, index, thread_id);

#ifdef ENABLE_RESIZE
  /*read_unlock(&table->metadata.rw_lock, thread_id);*/
#endif

  return ret;
}

static inline bool iceberg_lv3_remove_internal(iceberg_table * table, KeyType key, uint64_t lv3_index, uint8_t thread_id) {
  uint64_t bindex, boffset;
  get_block_index_offset(table, lv3_index, &bindex, &boffset);

  iceberg_metadata * metadata = &table->metadata;
  iceberg_lv3_list * lists = table->level3[bindex];

  while(__sync_lock_test_and_set(metadata->lv3_locks[bindex] + boffset, 1));

  if(metadata->lv3_sizes[bindex][boffset] == 0) return false;

  if(lists[boffset].head->key == key) {
    iceberg_lv3_node * old_head = lists[boffset].head;
    lists[boffset].head = lists[boffset].head->next_node;
    free(old_head);

    metadata->lv3_sizes[bindex][boffset]--;
    pc_add(&metadata->lv3_balls, -1, thread_id);
    metadata->lv3_locks[bindex][boffset] = 0;

    return true;
  }

  iceberg_lv3_node * current_node = lists[boffset].head;

  for(uint64_t i = 0; i < metadata->lv3_sizes[bindex][boffset] - 1; ++i) {
    if(current_node->next_node->key == key) {
      iceberg_lv3_node * old_node = current_node->next_node;
      current_node->next_node = current_node->next_node->next_node;
      free(old_node);

      metadata->lv3_sizes[bindex][boffset]--;
      pc_add(&metadata->lv3_balls, -1, thread_id);
      metadata->lv3_locks[bindex][boffset] = 0;

      return true;
    }

    current_node = current_node->next_node;
  }

  metadata->lv3_locks[bindex][boffset] = 0;
  return false;
}

static inline bool iceberg_lv3_remove(iceberg_table * table, KeyType key, uint64_t lv3_index, uint8_t thread_id) {

  bool ret = iceberg_lv3_remove_internal(table, key, lv3_index, thread_id);

  if (ret)
    return true;

#ifdef ENABLE_RESIZE
  // check if there's an active resize and block isn't fixed yet
  if (unlikely(lv3_index >= (table->metadata.nblocks >> 1) && is_lv3_resize_active(table))) {
    uint64_t mask = ~(1ULL << (table->metadata.block_bits - 1));
    uint64_t old_index = lv3_index & mask;
    uint64_t chunk_idx = old_index / 8;
    uint64_t mindex, moffset;
    get_marker_index_offset(table, chunk_idx, &mindex, &moffset);
    if (__atomic_load_n(&table->metadata.lv3_resize_marker[mindex][moffset], __ATOMIC_SEQ_CST) == 0) { // not fixed yet
      return iceberg_lv3_remove_internal(table, key, old_index, thread_id);
    } else {
      // wait for the old block to be fixed
      uint64_t dest_chunk_idx = lv3_index / 8;
      get_marker_index_offset(table, dest_chunk_idx, &mindex, &moffset);
      while (__atomic_load_n(&table->metadata.lv3_resize_marker[mindex][moffset], __ATOMIC_SEQ_CST) == 0)
        ;
    }
  }
#endif

  return false;
}

static inline bool iceberg_lv2_remove(iceberg_table * table, KeyType key, uint64_t lv3_index, uint8_t thread_id) {
  iceberg_metadata * metadata = &table->metadata;

  for(int i = 0; i < D_CHOICES; ++i) {
    uint8_t fprint;
    uint64_t index;

    split_hash(lv2_hash(key, i), &fprint, &index, metadata);

    uint64_t bindex, boffset;
    get_block_index_offset(table, index, &bindex, &boffset);
    iceberg_lv2_block * blocks = table->level2[bindex];

#ifdef ENABLE_RESIZE
    // check if there's an active resize and block isn't fixed yet
    if (unlikely(index >= (table->metadata.nblocks >> 1) && is_lv2_resize_active(table))) {
      uint64_t mask = ~(1ULL << (table->metadata.block_bits - 1));
      uint64_t old_index = index & mask;
      uint64_t chunk_idx = old_index / 8;
      uint64_t mindex, moffset;
      get_marker_index_offset(table, chunk_idx, &mindex, &moffset);
      if (__atomic_load_n(&table->metadata.lv2_resize_marker[mindex][moffset], __ATOMIC_SEQ_CST) == 0) { // not fixed yet
        uint64_t old_bindex, old_boffset;
        get_marker_index_offset(table, old_index, &old_bindex, &old_boffset);
        __mmask32 md_mask = slot_mask_32(metadata->lv2_md[old_bindex][old_boffset].block_md, fprint) & ((1 << (C_LV2 + MAX_LG_LG_N / D_CHOICES)) - 1);
        uint8_t popct = __builtin_popcount(md_mask);
        iceberg_lv2_block * blocks = table->level2[old_bindex];
        for(uint8_t i = 0; i < popct; ++i) {
          uint8_t slot = word_select(md_mask, i);

          if (blocks[old_boffset].slots[slot].key == key) {
            metadata->lv2_md[old_bindex][old_boffset].block_md[slot] = 0;
            blocks[old_boffset].slots[slot].key = blocks[old_boffset].slots[slot].val = 0;
            pc_add(&metadata->lv2_balls, -1, thread_id);
            return true;
          }
        }
      } else {
        // wait for the old block to be fixed
        uint64_t dest_chunk_idx = index / 8;
        get_marker_index_offset(table, dest_chunk_idx, &mindex, &moffset);
        while (__atomic_load_n(&table->metadata.lv2_resize_marker[mindex][moffset], __ATOMIC_SEQ_CST) == 0)
          ;
      }
    }
#endif

    __mmask32 md_mask = slot_mask_32(metadata->lv2_md[bindex][boffset].block_md, fprint) & ((1 << (C_LV2 + MAX_LG_LG_N / D_CHOICES)) - 1);
    uint8_t popct = __builtin_popcount(md_mask);

    for(uint8_t i = 0; i < popct; ++i) {
      uint8_t slot = word_select(md_mask, i);

      if (blocks[boffset].slots[slot].key == key) {
        metadata->lv2_md[bindex][boffset].block_md[slot] = 0;
        blocks[boffset].slots[slot].key = blocks[boffset].slots[slot].val = 0;
        pc_add(&metadata->lv2_balls, -1, thread_id);
        return true;
      }
    }
  }

  return iceberg_lv3_remove(table, key, lv3_index, thread_id);
}

bool iceberg_remove(iceberg_table * table, KeyType key, uint8_t thread_id) {

#ifdef ENABLE_RESIZE
  /*if (unlikely(!read_lock(&table->metadata.rw_lock, WAIT_FOR_LOCK, thread_id)))*/
    /*return false;*/
#endif

  iceberg_metadata * metadata = &table->metadata;
  uint8_t fprint;
  uint64_t index;

  split_hash(lv1_hash(key), &fprint, &index, metadata);
 
  uint64_t bindex, boffset;
  get_block_index_offset(table, index, &bindex, &boffset);
  iceberg_lv1_block * blocks = table->level1[bindex];

#ifdef ENABLE_RESIZE
  // check if there's an active resize and block isn't fixed yet
  if (unlikely(index >= (table->metadata.nblocks >> 1) && is_lv1_resize_active(table))) {
    uint64_t mask = ~(1ULL << (table->metadata.block_bits - 1));
    uint64_t old_index = index & mask;
    uint64_t chunk_idx = old_index / 8;
    uint64_t mindex, moffset;
    get_marker_index_offset(table, chunk_idx, &mindex, &moffset);
    if (__atomic_load_n(&table->metadata.lv1_resize_marker[mindex][moffset], __ATOMIC_SEQ_CST) == 0) { // not fixed yet
      uint64_t old_bindex, old_boffset;
      get_block_index_offset(table, old_index, &old_bindex, &old_boffset);
      __mmask64 md_mask = slot_mask_64(metadata->lv1_md[old_bindex][old_boffset].block_md, fprint);
      uint8_t popct = __builtin_popcountll(md_mask);

      iceberg_lv1_block * blocks = table->level1[old_bindex];
      for(uint8_t i = 0; i < popct; ++i) {
        uint8_t slot = word_select(md_mask, i);

        if (blocks[old_index].slots[slot].key == key) {
          metadata->lv1_md[old_bindex][old_boffset].block_md[slot] = 0;
          blocks[old_boffset].slots[slot].key = blocks[old_boffset].slots[slot].val = 0;
          pc_add(&metadata->lv1_balls, -1, thread_id);
          /*read_unlock(&table->metadata.rw_lock, thread_id);*/
          return true;
        }
      }
    } else {
      // wait for the old block to be fixed
      uint64_t dest_chunk_idx = index / 8;
      get_marker_index_offset(table, dest_chunk_idx, &mindex, &moffset);
      while (__atomic_load_n(&table->metadata.lv1_resize_marker[mindex][moffset], __ATOMIC_SEQ_CST) == 0)
        ;
    }
  }
#endif

  __mmask64 md_mask = slot_mask_64(metadata->lv1_md[bindex][boffset].block_md, fprint);
  uint8_t popct = __builtin_popcountll(md_mask);

  for(uint8_t i = 0; i < popct; ++i) {
    uint8_t slot = word_select(md_mask, i);

    if (blocks[boffset].slots[slot].key == key) {
      metadata->lv1_md[bindex][boffset].block_md[slot] = 0;
      blocks[boffset].slots[slot].key = blocks[boffset].slots[slot].val = 0;
      pc_add(&metadata->lv1_balls, -1, thread_id);
#ifdef ENABLE_RESIZE
      /*read_unlock(&table->metadata.rw_lock, thread_id);*/
#endif
      return true;
    }
  }

  bool ret = iceberg_lv2_remove(table, key, index, thread_id);

#ifdef ENABLE_RESIZE
  /*read_unlock(&table->metadata.rw_lock, thread_id);*/
#endif

  return ret;
}

static inline bool iceberg_lv3_get_value_internal(iceberg_table * table, KeyType key, ValueType *value, uint64_t lv3_index) {
  uint64_t bindex, boffset;
  get_block_index_offset(table, lv3_index, &bindex, &boffset);

  iceberg_metadata * metadata = &table->metadata;
  iceberg_lv3_list * lists = table->level3[bindex];

  while(__sync_lock_test_and_set(metadata->lv3_locks[bindex] + boffset, 1));

  if(likely(!metadata->lv3_sizes[bindex][boffset])) {
    metadata->lv3_locks[bindex][boffset] = 0;
    return false;
  }

  iceberg_lv3_node * current_node = lists[boffset].head;

  for(uint8_t i = 0; i < metadata->lv3_sizes[bindex][boffset]; ++i) {

    if(current_node->key == key) {
      *value = current_node->val;
      metadata->lv3_locks[bindex][boffset] = 0;
      return true;
    }

    current_node = current_node->next_node;
  }

  metadata->lv3_locks[bindex][boffset] = 0;

  return false;
}

static inline bool iceberg_lv3_get_value(iceberg_table * table, KeyType key, ValueType *value, uint64_t lv3_index) {
#ifdef ENABLE_RESIZE
  // check if there's an active resize and block isn't fixed yet
  if (unlikely(lv3_index >= (table->metadata.nblocks >> 1) && is_lv3_resize_active(table))) {
    uint64_t mask = ~(1ULL << (table->metadata.block_bits - 1));
    uint64_t old_index = lv3_index & mask;
    uint64_t chunk_idx = old_index / 8;
    uint64_t mindex, moffset;
    get_marker_index_offset(table, chunk_idx, &mindex, &moffset);
    if (__atomic_load_n(&table->metadata.lv3_resize_marker[mindex][moffset], __ATOMIC_SEQ_CST) == 0) { // not fixed yet
      return iceberg_lv3_get_value_internal(table, key, value, old_index);
    } else {
      // wait for the old block to be fixed
      uint64_t dest_chunk_idx = lv3_index / 8;
      get_marker_index_offset(table, dest_chunk_idx, &mindex, &moffset);
      while (__atomic_load_n(&table->metadata.lv3_resize_marker[mindex][moffset], __ATOMIC_SEQ_CST) == 0)
        ;
    }
  }
#endif

  return iceberg_lv3_get_value_internal(table, key, value, lv3_index);
}

static inline bool iceberg_lv2_get_value(iceberg_table * table, KeyType key, ValueType *value, uint64_t lv3_index) {

  iceberg_metadata * metadata = &table->metadata;

  for(uint8_t i = 0; i < D_CHOICES; ++i) {
    uint8_t fprint;
    uint64_t index;

    split_hash(lv2_hash(key, i), &fprint, &index, metadata);

    uint64_t bindex, boffset;
    get_block_index_offset(table, index, &bindex, &boffset);
    iceberg_lv2_block * blocks = table->level2[bindex];

#ifdef ENABLE_RESIZE
    // check if there's an active resize and block isn't fixed yet
    if (unlikely(index >= (table->metadata.nblocks >> 1) && is_lv2_resize_active(table))) {
      uint64_t mask = ~(1ULL << (table->metadata.block_bits - 1));
      uint64_t old_index = index & mask;
      uint64_t chunk_idx = old_index / 8;
      uint64_t mindex, moffset;
      get_marker_index_offset(table, chunk_idx, &mindex, &moffset);
      if (__atomic_load_n(&table->metadata.lv2_resize_marker[mindex][moffset], __ATOMIC_SEQ_CST) == 0) { // not fixed yet
        uint64_t old_bindex, old_boffset;
        get_block_index_offset(table, old_index, &old_bindex, &old_boffset);
        __mmask32 md_mask = slot_mask_32(metadata->lv2_md[old_bindex][old_boffset].block_md, fprint) & ((1 << (C_LV2 + MAX_LG_LG_N / D_CHOICES)) - 1);

        iceberg_lv2_block * blocks = table->level2[old_bindex];
        while (md_mask != 0) {
          int slot = __builtin_ctz(md_mask);
          md_mask = md_mask & ~(1U << slot);

          if (blocks[old_boffset].slots[slot].key == key) {
            *value = blocks[old_boffset].slots[slot].val;
            return true;
          }
        }
      } else {
        // wait for the old block to be fixed
        uint64_t dest_chunk_idx = index / 8;
        get_marker_index_offset(table, dest_chunk_idx, &mindex, &moffset);
        while (__atomic_load_n(&table->metadata.lv2_resize_marker[mindex][moffset], __ATOMIC_SEQ_CST) == 0)
          ;
      }
    }
#endif

    __mmask32 md_mask = slot_mask_32(metadata->lv2_md[bindex][boffset].block_md, fprint) & ((1 << (C_LV2 + MAX_LG_LG_N / D_CHOICES)) - 1);

    while (md_mask != 0) {
      int slot = __builtin_ctz(md_mask);
      md_mask = md_mask & ~(1U << slot);

      if (blocks[boffset].slots[slot].key == key) {
        *value = blocks[boffset].slots[slot].val;
        return true;
      }
    }

  }

  return iceberg_lv3_get_value(table, key, value, lv3_index);
}

bool iceberg_get_value(iceberg_table * table, KeyType key, ValueType *value, uint8_t thread_id) {

#ifdef ENABLE_RESIZE
  /*if (unlikely(!read_lock(&table->metadata.rw_lock, WAIT_FOR_LOCK, thread_id)))*/
    /*return false;*/
#endif

  iceberg_metadata * metadata = &table->metadata;
  
  uint8_t fprint;
  uint64_t index;

  split_hash(lv1_hash(key), &fprint, &index, metadata);

  uint64_t bindex, boffset;
  get_block_index_offset(table, index, &bindex, &boffset);
  iceberg_lv1_block * blocks = table->level1[bindex];

#ifdef ENABLE_RESIZE
  // check if there's an active resize and block isn't fixed yet
  if (unlikely(index >= (table->metadata.nblocks >> 1) && is_lv1_resize_active(table))) {
    uint64_t mask = ~(1ULL << (table->metadata.block_bits - 1));
    uint64_t old_index = index & mask;
    uint64_t chunk_idx = old_index / 8;
    uint64_t mindex, moffset;
    get_marker_index_offset(table, chunk_idx, &mindex, &moffset);
    if (__atomic_load_n(&table->metadata.lv1_resize_marker[mindex][moffset], __ATOMIC_SEQ_CST) == 0) { // not fixed yet
      uint64_t old_bindex, old_boffset;
      get_block_index_offset(table, old_index, &old_bindex, &old_boffset);
      __mmask64 md_mask = slot_mask_64(metadata->lv1_md[old_bindex][old_boffset].block_md, fprint);

      iceberg_lv1_block * blocks = table->level1[old_bindex];
      while (md_mask != 0) {
        int slot = __builtin_ctzll(md_mask);
        md_mask = md_mask & ~(1ULL << slot);

        if (blocks[old_boffset].slots[slot].key == key) {
          *value = blocks[old_boffset].slots[slot].val;
          /*read_unlock(&table->metadata.rw_lock, thread_id);*/
          return true;
        }
      }
    } else {
      // wait for the old block to be fixed
      uint64_t dest_chunk_idx = index / 8;
      get_marker_index_offset(table, dest_chunk_idx, &mindex, &moffset);
      while (__atomic_load_n(&table->metadata.lv1_resize_marker[mindex][moffset], __ATOMIC_SEQ_CST) == 0)
        ;
    }
  }
#endif

  __mmask64 md_mask = slot_mask_64(metadata->lv1_md[bindex][boffset].block_md, fprint);

  while (md_mask != 0) {
    int slot = __builtin_ctzll(md_mask);
    md_mask = md_mask & ~(1ULL << slot);

    if (blocks[boffset].slots[slot].key == key) {
      *value = blocks[boffset].slots[slot].val;
#ifdef ENABLE_RESIZE
      /*read_unlock(&table->metadata.rw_lock, thread_id);*/
#endif
      return true;
    }
  }

  bool ret = iceberg_lv2_get_value(table, key, value, index);

#ifdef ENABLE_RESIZE
  /*read_unlock(&table->metadata.rw_lock, thread_id);*/
#endif

  return ret;
}

#ifdef ENABLE_RESIZE
static bool iceberg_nuke_key(iceberg_table * table, uint64_t level, uint64_t index, uint64_t slot, uint64_t thread_id) {
  uint64_t bindex, boffset;
  get_block_index_offset(table, index, &bindex, &boffset);
  iceberg_metadata * metadata = &table->metadata;

  if (level == 1) {
    iceberg_lv1_block * blocks = table->level1[bindex];
    metadata->lv1_md[bindex][boffset].block_md[slot] = 0;
    blocks[boffset].slots[slot].key = blocks[boffset].slots[slot].val = 0;
    pc_add(&metadata->lv1_balls, -1, thread_id);
  } else if (level == 2) {
    iceberg_lv2_block * blocks = table->level2[bindex];
    metadata->lv2_md[bindex][boffset].block_md[slot] = 0;
    blocks[boffset].slots[slot].key = blocks[boffset].slots[slot].val = 0;
    pc_add(&metadata->lv2_balls, -1, thread_id);
  }

  return true;
}

static bool iceberg_lv1_move_block(iceberg_table * table, uint64_t bnum, uint8_t thread_id) {
  // grab a block 
  uint64_t bctr = __atomic_fetch_add(&table->metadata.lv1_resize_ctr, 1, __ATOMIC_SEQ_CST);
  if (bctr >= (table->metadata.nblocks >> 1))
    return true;

  uint64_t bindex, boffset;
  get_block_index_offset(table, bnum, &bindex, &boffset);
  // relocate items in level1
  for (uint64_t j = 0; j < (1 << SLOT_BITS); ++j) {
    KeyType key = table->level1[bindex][boffset].slots[j].key;
    if (key == 0)
      continue;
    ValueType value = table->level1[bindex][boffset].slots[j].val;
    uint8_t fprint;
    uint64_t index;

    split_hash(lv1_hash(key), &fprint, &index, &table->metadata);
    // move to new location
    if (index != bnum) {
      if (!iceberg_insert_internal(table, key, value, fprint, index, thread_id)) {
        printf("Failed insert during resize lv1\n");
        exit(0);
      }
      if (!iceberg_nuke_key(table, 1, bnum, j, thread_id)) {
        printf("Failed remove during resize lv1. key: %" PRIu64 ", block: %ld\n", key, bnum);
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

  uint64_t bindex, boffset;
  get_block_index_offset(table, bnum, &bindex, &boffset);
  uint64_t mask = ~(1ULL << (table->metadata.block_bits - 1));
  // relocate items in level2
  for (uint64_t j = 0; j < C_LV2 + MAX_LG_LG_N / D_CHOICES; ++j) {
    KeyType key = table->level2[bindex][boffset].slots[j].key;
    if (key == 0)
      continue;
    ValueType value = table->level2[bindex][boffset].slots[j].val;
    uint8_t fprint;
    uint64_t index;

    split_hash(lv1_hash(key), &fprint, &index, &table->metadata);

    for(int i = 0; i < D_CHOICES; ++i) {
      uint8_t l2fprint;
      uint64_t l2index;

      split_hash(lv2_hash(key, i), &l2fprint, &l2index, &table->metadata);

      // move to new location
      if ((l2index & mask) == bnum && l2index != bnum) {
        if (!iceberg_lv2_insert_internal(table, key, value, l2fprint, l2index, thread_id)) {
          if (!iceberg_lv2_insert(table, key, value, index, thread_id)) {
            printf("Failed insert during resize lv2\n");
            exit(0);
          }
        }
        if (!iceberg_nuke_key(table, 2, bnum, j, thread_id)) {
          printf("Failed remove during resize lv2\n");
          exit(0);
        }
        break;
        //ValueType *val;
        //if (!iceberg_get_value(table, key, &val, thread_id)) {
        //printf("Key not found during resize lv2: %ld\n", key);
        //exit(0);
        //}
      }
    }
  }

  return false;
}

static bool iceberg_lv3_move_block(iceberg_table * table, uint64_t bnum, uint8_t thread_id) {
  // grab a block 
  uint64_t bctr = __atomic_fetch_add(&table->metadata.lv3_resize_ctr, 1, __ATOMIC_SEQ_CST);
  if (bctr >= (table->metadata.nblocks >> 1))
    return true;

  uint64_t bindex, boffset;
  get_block_index_offset(table, bnum, &bindex, &boffset);
  // relocate items in level3
  if(unlikely(table->metadata.lv3_sizes[bindex][boffset])) {
    iceberg_lv3_node * current_node = table->level3[bindex][boffset].head;

    while (current_node != NULL) {
      KeyType key = current_node->key;
      ValueType value = current_node->val;

      uint8_t fprint;
      uint64_t index;

      split_hash(lv1_hash(key), &fprint, &index, &table->metadata);
      // move to new location
      if (index != bnum) {
        current_node = current_node->next_node;
        if (!iceberg_lv3_insert(table, key, value, index, thread_id)) {
          printf("Failed insert during resize lv3\n");
          exit(0);
        }
        if (!iceberg_lv3_remove(table, key, bnum, thread_id)) {
          printf("Failed remove during resize lv3: %" PRIu64 "\n", key);
          exit(0);
        }
        // ValueType *val;
        //if (!iceberg_get_value(table, key, &val, thread_id)) {
        // printf("Key not found during resize lv3: %ld\n", key);
        //exit(0);
        //}
      }
      else
        current_node = current_node->next_node;
    }
  }

  return false;
}
#endif
