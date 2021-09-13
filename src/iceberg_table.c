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
#include <fcntl.h>
#include <libpmem.h>

#include "hashutil.h"
#include "iceberg_precompute.h"
#include "iceberg_table.h"

#define likely(x)   __builtin_expect((x),1)
#define unlikely(x) __builtin_expect((x),0)

#define PMEM_PATH "/mnt/pmem1"
#define FILENAME_LEN 1024
#define NUM_LEVEL3_NODES 2048

#define MiB (1024 * 1024)

uint64_t seed[5] = { 12351327692179052ll, 23246347347385899ll, 35236262354132235ll, 13604702930934770ll, 57439820692984798ll };

pthread_cond_t resize_cond = PTHREAD_COND_INITIALIZER;
pthread_mutex_t resize_mutex = PTHREAD_MUTEX_INITIALIZER;

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

void split_hash(KeyType key,
                uint8_t *slot_choice,
                uint8_t *fprint,
                uint64_t *index,
                iceberg_metadata * metadata,
                uint64_t seed_idx)
{
  uint64_t hash = MurmurHash64A(&key, sizeof(KeyType), seed[seed_idx]);
  *slot_choice = hash & ((1 << FPRINT_BITS) - 1);
  *fprint = *slot_choice <= 1 ? 2 : *slot_choice;
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
  if (level == 1) {
      __mmask64 mask64 = slot_mask_64(table->metadata.lv1_md[index].block_md, 0);
      return (1ULL << SLOT_BITS) - __builtin_popcountll(mask64);
  } else if (level == 2) {
      __mmask32 mask32 = slot_mask_32(table->metadata.lv2_md[index].block_md, 0) & ((1 << (C_LV2 + MAX_LG_LG_N / D_CHOICES)) - 1);
      return (C_LV2 + MAX_LG_LG_N / D_CHOICES) - __builtin_popcountll(mask32);
  } else
      return table->metadata.lv3_sizes[index];
}

//static uint64_t iceberg_table_load(iceberg_table * table) {
//  uint64_t total = 0;
//
//  for (uint8_t i = 1; i <= 3; ++i) {
//    for (uint64_t j = 0; j < table->metadata.nblocks; ++j) {
//      total += iceberg_block_load(table, j, i); 
//    }
//  }
//
//  return total;
//}

//static double iceberg_block_load_factor(iceberg_table * table, uint64_t index, uint8_t level) {
//  if (level == 1)
//    return iceberg_block_load(table, index, level) / (double)(1ULL << SLOT_BITS);
//  else if (level == 2)
//    return iceberg_block_load(table, index, level) / (double)(C_LV2 + MAX_LG_LG_N / D_CHOICES);
//  else
//    return iceberg_block_load(table, index, level);
//}

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

  size_t mapped_len;
  int is_pmem;

  size_t level1_size = round_up(sizeof(iceberg_lv1_block) * total_blocks, 2 * 1024 * 1024);
  table->metadata.level1_size = level1_size;
  char level1_filename[FILENAME_LEN];
  sprintf(level1_filename, "%s/level1", PMEM_PATH);
  if ((table->level1 = (iceberg_lv1_block *)pmem_map_file(level1_filename,
          10ULL * 1024 * 1024 * 1024, PMEM_FILE_CREATE, 0666, &mapped_len,
          &is_pmem)) == NULL) {
    perror("pmem_map_file");
    exit(1);
  }
  assert(is_pmem);
  //assert(mapped_len == level1_size);
  pmem_memset_persist(table->level1, 0, 10ULL * 1024 * 1024 * 1024);

  size_t level2_size = round_up(sizeof(iceberg_lv2_block) * total_blocks, 2 * 1024 * 1024);
  table->metadata.level2_size = level2_size;
  char level2_filename[FILENAME_LEN];
  sprintf(level2_filename, "%s/level2", PMEM_PATH);
  if ((table->level2 = (iceberg_lv2_block *)pmem_map_file(level2_filename,
          10ULL * 1024 * 1024 * 1024, PMEM_FILE_CREATE, 0666, &mapped_len,
          &is_pmem)) == NULL) {
    perror("pmem_map_file");
  }
  assert(is_pmem);
  //assert(mapped_len == level2_size);
  pmem_memset_persist(table->level2, 0, 10ULL * 1024 * 1024 * 1024);

  size_t level3_size = sizeof(iceberg_lv3_list) * total_blocks;
  table->metadata.level3_size = level3_size;
  char level3_filename[FILENAME_LEN];
  sprintf(level3_filename, "%s/level3", PMEM_PATH);
  if ((table->level3 = (iceberg_lv3_list *)pmem_map_file(level3_filename,
          10ULL * 1024 * 1024 * 1024, PMEM_FILE_CREATE, 0666, &mapped_len,
          &is_pmem)) == NULL) {
    perror("pmem_map_file");
    exit(1);
  }
  assert(is_pmem);
  //assert(mapped_len == level3_size);
  pmem_memset_persist(table->level3, 0, 10ULL * 1024 * 1024 * 1024);

  size_t level3_nodes_size = NUM_LEVEL3_NODES * sizeof(iceberg_lv3_node);
  char level3_nodes_filename[FILENAME_LEN];
  sprintf(level3_nodes_filename, "%s/level3_data", PMEM_PATH);
  table->level3_nodes =
    (iceberg_lv3_node *)pmem_map_file(level3_nodes_filename,
        level3_nodes_size, PMEM_FILE_CREATE, 0666,
        &mapped_len, &is_pmem);
  if (table->level3_nodes == NULL) {
    perror("pmem_map_file");
    exit(1);
  }
  assert(is_pmem);
  assert(mapped_len == level3_nodes_size);
  pmem_memset_persist(table->level3_nodes, 0, level3_nodes_size);

  table->metadata.total_size_in_bytes = total_size_in_bytes;
  table->metadata.nslots = 1 << log_slots;
  table->metadata.nblocks = total_blocks;
  table->metadata.block_bits = log_slots - SLOT_BITS;

  uint32_t procs = get_nprocs();
  pc_init(&table->metadata.lv1_balls, &table->metadata.lv1_ctr, procs, 1000);
  pc_init(&table->metadata.lv2_balls, &table->metadata.lv2_ctr, procs, 1000);
  pc_init(&table->metadata.lv3_balls, &table->metadata.lv3_ctr, procs, 1000);

  size_t lv1_md_size = sizeof(iceberg_lv1_block_md) * total_blocks + 64;
  table->metadata.lv1_md = (iceberg_lv1_block_md *)mmap(NULL, lv1_md_size, PROT_READ | PROT_WRITE, mmap_flags, 0, 0);
  if (!table->metadata.lv1_md) {
    perror("lv1_md malloc failed");
    exit(1);
  }

  size_t lv2_md_size = sizeof(iceberg_lv2_block_md) * total_blocks + 32;
  table->metadata.lv2_md = (iceberg_lv2_block_md *)mmap(NULL, lv2_md_size, PROT_READ | PROT_WRITE, mmap_flags, 0, 0);
  if (!table->metadata.lv2_md) {
    perror("lv2_md malloc failed");
    exit(1);
  }
  table->metadata.lv3_sizes = (uint64_t *)mmap(NULL, sizeof(uint64_t) * total_blocks, PROT_READ | PROT_WRITE, mmap_flags, 0, 0);
  if (!table->metadata.lv3_sizes) {
    perror("lv3_sizes malloc failed");
    exit(1);
  }
  table->metadata.lv3_locks = (uint8_t *)mmap(NULL, sizeof(uint8_t) * total_blocks, PROT_READ | PROT_WRITE, mmap_flags, 0, 0);
  if (!table->metadata.lv3_locks) {
    perror("lv3_locks malloc failed");
    exit(1);
  }

#ifdef ENABLE_RESIZE
  table->metadata.lv1_resize_ctr = total_blocks;
  table->metadata.lv2_resize_ctr = total_blocks;
  table->metadata.lv3_resize_ctr = total_blocks;

  // create one marker for 8 blocks.
  size_t resize_marker_size = sizeof(uint8_t) * total_blocks / 8;
  table->metadata.lv1_resize_marker = (uint8_t *)mmap(NULL, resize_marker_size, PROT_READ | PROT_WRITE, mmap_flags, 0, 0);
  if (!table->metadata.lv1_resize_marker) {
    perror("level1 resize ctr malloc failed");
    exit(1);
  }
  table->metadata.lv2_resize_marker = (uint8_t *)mmap(NULL, resize_marker_size, PROT_READ | PROT_WRITE, mmap_flags, 0, 0);
  if (!table->metadata.lv2_resize_marker) {
    perror("level2 resize ctr malloc failed");
    exit(1);
  }
  table->metadata.lv3_resize_marker = (uint8_t *)mmap(NULL, resize_marker_size, PROT_READ | PROT_WRITE, mmap_flags, 0, 0);
  if (!table->metadata.lv3_resize_marker) {
    perror("level3 resize ctr malloc failed");
    exit(1);
  }
  memset(table->metadata.lv1_resize_marker, 1, resize_marker_size);
  memset(table->metadata.lv2_resize_marker, 1, resize_marker_size);
  memset(table->metadata.lv3_resize_marker, 1, resize_marker_size);

  rw_lock_init(&table->metadata.rw_lock);
#endif

  pmem_memset_persist(table->level1, 0, level1_size);
  pmem_memset_persist(table->level2, 0, level2_size);
  memset(table->level3, 0, level3_size);

  for (uint64_t i = 0; i < total_blocks; i++) {
    table->level3[i].head_idx = -1;
  }
  pmem_persist(table->level3, level3_size);
  pmem_memset_persist(table->level3_nodes, 0, level3_nodes_size);

  memset((char *)table->metadata.lv1_md, 0, lv1_md_size);
  memset((char *)table->metadata.lv2_md, 0, lv2_md_size);

  memset(table->metadata.lv3_sizes, 0, total_blocks * sizeof(uint64_t));
  memset(table->metadata.lv3_locks, 0, total_blocks * sizeof(uint8_t));

  return 0;
}

uint64_t iceberg_dismount(iceberg_table * table) {

  size_t total_blocks = table->metadata.nblocks;

  size_t lv1_md_size = sizeof(iceberg_lv1_block_md) * total_blocks + 64;
  munmap(table->metadata.lv1_md, lv1_md_size);
  size_t lv2_md_size = sizeof(iceberg_lv2_block_md) * total_blocks + 32;
  munmap(table->metadata.lv2_md, lv2_md_size);

  munmap(table->metadata.lv3_sizes, sizeof(uint64_t) * total_blocks);
  munmap(table->metadata.lv3_locks, sizeof(uint8_t) * total_blocks);

  pc_destructor(&table->metadata.lv1_balls);
  pc_destructor(&table->metadata.lv2_balls);
  pc_destructor(&table->metadata.lv3_balls);

  size_t level1_size = round_up(sizeof(iceberg_lv1_block) * total_blocks, 2 * 1024 * 1024);
  pmem_unmap(table->level1, level1_size);

  size_t level2_size = round_up(sizeof(iceberg_lv2_block) * total_blocks, 2 * 1024 * 1024);
  pmem_unmap(table->level2, level2_size);

  size_t level3_size = sizeof(iceberg_lv3_list) * total_blocks;
  pmem_unmap(table->level3, level3_size);

  size_t level3_nodes_size = NUM_LEVEL3_NODES * sizeof(iceberg_lv3_node);
  pmem_unmap(table->level3_nodes, level3_nodes_size);

#ifdef ENABLE_RESIZE
  size_t resize_marker_size = sizeof(uint8_t) * total_blocks / 8;
  munmap(table->metadata.lv1_resize_marker, resize_marker_size);
  munmap(table->metadata.lv2_resize_marker, resize_marker_size);
  munmap(table->metadata.lv3_resize_marker, resize_marker_size);
#endif

  return table->metadata.block_bits + SLOT_BITS;
}

int iceberg_mount(iceberg_table *table, uint64_t log_slots) {
  memset(table, 0, sizeof(*table));

  uint64_t total_blocks = 1 << (log_slots - SLOT_BITS);
  uint64_t total_size_in_bytes = (sizeof(iceberg_lv1_block) + sizeof(iceberg_lv2_block) + sizeof(iceberg_lv1_block_md) + sizeof(iceberg_lv2_block_md)) * total_blocks;

  assert(table);

#if defined(HUGE_TLB)
  int mmap_flags = MAP_PRIVATE | MAP_ANONYMOUS | MAP_POPULATE | MAP_HUGETLB;
#else
  int mmap_flags = MAP_PRIVATE | MAP_ANONYMOUS | MAP_POPULATE;
#endif

  size_t mapped_len;
  int is_pmem;

  size_t level1_size = round_up(sizeof(iceberg_lv1_block) * total_blocks, 2 * 1024 * 1024);
  table->metadata.level1_size = level1_size;
  char level1_filename[FILENAME_LEN];
  sprintf(level1_filename, "%s/level1", PMEM_PATH);
  if ((table->level1 = (iceberg_lv1_block *)pmem_map_file(level1_filename,
          0, 0, 0666, &mapped_len, &is_pmem)) == NULL) {
    perror("pmem_map_file");
    exit(1);
  }
  assert(is_pmem);
  assert(mapped_len == level1_size);

  size_t level2_size = round_up(sizeof(iceberg_lv2_block) * total_blocks, 2 * 1024 * 1024);
  table->metadata.level2_size = level2_size;
  char level2_filename[FILENAME_LEN];
  sprintf(level2_filename, "%s/level2", PMEM_PATH);
  if ((table->level2 = (iceberg_lv2_block *)pmem_map_file(level2_filename,
          0, 0, 0666, &mapped_len, &is_pmem)) == NULL) {
    perror("pmem_map_file");
    assert(is_pmem);
    assert(mapped_len == level2_size);
  }

  size_t level3_size = sizeof(iceberg_lv3_list) * total_blocks;
  table->metadata.level3_size = level3_size;
  char level3_filename[FILENAME_LEN];
  sprintf(level3_filename, "%s/level3", PMEM_PATH);
  if ((table->level3 = (iceberg_lv3_list *)pmem_map_file(level3_filename,
          0, 0, 0666, &mapped_len, &is_pmem)) == NULL) {
    perror("pmem_map_file");
    exit(1);
  }

  assert(is_pmem);
  assert(mapped_len == level3_size);

  size_t level3_nodes_size = NUM_LEVEL3_NODES * sizeof(iceberg_lv3_node);
  char level3_nodes_filename[FILENAME_LEN];
  sprintf(level3_nodes_filename, "%s/level3_data", PMEM_PATH);
  table->level3_nodes =
    (iceberg_lv3_node *)pmem_map_file(level3_nodes_filename,
        0, 0, 0666, &mapped_len, &is_pmem);
  if (table->level3_nodes == NULL) {
    perror("pmem_map_file");
    exit(1);
  }
  assert(is_pmem);
  assert(mapped_len == level3_nodes_size);

  table->metadata.total_size_in_bytes = total_size_in_bytes;
  table->metadata.nslots = 1 << log_slots;
  table->metadata.nblocks = total_blocks;
  table->metadata.block_bits = log_slots - SLOT_BITS;

  uint32_t procs = get_nprocs();
  pc_init(&table->metadata.lv1_balls, &table->metadata.lv1_ctr, procs, 1000);
  pc_init(&table->metadata.lv2_balls, &table->metadata.lv2_ctr, procs, 1000);
  pc_init(&table->metadata.lv3_balls, &table->metadata.lv3_ctr, procs, 1000);

  size_t lv1_md_size = sizeof(iceberg_lv1_block_md) * total_blocks + 64;
  table->metadata.lv1_md = (iceberg_lv1_block_md *)mmap(NULL, lv1_md_size, PROT_READ | PROT_WRITE, mmap_flags, 0, 0);
  if (!table->metadata.lv1_md) {
    perror("lv1_md malloc failed");
    exit(1);
  }

  size_t lv2_md_size = sizeof(iceberg_lv2_block_md) * total_blocks + 32;
  table->metadata.lv2_md = (iceberg_lv2_block_md *)mmap(NULL, lv2_md_size, PROT_READ | PROT_WRITE, mmap_flags, 0, 0);
  if (!table->metadata.lv2_md) {
    perror("lv2_md malloc failed");
    exit(1);
  }
  table->metadata.lv3_sizes = (uint64_t *)mmap(NULL, sizeof(uint64_t) * total_blocks, PROT_READ | PROT_WRITE, mmap_flags, 0, 0);
  if (!table->metadata.lv3_sizes) {
    perror("lv3_sizes malloc failed");
    exit(1);
  }
  table->metadata.lv3_locks = (uint8_t *)mmap(NULL, sizeof(uint8_t) * total_blocks, PROT_READ | PROT_WRITE, mmap_flags, 0, 0);
  if (!table->metadata.lv3_locks) {
    perror("lv3_locks malloc failed");
    exit(1);
  }

  // recover the metadata
  // level 1
  for (uint64_t i = 0; i < total_blocks; i++) {
    iceberg_lv1_block *block = &table->level1[i];
    uint8_t *block_md = table->metadata.lv1_md[i].block_md;
    for (uint64_t slot = 0; slot < 1 << SLOT_BITS; slot++) {
      KeyType key = block->slots[slot].key;
      if (key != 0) {
        uint8_t slot_choice;
        uint8_t fprint;
        uint64_t index;

        split_hash(key, &slot_choice, &fprint, &index, &table->metadata, 0);
        assert(index == i);
        block_md[slot] = fprint;
        pc_add(&table->metadata.lv1_balls, 1, 0);
      }
    }
  }

  // level 2
  for (uint64_t i = 0; i < total_blocks; i++) {
    iceberg_lv2_block *block = &table->level2[i];
    uint8_t *block_md = table->metadata.lv2_md[i].block_md;
    for (uint64_t slot = 0; slot < C_LV2 + MAX_LG_LG_N / D_CHOICES; slot++) {
      KeyType key = block->slots[slot].key;
      if (key != 0) {
        uint8_t slot_choice;
        uint8_t fprint;
        uint64_t index;

        split_hash(key, &slot_choice, &fprint, &index, &table->metadata, 1);
        if (index != i) {
          split_hash(key, &slot_choice, &fprint, &index, &table->metadata, 2);
        }

        assert(index == i);
        block_md[slot] = fprint;
        pc_add(&table->metadata.lv2_balls, 1, 0);
      }
    }
  }

  // level 3
  for (uint64_t i = 0; i < total_blocks; i++) {
    ptrdiff_t idx = table->level3[i].head_idx;
    while (idx != -1) {
      idx = table->level3_nodes[idx].next_idx;
      table->metadata.lv3_sizes[i]++;
      pc_add(&table->metadata.lv3_balls, 1, 0);
    }
  }

#ifdef ENABLE_RESIZE
  table->metadata.lv1_resize_ctr = total_blocks;
  table->metadata.lv2_resize_ctr = total_blocks;
  table->metadata.lv3_resize_ctr = total_blocks;

  // create one marker for 8 blocks.
  size_t resize_marker_size = sizeof(uint8_t) * total_blocks / 8;
  table->metadata.lv1_resize_marker = (uint8_t *)mmap(NULL, resize_marker_size, PROT_READ | PROT_WRITE, mmap_flags, 0, 0);
  if (!table->metadata.lv1_resize_marker) {
    perror("level1 resize ctr malloc failed");
    exit(1);
  }
  table->metadata.lv2_resize_marker = (uint8_t *)mmap(NULL, resize_marker_size, PROT_READ | PROT_WRITE, mmap_flags, 0, 0);
  if (!table->metadata.lv2_resize_marker) {
    perror("level2 resize ctr malloc failed");
    exit(1);
  }
  table->metadata.lv3_resize_marker = (uint8_t *)mmap(NULL, resize_marker_size, PROT_READ | PROT_WRITE, mmap_flags, 0, 0);
  if (!table->metadata.lv3_resize_marker) {
    perror("level3 resize ctr malloc failed");
    exit(1);
  }
  memset(table->metadata.lv1_resize_marker, 1, resize_marker_size);
  memset(table->metadata.lv2_resize_marker, 1, resize_marker_size);
  memset(table->metadata.lv3_resize_marker, 1, resize_marker_size);

  rw_lock_init(&table->metadata.rw_lock);
#endif

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
  struct timespec start;
  clock_gettime(CLOCK_MONOTONIC_RAW, &start);
  // grab write lock
  if (!write_lock(&table->metadata.rw_lock, TRY_ONCE_LOCK))
    return false;

  if (unlikely(!need_resize(table))) {
    write_unlock(&table->metadata.rw_lock);
    return false;
  }

  if (is_resize_active(table)) {
    // finish the current resize
    printf("Running iceberg_end\n");
    iceberg_end(table);
    /*write_unlock(&table->metadata.rw_lock);*/
    /*return false;*/
  }

  printf("Setting up resize\nCurrent stats: \n");

  printf("Load factor: %f\n", iceberg_load_factor(table));
  printf("Number level 1 inserts: %ld\n", lv1_balls(table));
  printf("Number level 2 inserts: %ld\n", lv2_balls(table));
  printf("Number level 3 inserts: %ld\n", lv3_balls(table));
  printf("Total inserts: %ld\n", tot_balls(table));

  // reset the block ctr
  table->metadata.lv1_resize_ctr = 0;
  table->metadata.lv2_resize_ctr = 0;
  table->metadata.lv3_resize_ctr = 0;

  // compute new sizes
  uint64_t old_total_blocks = table->metadata.nblocks;
  uint64_t total_blocks = table->metadata.nblocks * 2;
  uint64_t total_size_in_bytes = (sizeof(iceberg_lv1_block) + sizeof(iceberg_lv2_block) + sizeof(iceberg_lv1_block_md) + sizeof(iceberg_lv2_block_md)) * total_blocks;

  //int fd;
  //int rc;
  //size_t mapped_len;
  //int is_pmem;
  //// remap level1
  //size_t old_level1_size = table->metadata.level1_size;
  //size_t level1_size = round_up(sizeof(iceberg_lv1_block) * total_blocks, 2 * MiB);
  //if (old_level1_size != level1_size) {
  //  pmem_unmap(table->level1, table->metadata.level1_size);
  //  table->metadata.level1_size = level1_size;
  //  char level1_filename[FILENAME_LEN];
  //  sprintf(level1_filename, "%s/level1", PMEM_PATH);
  //  fd = open(level1_filename, O_RDWR);
  //  rc = fallocate(fd, 0, old_level1_size, level1_size - old_level1_size);
  //  assert(rc == 0);
  //  rc = close(fd);
  //  assert(rc == 0);
  //  table->level1 = pmem_map_file(level1_filename, 0, 0, 0, &mapped_len, &is_pmem);
  //  assert(mapped_len == level1_size);
  //  assert(is_pmem);
  //  if (table->level1 == NULL) {
  //    perror("level1 remap failed");
  //    exit(1);
  //  }
  //}

  //// remap level2
  //size_t old_level2_size = table->metadata.level2_size;
  //size_t level2_size = round_up(sizeof(iceberg_lv2_block) * total_blocks, 2 * MiB);
  //if (old_level2_size != level2_size) {
  //  fflush(stdout);
  //  pmem_unmap(table->level2, table->metadata.level2_size);
  //  table->metadata.level2_size = level2_size;
  //  char level2_filename[FILENAME_LEN];
  //  sprintf(level2_filename, "%s/level2", PMEM_PATH);
  //  fd = open(level2_filename, O_RDWR);
  //  rc = fallocate(fd, 0, old_level2_size, level2_size - old_level2_size);
  //  assert(rc == 0);
  //  rc = close(fd);
  //  assert(rc == 0);
  //  table->level2 = pmem_map_file(level2_filename, 0, 0, 0, &mapped_len, &is_pmem);
  //  assert(mapped_len == level2_size);
  //  assert(is_pmem);
  //  if (table->level2 == NULL) {
  //    perror("level2 remap failed");
  //    exit(1);
  //  }
  //}

  //// remap level3
  //size_t old_level3_size = table->metadata.level3_size;
  //size_t level3_size = round_up(sizeof(iceberg_lv3_list) * total_blocks, 2 * MiB);
  //if (old_level3_size != level3_size) {
  //  pmem_unmap(table->level3, table->metadata.level3_size);
  //  table->metadata.level3_size = level3_size;
  //  char level3_filename[FILENAME_LEN];
  //  sprintf(level3_filename, "%s/level3", PMEM_PATH);
  //  fd = open(level3_filename, O_RDWR);
  //  rc = fallocate(fd, 0, old_level3_size, level3_size - old_level3_size);
  //  assert(rc == 0);
  //  rc = close(fd);
  //  assert(rc == 0);
  //  table->level3 = pmem_map_file(level3_filename, 0, 0, 0, &mapped_len, &is_pmem);
  //  assert(mapped_len == level3_size);
  //  assert(is_pmem);
  //  if (table->level3 == NULL) {
  //    perror("level3 remap failed");
  //    exit(1);
  //  }
  //  for (uint64_t i = old_total_blocks; i < total_blocks; i++) {
  //    table->level3[i].head_idx = -1;
  //  }
  //}

  // update metadata
  table->metadata.total_size_in_bytes = total_size_in_bytes;
  table->metadata.nslots *= 2;
  table->metadata.nblocks = total_blocks;
  table->metadata.block_bits += 1;

  // remap level1 metadata
  size_t lv1_md_size = sizeof(iceberg_lv1_block_md) * total_blocks + 64;
  size_t old_lv1_md_size = sizeof(iceberg_lv1_block_md) * total_blocks/2 + 64;
  void * lm1temp = mremap(table->metadata.lv1_md, old_lv1_md_size, lv1_md_size, MREMAP_MAYMOVE);
  if (lm1temp == (void *)-1) {
    perror("lv1_md remap failed");
    exit(1);
  }
  table->metadata.lv1_md = lm1temp;

  // remap level2 metadata
  size_t lv2_md_size = sizeof(iceberg_lv2_block_md) * total_blocks + 32;
  size_t old_lv2_md_size = sizeof(iceberg_lv2_block_md) * total_blocks/2 + 32;
  void * lm2temp = mremap(table->metadata.lv2_md, old_lv2_md_size, lv2_md_size, MREMAP_MAYMOVE);
  if (lm2temp == (void *)-1) {
    perror("lv2_md remap failed");
    exit(1);
  }
  table->metadata.lv2_md = lm2temp;

  // re alloc level3 metadata (sizes, locks)
  size_t lv3_sizes_size = sizeof(uint64_t) * total_blocks;
  void * lv3stemp = mremap(table->metadata.lv3_sizes, lv3_sizes_size/2, lv3_sizes_size, MREMAP_MAYMOVE);
  if (lv3stemp == (void *)-1) {
    perror("lv3_sizes remap failed");
    exit(1);
  }
  table->metadata.lv3_sizes = lv3stemp;

  size_t lv3_locks_size = sizeof(uint8_t) * total_blocks;
  void * lv3ltemp = mremap(table->metadata.lv3_locks, lv3_locks_size/2, lv3_locks_size, MREMAP_MAYMOVE);
  if (lv3ltemp == (void *)-1) {
    perror("lv3_locks remap failed");
    exit(1);
  }
  table->metadata.lv3_locks = lv3ltemp;

  // remap resize markers
  // resize_marker_size
  size_t resize_marker_size = sizeof(uint8_t) * total_blocks / 8;
  void * l1mtemp = mremap(table->metadata.lv1_resize_marker, resize_marker_size/2, resize_marker_size, MREMAP_MAYMOVE);
  if (l1mtemp == (void *)-1) {
    perror("level1 remap failed");
    exit(1);
  }
  table->metadata.lv1_resize_marker = l1mtemp;

  void * l2mtemp = mremap(table->metadata.lv2_resize_marker, resize_marker_size/2, resize_marker_size, MREMAP_MAYMOVE);
  if (l2mtemp == (void *)-1) {
    perror("level1 remap failed");
    exit(1);
  }
  table->metadata.lv2_resize_marker = l2mtemp;

  void * l3mtemp = mremap(table->metadata.lv3_resize_marker, resize_marker_size/2, resize_marker_size, MREMAP_MAYMOVE);
  if (l3mtemp == (void *)-1) {
    perror("level1 remap failed");
    exit(1);
  }
  table->metadata.lv3_resize_marker = l3mtemp;

  // resetting the resize markers. First half of the table needs resizing.
  // Assuming a new resize won't be invoked while another resize is active.
  memset(table->metadata.lv1_resize_marker, 0, resize_marker_size);
  memset(table->metadata.lv2_resize_marker, 0, resize_marker_size);
  memset(table->metadata.lv3_resize_marker, 0, resize_marker_size);

  /*printf("Setting up finished\n");*/
  write_unlock(&table->metadata.rw_lock);

  struct timespec end;
  clock_gettime(CLOCK_MONOTONIC_RAW, &end);
  uint64_t elapsed_ns = 1000000000ULL * (end.tv_sec - start.tv_sec) + end.tv_nsec - start.tv_nsec;
  printf("Setup time: %luns\n", elapsed_ns);

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
      // if fixing is needed set the marker
      if (!__sync_lock_test_and_set(&table->metadata.lv1_resize_marker[chunk_idx], 1)) {
        for (uint8_t i = 0; i < 8; ++i) {
          uint64_t idx = chunk_idx * 8 + i;
          iceberg_lv1_move_block(table, idx, 0);
        }
        // set the marker for the dest block
        uint64_t dest_chunk_idx = chunk_idx + table->metadata.nblocks / 8 / 2;
        __sync_lock_test_and_set(&table->metadata.lv1_resize_marker[dest_chunk_idx], 1);
      }
    }
  }
  if (is_lv2_resize_active(table)) {
    for (uint64_t j = 0; j < table->metadata.nblocks / 8; ++j) {
      uint64_t chunk_idx = j;
      // if fixing is needed set the marker
      if (!__sync_lock_test_and_set(&table->metadata.lv2_resize_marker[chunk_idx], 1)) {
        for (uint8_t i = 0; i < 8; ++i) {
          uint64_t idx = chunk_idx * 8 + i;
          iceberg_lv2_move_block(table, idx, 0);
        }
        // set the marker for the dest block
        uint64_t dest_chunk_idx = chunk_idx + table->metadata.nblocks / 8 / 2;
        __sync_lock_test_and_set(&table->metadata.lv2_resize_marker[dest_chunk_idx], 1);
      }
    }
  }
  if (is_lv3_resize_active(table)) {
    for (uint64_t j = 0; j < table->metadata.nblocks / 8; ++j) {
      uint64_t chunk_idx = j;
      // if fixing is needed set the marker
      if (!__sync_lock_test_and_set(&table->metadata.lv3_resize_marker[chunk_idx], 1)) {
        for (uint8_t i = 0; i < 8; ++i) {
          uint64_t idx = chunk_idx * 8 + i;
          iceberg_lv3_move_block(table, idx, 0);
        }
        // set the marker for the dest block
        uint64_t dest_chunk_idx = chunk_idx + table->metadata.nblocks / 8 / 2;
        __sync_lock_test_and_set(&table->metadata.lv3_resize_marker[dest_chunk_idx], 1);
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
    // if fixing is needed set the marker
    if (!__sync_lock_test_and_set(&table->metadata.lv3_resize_marker[chunk_idx], 1)) {
      for (uint8_t i = 0; i < 8; ++i) {
        uint64_t idx = chunk_idx * 8 + i;
        /*printf("LV3 Before: Moving block: %ld load: %f\n", idx, iceberg_block_load(table, idx, 3));*/
        iceberg_lv3_move_block(table, idx, thread_id);
        /*printf("LV3 After: Moving block: %ld load: %f\n", idx, iceberg_block_load(table, idx, 3));*/
      }
      // set the marker for the dest block
      uint64_t dest_chunk_idx = chunk_idx + table->metadata.nblocks / 8 / 2;
      __sync_lock_test_and_set(&table->metadata.lv3_resize_marker[dest_chunk_idx], 1);
    }
  }
#endif

  iceberg_metadata * metadata = &table->metadata;
  iceberg_lv3_list * lists = table->level3;
  iceberg_lv3_node * level3_nodes = table->level3_nodes;

  while(__sync_lock_test_and_set(metadata->lv3_locks + lv3_index, 1));

  iceberg_lv3_node *new_node = NULL;
  ptrdiff_t new_node_idx = -1;
  ptrdiff_t start = lv3_index % NUM_LEVEL3_NODES;
  for (ptrdiff_t i = start; i != NUM_LEVEL3_NODES + start; i++) {
    ptrdiff_t j = i % NUM_LEVEL3_NODES;
    if (__sync_bool_compare_and_swap(&level3_nodes[j].in_use, 0, 1)) {
      new_node = &level3_nodes[j];
      new_node_idx = j;
      break;
    }
  }
  if (new_node == NULL) {
    return false;
  }

  new_node->key = key;
  new_node->val = value;
  new_node->next_idx = lists[lv3_index].head_idx;
  pmem_persist(new_node, sizeof(*new_node));
  lists[lv3_index].head_idx = new_node_idx;
  pmem_persist(&lists[lv3_index], sizeof(lists[lv3_index]));

  metadata->lv3_sizes[lv3_index]++;
  pc_add(&metadata->lv3_balls, 1, thread_id);
  metadata->lv3_locks[lv3_index] = 0;

  return true;
}

static inline bool iceberg_lv2_insert_internal(iceberg_table * table, KeyType key, ValueType value, uint8_t fprint, uint64_t index, uint8_t slot_choice, uint8_t thread_id) {
  iceberg_metadata * metadata = &table->metadata;
  iceberg_lv2_block * blocks = table->level2;
  __mmask32 md_mask = slot_mask_32(metadata->lv2_md[index].block_md, 0) & ((1 << (C_LV2 + MAX_LG_LG_N / D_CHOICES)) - 1);
  uint8_t popct = __builtin_popcountll(md_mask);

  uint8_t start = popct == 0 ? 0 : slot_choice % popct;
  for(uint8_t i = start; i != start + popct; ++i) {

    uint8_t slot = word_select(md_mask, i % popct);

    if(__sync_bool_compare_and_swap(metadata->lv2_md[index].block_md + slot, 0, 1)) {

      pc_add(&metadata->lv2_balls, 1, thread_id);
      blocks[index].slots[slot].key = key;
      blocks[index].slots[slot].val = value;
      pmem_persist(&blocks[index].slots[slot], sizeof(kv_pair));

      metadata->lv2_md[index].block_md[slot] = fprint;
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
  uint8_t slot_choice;
  uint64_t index1, index2;

  split_hash(key, &slot_choice, &fprint1, &index1, metadata, 1);
  split_hash(key, &slot_choice, &fprint2, &index2, metadata, 2);

  __mmask32 md_mask1 = slot_mask_32(metadata->lv2_md[index1].block_md, 0) & ((1 << (C_LV2 + MAX_LG_LG_N / D_CHOICES)) - 1);
  __mmask32 md_mask2 = slot_mask_32(metadata->lv2_md[index2].block_md, 0) & ((1 << (C_LV2 + MAX_LG_LG_N / D_CHOICES)) - 1);

  uint8_t popct1 = __builtin_popcount(md_mask1);
  uint8_t popct2 = __builtin_popcount(md_mask2);

  if(popct2 > popct1) {
    fprint1 = fprint2;
    index1 = index2;
    md_mask1 = md_mask2;
    popct1 = popct2;
  }

#ifdef ENABLE_RESIZE
  // move blocks if resize is active and not already moved.
  if (unlikely(index1 < (table->metadata.nblocks >> 1) && is_lv2_resize_active(table))) {
    uint64_t chunk_idx = index1 / 8;
    // if fixing is needed set the marker
    if (!__sync_lock_test_and_set(&table->metadata.lv2_resize_marker[chunk_idx], 1)) {
      for (uint8_t i = 0; i < 8; ++i) {
        uint64_t idx = chunk_idx * 8 + i;
        /*printf("LV2 Before: Moving block: %ld load: %f\n", idx, iceberg_block_load(table, idx, 2));*/
        iceberg_lv2_move_block(table, idx, thread_id);
        /*printf("LV2 After: Moving block: %ld load: %f\n", idx, iceberg_block_load(table, idx, 2));*/
      }
      // set the marker for the dest block
      uint64_t dest_chunk_idx = chunk_idx + table->metadata.nblocks / 8 / 2;
      __sync_lock_test_and_set(&table->metadata.lv2_resize_marker[dest_chunk_idx], 1);
    }
  }
#endif

  if (iceberg_lv2_insert_internal(table, key, value, fprint1, index1, slot_choice, thread_id))
    return true;

  return iceberg_lv3_insert(table, key, value, lv3_index, thread_id);
}

static bool iceberg_insert_internal(iceberg_table * table, KeyType key, ValueType value, uint8_t fprint, uint64_t index, uint8_t slot_choice, uint8_t thread_id) {
  iceberg_metadata * metadata = &table->metadata;
  iceberg_lv1_block * blocks = table->level1;	

    __mmask64 md_mask = slot_mask_64(metadata->lv1_md[index].block_md, 0);

  uint8_t popct = __builtin_popcountll(md_mask);

  uint8_t start = popct == 0 ? 0 : slot_choice % popct;
  for(uint8_t i = start; i != start + popct; ++i) {

    uint8_t slot = word_select(md_mask, i % popct);

    if(__sync_bool_compare_and_swap(metadata->lv1_md[index].block_md + slot, 0, 1)) {

      pc_add(&metadata->lv1_balls, 1, thread_id);
      blocks[index].slots[slot].key = key;
      blocks[index].slots[slot].val = value;
      pmem_persist(&blocks[index].slots[slot], sizeof(kv_pair));

      metadata->lv1_md[index].block_md[slot] = fprint;
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

  if (unlikely(!read_lock(&table->metadata.rw_lock, WAIT_FOR_LOCK, thread_id)))
    return false;
#endif

  iceberg_metadata * metadata = &table->metadata;
  uint8_t fprint;
  uint8_t slot_choice;
  uint64_t index;

  split_hash(key, &slot_choice, &fprint, &index, metadata, 0);

#ifdef ENABLE_RESIZE
  // move blocks if resize is active and not already moved.
  if (unlikely(index < (table->metadata.nblocks >> 1) && is_lv1_resize_active(table))) {
    uint64_t chunk_idx = index / 8;
    // if fixing is needed set the marker
    if (!__sync_lock_test_and_set(&table->metadata.lv1_resize_marker[chunk_idx], 1)) {
      for (uint8_t i = 0; i < 8; ++i) {
        uint64_t idx = chunk_idx * 8 + i;
        /*printf("LV1 Before: Moving block: %ld load: %f\n", idx, iceberg_block_load(table, idx, 1));*/
        iceberg_lv1_move_block(table, idx, thread_id);
        /*printf("LV1 After: Moving block: %ld load: %f\n", idx, iceberg_block_load(table, idx, 1));*/
      }
      // set the marker for the dest block
      uint64_t dest_chunk_idx = chunk_idx + table->metadata.nblocks / 8 / 2;
      __sync_lock_test_and_set(&table->metadata.lv1_resize_marker[dest_chunk_idx], 1);
    }
  }
#endif

  bool ret = iceberg_insert_internal(table, key, value, fprint, index, slot_choice, thread_id);
  if (!ret)
    ret = iceberg_lv2_insert(table, key, value, index, thread_id);

#ifdef ENABLE_RESIZE
  read_unlock(&table->metadata.rw_lock, thread_id);
#endif

  return ret;
}

static inline bool iceberg_lv3_remove_internal(iceberg_table * table, KeyType key, uint64_t lv3_index, uint8_t thread_id) {

  iceberg_metadata * metadata = &table->metadata;
  iceberg_lv3_list * lists = table->level3;
  iceberg_lv3_node * lv3_nodes = table->level3_nodes;

  while(__sync_lock_test_and_set(metadata->lv3_locks + lv3_index, 1));

  if(metadata->lv3_sizes[lv3_index] == 0) return false;

  assert(lists[lv3_index].head_idx != -1);
  iceberg_lv3_node *head = &lv3_nodes[lists[lv3_index].head_idx];

  if(head->key == key) {
    lists[lv3_index].head_idx = head->next_idx;
    pmem_memset_persist(head, 0, sizeof(*head));

    metadata->lv3_sizes[lv3_index]--;
    pc_add(&metadata->lv3_balls, -1, thread_id);
    metadata->lv3_locks[lv3_index] = 0;

    return true;
  }

  iceberg_lv3_node * current_node = head;

  for (uint64_t i = 0; i < metadata->lv3_sizes[lv3_index] - 1; ++i) {

    assert(current_node->next_idx != -1);
    iceberg_lv3_node *next_node = &lv3_nodes[current_node->next_idx];

    if (next_node->key == key) {

      current_node->next_idx = next_node->next_idx;
      pmem_memset_persist(next_node, 0, sizeof(*next_node));

      metadata->lv3_sizes[lv3_index]--;
      pc_add(&metadata->lv3_balls, -1, thread_id);
      metadata->lv3_locks[lv3_index] = 0;

      return true;
    }

    current_node = next_node;
  }

  metadata->lv3_locks[lv3_index] = 0;
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
    if (__atomic_load_n(&table->metadata.lv3_resize_marker[chunk_idx], __ATOMIC_SEQ_CST) == 0) { // not fixed yet
      return iceberg_lv3_remove_internal(table, key, old_index, thread_id);

    } else {
      // wait for the old block to be fixed
      uint64_t dest_chunk_idx = lv3_index / 8;
      while (__atomic_load_n(&table->metadata.lv3_resize_marker[dest_chunk_idx], __ATOMIC_SEQ_CST) == 0)
        ;
    }
  }
#endif

  return false;
}

static inline bool iceberg_lv2_remove(iceberg_table * table, KeyType key, uint64_t lv3_index, uint8_t thread_id) {

  iceberg_metadata * metadata = &table->metadata;
  iceberg_lv2_block * blocks = table->level2;

  for(int i = 0; i < D_CHOICES; ++i) {

    uint8_t fprint;
    uint8_t slot_choice; // ununsed
    uint64_t index;

    split_hash(key, &slot_choice, &fprint, &index, metadata, 1 + i);

#ifdef ENABLE_RESIZE
    // check if there's an active resize and block isn't fixed yet
    if (unlikely(index >= (table->metadata.nblocks >> 1) && is_lv2_resize_active(table))) {
      uint64_t mask = ~(1ULL << (table->metadata.block_bits - 1));
      uint64_t old_index = index & mask;
      uint64_t chunk_idx = old_index / 8;
      if (__atomic_load_n(&table->metadata.lv2_resize_marker[chunk_idx], __ATOMIC_SEQ_CST) == 0) { // not fixed yet
        __mmask32 md_mask = slot_mask_32(metadata->lv2_md[old_index].block_md, fprint) & ((1 << (C_LV2 + MAX_LG_LG_N / D_CHOICES)) - 1);
        uint8_t popct = __builtin_popcount(md_mask);

        for(uint8_t i = 0; i < popct; ++i) {

          uint8_t slot = word_select(md_mask, i);

          if (blocks[old_index].slots[slot].key == key) {

            metadata->lv2_md[old_index].block_md[slot] = 0;
            blocks[old_index].slots[slot].key = blocks[old_index].slots[slot].val = 0;
            pmem_persist(&blocks[index].slots[slot], sizeof(kv_pair));
            pc_add(&metadata->lv2_balls, -1, thread_id);
            return true;
          }
        }
      } else {
        // wait for the old block to be fixed
        uint64_t dest_chunk_idx = index / 8;
        while (__atomic_load_n(&table->metadata.lv2_resize_marker[dest_chunk_idx], __ATOMIC_SEQ_CST) == 0)
          ;
      }
    }
#endif

    __mmask32 md_mask = slot_mask_32(metadata->lv2_md[index].block_md, fprint) & ((1 << (C_LV2 + MAX_LG_LG_N / D_CHOICES)) - 1);
    uint8_t popct = __builtin_popcount(md_mask);

    for(uint8_t i = 0; i < popct; ++i) {

      uint8_t slot = word_select(md_mask, i);

      if (blocks[index].slots[slot].key == key) {

        metadata->lv2_md[index].block_md[slot] = 0;
        blocks[index].slots[slot].key = blocks[index].slots[slot].val = 0;
        pc_add(&metadata->lv2_balls, -1, thread_id);
        return true;
      }
    }
  }

  return iceberg_lv3_remove(table, key, lv3_index, thread_id);
}

bool iceberg_remove(iceberg_table * table, KeyType key, uint8_t thread_id) {

#ifdef ENABLE_RESIZE
  if (unlikely(!read_lock(&table->metadata.rw_lock, WAIT_FOR_LOCK, thread_id)))
    return false;
#endif

  iceberg_metadata * metadata = &table->metadata;
  iceberg_lv1_block * blocks = table->level1;

  uint8_t fprint;
  uint8_t slot_choice; // unused
  uint64_t index;

  split_hash(key, &slot_choice, &fprint, &index, metadata, 0);

#ifdef ENABLE_RESIZE
  // check if there's an active resize and block isn't fixed yet
  if (unlikely(index >= (table->metadata.nblocks >> 1) && is_lv1_resize_active(table))) {
    uint64_t mask = ~(1ULL << (table->metadata.block_bits - 1));
    uint64_t old_index = index & mask;
    uint64_t chunk_idx = old_index / 8;
    if (__atomic_load_n(&table->metadata.lv1_resize_marker[chunk_idx], __ATOMIC_SEQ_CST) == 0) { // not fixed yet
      __mmask64 md_mask = slot_mask_64(metadata->lv1_md[old_index].block_md, fprint);
      uint8_t popct = __builtin_popcountll(md_mask);

      for(uint8_t i = 0; i < popct; ++i) {

        uint8_t slot = word_select(md_mask, i);

        if (blocks[old_index].slots[slot].key == key) {

          metadata->lv1_md[old_index].block_md[slot] = 0;
          blocks[old_index].slots[slot].key = blocks[old_index].slots[slot].val = 0;
          pmem_persist(&blocks[index].slots[slot], sizeof(kv_pair));
          pc_add(&metadata->lv1_balls, -1, thread_id);
          read_unlock(&table->metadata.rw_lock, thread_id);
          return true;
        }
      }
    } else {
      // wait for the old block to be fixed
      uint64_t dest_chunk_idx = index / 8;
      while (__atomic_load_n(&table->metadata.lv1_resize_marker[dest_chunk_idx], __ATOMIC_SEQ_CST) == 0)
        ;
    }
  }
#endif

  __mmask64 md_mask = slot_mask_64(metadata->lv1_md[index].block_md, fprint);
  uint8_t popct = __builtin_popcountll(md_mask);

  for(uint8_t i = 0; i < popct; ++i) {

    uint8_t slot = word_select(md_mask, i);

    if (blocks[index].slots[slot].key == key) {

      metadata->lv1_md[index].block_md[slot] = 0;
      blocks[index].slots[slot].key = blocks[index].slots[slot].val = 0;
      pc_add(&metadata->lv1_balls, -1, thread_id);
#ifdef ENABLE_RESIZE
      read_unlock(&table->metadata.rw_lock, thread_id);
#endif
      return true;
    }
  }

  bool ret = iceberg_lv2_remove(table, key, index, thread_id);

#ifdef ENABLE_RESIZE
  read_unlock(&table->metadata.rw_lock, thread_id);
#endif

  return ret;
}

static inline bool iceberg_lv3_get_value_internal(iceberg_table * table, KeyType key, ValueType **value, uint64_t lv3_index) {

  iceberg_metadata * metadata = &table->metadata;
  iceberg_lv3_list * lists = table->level3;
  iceberg_lv3_node * lv3_nodes = table->level3_nodes;

  while(__sync_lock_test_and_set(metadata->lv3_locks + lv3_index, 1));

  if(likely(!metadata->lv3_sizes[lv3_index])) {
    metadata->lv3_locks[lv3_index] = 0;
    return false;
  }

  assert(lists[lv3_index].head_idx != -1);
  iceberg_lv3_node * current_node = &lv3_nodes[lists[lv3_index].head_idx];

  for(uint8_t i = 0; i < metadata->lv3_sizes[lv3_index]; ++i) {

    assert(current_node != NULL);
    if(current_node->key == key) {

      *value = &current_node->val;
      metadata->lv3_locks[lv3_index] = 0;
      return true;
    }

    if (current_node->next_idx != -1) {
      current_node = &lv3_nodes[current_node->next_idx];
    } else {
      current_node = NULL;
    }
  }

  metadata->lv3_locks[lv3_index] = 0;

  return false;
}

static inline bool iceberg_lv3_get_value(iceberg_table * table, KeyType key, ValueType **value, uint64_t lv3_index) {
#ifdef ENABLE_RESIZE
  // check if there's an active resize and block isn't fixed yet
  if (unlikely(lv3_index >= (table->metadata.nblocks >> 1) && is_lv3_resize_active(table))) {
    uint64_t mask = ~(1ULL << (table->metadata.block_bits - 1));
    uint64_t old_index = lv3_index & mask;
    uint64_t chunk_idx = old_index / 8;
    if (__atomic_load_n(&table->metadata.lv3_resize_marker[chunk_idx], __ATOMIC_SEQ_CST) == 0) { // not fixed yet
      return iceberg_lv3_get_value_internal(table, key, value, old_index);
    } else {
      // wait for the old block to be fixed
      uint64_t dest_chunk_idx = lv3_index / 8;
      while (__atomic_load_n(&table->metadata.lv3_resize_marker[dest_chunk_idx], __ATOMIC_SEQ_CST) == 0)
        ;
    }
  }
#endif

  return iceberg_lv3_get_value_internal(table, key, value, lv3_index);
}

static inline bool iceberg_lv2_get_value(iceberg_table * table, KeyType key, ValueType **value, uint64_t lv3_index) {

  iceberg_metadata * metadata = &table->metadata;
  iceberg_lv2_block * blocks = table->level2;

  for(uint8_t i = 0; i < D_CHOICES; ++i) {

    uint8_t fprint;
    uint8_t slot_choice; // unused
    uint64_t index;

    split_hash(key, &slot_choice, &fprint, &index, metadata, i + 1);

#ifdef ENABLE_RESIZE
    // check if there's an active resize and block isn't fixed yet
    if (unlikely(index >= (table->metadata.nblocks >> 1) && is_lv2_resize_active(table))) {
      uint64_t mask = ~(1ULL << (table->metadata.block_bits - 1));
      uint64_t old_index = index & mask;
      uint64_t chunk_idx = old_index / 8;
      if (__atomic_load_n(&table->metadata.lv2_resize_marker[chunk_idx], __ATOMIC_SEQ_CST) == 0) { // not fixed yet
        __mmask32 md_mask = slot_mask_32(metadata->lv2_md[old_index].block_md, fprint) & ((1 << (C_LV2 + MAX_LG_LG_N / D_CHOICES)) - 1);

        while (md_mask != 0) {
          int slot = __builtin_ctz(md_mask);
          md_mask = md_mask & ~(1U << slot);

          if (blocks[old_index].slots[slot].key == key) {
            *value = &blocks[old_index].slots[slot].val;
            return true;
          }
        }
      } else {
        // wait for the old block to be fixed
        uint64_t dest_chunk_idx = index / 8;
        while (__atomic_load_n(&table->metadata.lv2_resize_marker[dest_chunk_idx], __ATOMIC_SEQ_CST) == 0)
          ;
      }
    }
#endif

    __mmask32 md_mask = slot_mask_32(metadata->lv2_md[index].block_md, fprint) & ((1 << (C_LV2 + MAX_LG_LG_N / D_CHOICES)) - 1);

    while (md_mask != 0) {
      int slot = __builtin_ctz(md_mask);
      md_mask = md_mask & ~(1U << slot);

      if (blocks[index].slots[slot].key == key) {
        *value = &blocks[index].slots[slot].val;
        return true;
      }
    }

  }

  return iceberg_lv3_get_value(table, key, value, lv3_index);
}

bool iceberg_get_value(iceberg_table * table, KeyType key, ValueType **value, uint8_t thread_id) {

#ifdef ENABLE_RESIZE
  if (unlikely(!read_lock(&table->metadata.rw_lock, WAIT_FOR_LOCK, thread_id)))
    return false;
#endif

  iceberg_metadata * metadata = &table->metadata;
  iceberg_lv1_block * blocks = table->level1;

  uint8_t fprint;
  uint8_t slot_choice; // unused
  uint64_t index;

  split_hash(key, &slot_choice, &fprint, &index, metadata, 0);

#ifdef ENABLE_RESIZE
  // check if there's an active resize and block isn't fixed yet
  if (unlikely(index >= (table->metadata.nblocks >> 1) && is_lv1_resize_active(table))) {
    uint64_t mask = ~(1ULL << (table->metadata.block_bits - 1));
    uint64_t old_index = index & mask;
    uint64_t chunk_idx = old_index / 8;
    if (__atomic_load_n(&table->metadata.lv1_resize_marker[chunk_idx], __ATOMIC_SEQ_CST) == 0) { // not fixed yet
      __mmask64 md_mask = slot_mask_64(metadata->lv1_md[old_index].block_md, fprint);

      while (md_mask != 0) {
        int slot = __builtin_ctzll(md_mask);
        md_mask = md_mask & ~(1ULL << slot);

        if (blocks[old_index].slots[slot].key == key) {
          *value = &blocks[old_index].slots[slot].val;
          read_unlock(&table->metadata.rw_lock, thread_id);
          return true;
        }
      }
    } else {
      // wait for the old block to be fixed
      uint64_t dest_chunk_idx = index / 8;
      while (__atomic_load_n(&table->metadata.lv1_resize_marker[dest_chunk_idx], __ATOMIC_SEQ_CST) == 0)
        ;
    }
  }
#endif

  __mmask64 md_mask = slot_mask_64(metadata->lv1_md[index].block_md, fprint);

  while (md_mask != 0) {
    int slot = __builtin_ctzll(md_mask);
    md_mask = md_mask & ~(1ULL << slot);

    if (blocks[index].slots[slot].key == key) {
      *value = &blocks[index].slots[slot].val;
#ifdef ENABLE_RESIZE
      read_unlock(&table->metadata.rw_lock, thread_id);
#endif
      return true;
    }
  }

  bool ret = iceberg_lv2_get_value(table, key, value, index);

#ifdef ENABLE_RESIZE
  read_unlock(&table->metadata.rw_lock, thread_id);
#endif

  return ret;
}

#ifdef ENABLE_RESIZE
static bool iceberg_nuke_key(iceberg_table * table, uint64_t level, uint64_t index, uint64_t slot, uint64_t thread_id) {

  iceberg_metadata * metadata = &table->metadata;

  if (level == 1) {
    iceberg_lv1_block * blocks = table->level1;
    metadata->lv1_md[index].block_md[slot] = 0;
    blocks[index].slots[slot].key = blocks[index].slots[slot].val = 0;
    pmem_persist(&blocks[index].slots[slot], sizeof(kv_pair));
    pc_add(&metadata->lv1_balls, -1, thread_id);
  } else if (level == 2) {
    iceberg_lv2_block * blocks = table->level2;
    metadata->lv2_md[index].block_md[slot] = 0;
    blocks[index].slots[slot].key = blocks[index].slots[slot].val = 0;
    pmem_persist(&blocks[index].slots[slot], sizeof(kv_pair));
    pc_add(&metadata->lv2_balls, -1, thread_id);
  }

  return true;
}

static bool iceberg_lv1_move_block(iceberg_table * table, uint64_t bnum, uint8_t thread_id) {
  // grab a block
  uint64_t bctr = __atomic_fetch_add(&table->metadata.lv1_resize_ctr, 1, __ATOMIC_SEQ_CST);
  if (bctr >= (table->metadata.nblocks >> 1))
    return true;

  uint64_t dest_j = 0;
  // relocate items in level1
  for (uint64_t j = 0; j < (1 << SLOT_BITS); ++j) {
    KeyType key = table->level1[bnum].slots[j].key;
    if (key == 0)
      continue;
    ValueType value = table->level1[bnum].slots[j].val;
    uint8_t fprint;
    uint8_t slot_choice; // unused
    uint64_t index;

    split_hash(key, &slot_choice, &fprint, &index, &table->metadata, 0);
    // move to new location
    if (index != bnum) {
      uint8_t slot_empty = 0;
      while (!__atomic_compare_exchange_n(&table->metadata.lv1_md[index].block_md[dest_j], &slot_empty, 1, false, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST)) {
        dest_j++;
        assert(dest_j < 64);
        slot_empty = 0;
      }
      table->level1[index].slots[dest_j].key = key;
      table->level1[index].slots[dest_j].val = value;
      table->metadata.lv1_md[index].block_md[dest_j] = fprint;
      dest_j++;
    }
  }
  pmem_persist(&table->level1[bnum].slots[0], dest_j * sizeof(kv_pair));

  for (uint64_t j = 0; j < (1 << SLOT_BITS); ++j) {
    KeyType key = table->level1[bnum].slots[j].key;
    if (key == 0) {
      continue;
    }
    ValueType value = table->level1[bnum].slots[j].val;
    uint8_t fprint;
    uint8_t slot_choice; // unused
    uint64_t index;

    split_hash(key, &slot_choice, &fprint, &index, &table->metadata, 0);
    // kill the dead item
    if (index != bnum) {
      table->level1[bnum].slots[j].key = table->level1[bnum].slots[j].val = 0;
      table->metadata.lv1_md[bnum].block_md[j] = 0;
    }
  }

  return false;
}

static bool iceberg_lv2_move_block(iceberg_table * table, uint64_t bnum, uint8_t thread_id) {
  // grab a block
  uint64_t bctr = __atomic_fetch_add(&table->metadata.lv2_resize_ctr, 1, __ATOMIC_SEQ_CST);
  if (bctr >= (table->metadata.nblocks >> 1))
    return true;

  uint64_t mask = ~(1ULL << (table->metadata.block_bits - 1));
  // relocate items in level2
  for (uint64_t j = 0; j < C_LV2 + MAX_LG_LG_N / D_CHOICES; ++j) {
    KeyType key = table->level2[bnum].slots[j].key;
    if (key == 0)
      continue;
    ValueType value = table->level2[bnum].slots[j].val;
    uint8_t fprint;
    uint8_t slot_choice; // unused
    uint64_t index;

    split_hash(key, &slot_choice, &fprint, &index, &table->metadata, 0);

    for(int i = 0; i < D_CHOICES; ++i) {
      uint8_t l2fprint;
      uint8_t l2slot_choice;
      uint64_t l2index;

      split_hash(key, &l2slot_choice, &l2fprint, &l2index, &table->metadata, i + 1);

      // move to new location
      if ((l2index & mask) == bnum && l2index != bnum) {
        if (!iceberg_lv2_insert_internal(table, key, value, l2fprint, l2index, l2slot_choice, thread_id)) {
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
  iceberg_lv3_list * lists = table->level3;
  iceberg_lv3_node * lv3_nodes = table->level3_nodes;

  // grab a block
  uint64_t bctr = __atomic_fetch_add(&table->metadata.lv3_resize_ctr, 1, __ATOMIC_SEQ_CST);
  if (bctr >= (table->metadata.nblocks >> 1))
    return true;

  // relocate items in level3
  if(unlikely(table->metadata.lv3_sizes[bnum])) {
    iceberg_lv3_node * current_node = &lv3_nodes[lists[bnum].head_idx];

    while (current_node != NULL) {
      KeyType key = current_node->key;
      ValueType value = current_node->val;

      uint8_t fprint;
      uint8_t slot_choice; // unused
      uint64_t index;

      split_hash(key, &slot_choice, &fprint, &index, &table->metadata, 0);
      // move to new location
      if (index != bnum) {
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

      if (current_node->next_idx != -1) {
        current_node = &lv3_nodes[current_node->next_idx];
      } else {
        current_node = NULL;
      }
    }
  }

  return false;
}
#endif
