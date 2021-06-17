#define _GNU_SOURCE
#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <stdlib.h>
#include <pthread.h>
#include <immintrin.h>
#include <tmmintrin.h>
#include <sys/mman.h>

#include "hashutil.h"
#include "iceberg_precompute.h"
#include "iceberg_table.h"

#define likely(x)   __builtin_expect((x),1)
#define unlikely(x) __builtin_expect((x),0)

#define RESIZE_THREADS 1

uint64_t seed[5] = { 12351327692179052ll, 23246347347385899ll, 35236262354132235ll, 13604702930934770ll, 57439820692984798ll };

pthread_cond_t resize_cond = PTHREAD_COND_INITIALIZER;
pthread_mutex_t resize_mutex = PTHREAD_MUTEX_INITIALIZER;

static inline uint64_t nonzero_fprint(uint64_t hash) {
  return hash & ((1 << FPRINT_BITS) - 2) ? hash : hash | 2;
}

static inline uint64_t lv1_hash(KeyType key) {
  return nonzero_fprint(MurmurHash64A(&key, FPRINT_BITS, seed[0]));
}

static inline uint64_t lv1_hash_inline(KeyType key) {
  return nonzero_fprint(MurmurHash64A_inline(&key, FPRINT_BITS, seed[0]));
}

static inline uint64_t lv2_hash(KeyType key, uint8_t i) {
  return nonzero_fprint(MurmurHash64A(&key, FPRINT_BITS, seed[i + 1]));
}

static inline uint64_t lv2_hash_inline(KeyType key, uint8_t i) {
  return nonzero_fprint(MurmurHash64A_inline(&key, FPRINT_BITS, seed[i + 1]));
}

static inline uint8_t word_select(uint64_t val, int rank) {
  val = _pdep_u64(one[rank], val);
  return _tzcnt_u64(val);
}

uint64_t lv1_balls(iceberg_table * table) {
  //pc_sync(table->metadata.lv1_balls);
  return *(table->metadata.lv1_balls.global_counter);
}

uint64_t lv2_balls(iceberg_table * table) {
  //pc_sync(table->metadata.lv2_balls);
  return *(table->metadata.lv2_balls.global_counter);
}

uint64_t lv3_balls(iceberg_table * table) {
  pc_sync(&table->metadata.lv3_balls);
  return *(table->metadata.lv3_balls.global_counter);
}

uint64_t tot_balls(iceberg_table * table) {
  return lv1_balls(table) + lv2_balls(table) + lv3_balls(table);
}

static inline uint64_t total_capacity(iceberg_table * table) {
  return lv3_balls(table) + table->metadata.nblocks * ((1 << SLOT_BITS) + C_LV2 + MAX_LG_LG_N / D_CHOICES);
}

inline double iceberg_load_factor(iceberg_table * table) {
  return (double)tot_balls(table) / (double)total_capacity(table);
}

bool need_resize(iceberg_table * table) {
  uint64_t total_items = tot_balls(table);
  if (unlikely(total_items && total_items % table->metadata.load_check == 0)) {
    double lf = iceberg_load_factor(table);
    if (lf >= 0.9)
      return true;
  }
  return false;
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

static inline size_t round_up(size_t n, size_t k) {
  size_t rem = n % k;
  if (rem == 0) {
    return n;
  }
  n += k - rem;
  return n;
}

static void * iceberg_resize(void * t);

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
  table->level1 = (iceberg_lv1_block *)mmap(NULL, level1_size, PROT_READ | PROT_WRITE, mmap_flags, 0, 0);
  if (!table->level1) {
    perror("level1 malloc failed");
    exit(1);
  }
  size_t level2_size = sizeof(iceberg_lv2_block) * total_blocks;
  //table->level2 = (iceberg_lv2_block *)malloc(level2_size);
  table->level2 = (iceberg_lv2_block *)mmap(NULL, level2_size, PROT_READ | PROT_WRITE, mmap_flags, 0, 0);
  if (!table->level2) {
    perror("level2 malloc failed");
    exit(1);
  }
  size_t level3_size = sizeof(iceberg_lv3_list) * total_blocks;
  table->level3 = (iceberg_lv3_list *)mmap(NULL, level3_size, PROT_READ | PROT_WRITE, mmap_flags, 0, 0);
  if (!table->level3) {
    perror("level3 malloc failed");
    exit(1);
  }

  table->metadata.total_size_in_bytes = total_size_in_bytes;
  table->metadata.nslots = 1 << log_slots;
  table->metadata.nblocks = total_blocks;
  table->metadata.block_bits = log_slots - SLOT_BITS;
  table->metadata.resize_ctr = 0;
  table->metadata.lv1_resize_block_ctr = total_blocks;
  table->metadata.lv2_resize_block_ctr = total_blocks;
  table->metadata.lv3_resize_block_ctr = total_blocks;
  table->metadata.load_check = 0.05 * table->metadata.nslots;

  pc_init(&table->metadata.lv1_balls, &table->metadata.lv1_ctr, 64, 1000);
  pc_init(&table->metadata.lv2_balls, &table->metadata.lv2_ctr, 64, 1000);
  pc_init(&table->metadata.lv3_balls, &table->metadata.lv3_ctr, 64, 1000);

  size_t lv1_md_size = sizeof(iceberg_lv1_block_md) * total_blocks + 64;
  //table->metadata.lv1_md = (iceberg_lv1_block_md *)malloc(sizeof(iceberg_lv1_block_md) * total_blocks);
  table->metadata.lv1_md = (iceberg_lv1_block_md *)mmap(NULL, lv1_md_size, PROT_READ | PROT_WRITE, mmap_flags, 0, 0);
  if (!table->metadata.lv1_md) {
    perror("lv1_md malloc failed");
    exit(1);
  }
  //table->metadata.lv2_md = (iceberg_lv2_block_md *)malloc(sizeof(iceberg_lv2_block_md) * total_blocks);
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

  rw_lock_init(&table->metadata.rw_lock);

  pthread_t resize_thr;
  if (pthread_create(&resize_thr, NULL, iceberg_resize, table) != 0) {
    perror("Error creating resize thread\n");
    exit(1);
  }
  table->metadata.end_flag = false;

  for (uint64_t i = 0; i < total_blocks; ++i) {

    for (uint64_t j = 0; j < (1 << SLOT_BITS); ++j) {
      table->metadata.lv1_md[i].block_md[j] = 0;
      table->level1[i].slots[j].key = table->level1[i].slots[j].val = 0;
    }

    for (uint64_t j = 0; j < C_LV2 + MAX_LG_LG_N / D_CHOICES; ++j) {
      table->metadata.lv2_md[i].block_md[j] = 0;
      table->level2[i].slots[j].key = table->level2[i].slots[j].val = 0;
    }

    table->level3->head = NULL;
    table->metadata.lv3_sizes[i] = table->metadata.lv3_locks[i] = 0;
  }

  return 0;
}

void iceberg_end(iceberg_table * table) {
  table->metadata.end_flag = true;
  pthread_mutex_lock(&resize_mutex);
  pthread_cond_signal(&resize_cond);
  pthread_mutex_unlock(&resize_mutex);
}

static bool iceberg_setup_resize(iceberg_table * table, uint8_t ctr) {
  printf("Setting up resize\n");
  // resize happened b/w releasing read lock and now
  if (ctr != table->metadata.resize_ctr)
    return true;

  // grab write lock
  if (!write_lock(&table->metadata.rw_lock, TRY_ONCE_LOCK))
    return false;

  // incr resize ctr
  table->metadata.resize_ctr += 1;

  // reset the block ctr 
  table->metadata.lv1_resize_block_ctr = 0;
  table->metadata.lv2_resize_block_ctr = 0;
  table->metadata.lv3_resize_block_ctr = 0;

  // compute new sizes
  uint64_t total_blocks = table->metadata.nblocks * 2;
  uint64_t total_size_in_bytes = (sizeof(iceberg_lv1_block) + sizeof(iceberg_lv2_block) + sizeof(iceberg_lv1_block_md) + sizeof(iceberg_lv2_block_md)) * total_blocks;

  // remap level1
  size_t level1_size = sizeof(iceberg_lv1_block) * total_blocks;
  void * l1temp = mremap(table->level1, level1_size/2, level1_size, MREMAP_MAYMOVE);
  if (l1temp == (void *)-1) {
    perror("level1 remap failed");
    exit(1);
  }
  table->level1 = l1temp;

  // remap level2
  size_t level2_size = sizeof(iceberg_lv2_block) * total_blocks;
  void * l2temp = mremap(table->level2, level2_size/2, level2_size, MREMAP_MAYMOVE);
  if (l2temp == (void *)-1) {
    perror("level2 remap failed");
    exit(1);
  }
  table->level2 = l2temp;

  // remap level3
  size_t level3_size = sizeof(iceberg_lv3_list) * total_blocks;
  void * l3temp = mremap(table->level3, level3_size/2, level3_size, MREMAP_MAYMOVE);
  if (l3temp == (void *)-1) {
    perror("level3 remap failed");
    exit(1);
  }
  table->level3 = l3temp;

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

  write_unlock(&table->metadata.rw_lock);
  printf("Setting up finished\n");
  return true;
}

static inline bool iceberg_lv3_insert(iceberg_table * table, KeyType key, ValueType value, uint64_t lv3_index, uint8_t thread_id) {

  iceberg_metadata * metadata = &table->metadata;
  iceberg_lv3_list * lists = table->level3;

  while(__sync_lock_test_and_set(metadata->lv3_locks + lv3_index, 1));

  iceberg_lv3_node * new_node = (iceberg_lv3_node *)malloc(sizeof(iceberg_lv3_node));
  new_node->key = key;
  new_node->val = value;
  new_node->next_node = lists[lv3_index].head;
  lists[lv3_index].head = new_node;

  metadata->lv3_sizes[lv3_index]++;
  pc_add(&metadata->lv3_balls, 1, thread_id);
  metadata->lv3_locks[lv3_index] = 0;

  return true;
}

static inline bool iceberg_lv2_insert(iceberg_table * table, KeyType key, ValueType value, uint64_t lv3_index, uint8_t thread_id) {

  iceberg_metadata * metadata = &table->metadata;
  iceberg_lv2_block * blocks = table->level2;

  if (metadata->lv2_ctr == (int64_t)(C_LV2 * metadata->nblocks))
    return iceberg_lv3_insert(table, key, value, lv3_index, thread_id);

  uint8_t fprint1, fprint2;
  uint64_t index1, index2;

  split_hash(lv2_hash(key, 0), &fprint1, &index1, metadata);
  split_hash(lv2_hash(key, 1), &fprint2, &index2, metadata);

  __mmask32 md_mask1 = slot_mask_32(metadata->lv2_md[index1].block_md, 0) & ((1 << (C_LV2 + MAX_LG_LG_N / D_CHOICES)) - 1);
  __mmask32 md_mask2 = slot_mask_32(metadata->lv2_md[index2].block_md, 0) & ((1 << (C_LV2 + MAX_LG_LG_N / D_CHOICES)) - 1);

  uint8_t popct1 = __builtin_popcountll(md_mask1);
  uint8_t popct2 = __builtin_popcountll(md_mask2);

  if(popct2 > popct1) {
    fprint1 = fprint2;
    index1 = index2;
    md_mask1 = md_mask2;
    popct1 = popct2;
  }

  for(uint8_t i = 0; i < popct1; ++i) {

    uint8_t slot = word_select(md_mask1, i);

    if(__sync_bool_compare_and_swap(metadata->lv2_md[index1].block_md + slot, 0, 1)) {

      pc_add(&metadata->lv2_balls, 1, thread_id);
      blocks[index1].slots[slot].key = key;
      blocks[index1].slots[slot].val = value;

      metadata->lv2_md[index1].block_md[slot] = fprint1;
      return true;
    }
  }

  return iceberg_lv3_insert(table, key, value, lv3_index, thread_id);
}

bool iceberg_insert(iceberg_table * table, KeyType key, ValueType value, uint8_t thread_id) {

  if (unlikely(!read_lock(&table->metadata.rw_lock, WAIT_FOR_LOCK, thread_id)))
    return false;

  uint8_t ctr = table->metadata.resize_ctr;
  if (unlikely(need_resize(table))) {
    read_unlock(&table->metadata.rw_lock, thread_id);
    if (iceberg_setup_resize(table, ctr)) {
      pthread_mutex_lock(&resize_mutex);
      pthread_cond_signal(&resize_cond);
      pthread_mutex_unlock(&resize_mutex);
    }
    /*iceberg_resize(table, thread_id);*/
    /*printf("Resize done\n");*/

    if (unlikely(!read_lock(&table->metadata.rw_lock, WAIT_FOR_LOCK, thread_id)))
      return false;
  }

  iceberg_metadata * metadata = &table->metadata;
  iceberg_lv1_block * blocks = table->level1;	

  uint8_t fprint;
  uint64_t index;

  split_hash(lv1_hash(key), &fprint, &index, metadata);

  __mmask64 md_mask = slot_mask_64(metadata->lv1_md[index].block_md, 0);

  uint8_t popct = __builtin_popcountll(md_mask);

  for(uint8_t i = 0; i < popct; ++i) {

    uint8_t slot = word_select(md_mask, i);

    if(__sync_bool_compare_and_swap(metadata->lv1_md[index].block_md + slot, 0, 1)) {

      pc_add(&metadata->lv1_balls, 1, thread_id);
      blocks[index].slots[slot].key = key;
      blocks[index].slots[slot].val = value;

      metadata->lv1_md[index].block_md[slot] = fprint;
      read_unlock(&table->metadata.rw_lock, thread_id);
      return true;
    }
  }

  bool ret = iceberg_lv2_insert(table, key, value, index, thread_id);

  read_unlock(&table->metadata.rw_lock, thread_id);
  return ret;
}

static inline bool iceberg_lv3_remove(iceberg_table * table, KeyType key, uint64_t lv3_index, uint8_t thread_id) {

  iceberg_metadata * metadata = &table->metadata;
  iceberg_lv3_list * lists = table->level3;

  while(__sync_lock_test_and_set(metadata->lv3_locks + lv3_index, 1));

  if(metadata->lv3_sizes[lv3_index] == 0) return false;

  if(lists[lv3_index].head->key == key) {

    iceberg_lv3_node * old_head = lists[lv3_index].head;
    lists[lv3_index].head = lists[lv3_index].head->next_node;
    free(old_head);

    metadata->lv3_sizes[lv3_index]--;
    pc_add(&metadata->lv3_balls, -1, thread_id);
    metadata->lv3_locks[lv3_index] = 0;

    return true;
  }

  iceberg_lv3_node * current_node = lists[lv3_index].head;

  for(uint64_t i = 0; i < metadata->lv3_sizes[lv3_index] - 1; ++i) {

    if(current_node->next_node->key == key) {

      iceberg_lv3_node * old_node = current_node->next_node;
      current_node->next_node = current_node->next_node->next_node;
      free(old_node);

      metadata->lv3_sizes[lv3_index]--;
      pc_add(&metadata->lv3_balls, -1, thread_id);
      metadata->lv3_locks[lv3_index] = 0;

      return true;
    }

    current_node = current_node->next_node;
  }

  metadata->lv3_locks[lv3_index] = 0;
  return false;
}

static inline bool iceberg_lv2_remove(iceberg_table * table, KeyType key, uint64_t lv3_index, uint8_t thread_id, KeyType mask) {

  iceberg_metadata * metadata = &table->metadata;
  iceberg_lv2_block * blocks = table->level2;

  for(int i = 0; i < D_CHOICES; ++i) {

    uint8_t fprint;
    uint64_t index;

    split_hash(lv2_hash(key, i) & mask, &fprint, &index, metadata);

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

  if (unlikely(!read_lock(&table->metadata.rw_lock, WAIT_FOR_LOCK, thread_id)))
    return false;

  iceberg_metadata * metadata = &table->metadata;
  iceberg_lv1_block * blocks = table->level1;
  uint64_t mask = UINT64_MAX;

  uint8_t fprint;
  uint64_t index;

  split_hash(lv1_hash(key), &fprint, &index, metadata);

  __mmask64 md_mask = slot_mask_64(metadata->lv1_md[index].block_md, fprint);
  uint8_t popct = __builtin_popcountll(md_mask);

  for(uint8_t i = 0; i < popct; ++i) {

    uint8_t slot = word_select(md_mask, i);

    if (blocks[index].slots[slot].key == key) {

      metadata->lv1_md[index].block_md[slot] = 0;
      blocks[index].slots[slot].key = blocks[index].slots[slot].val = 0;
      pc_add(&metadata->lv1_balls, -1, thread_id);
      read_unlock(&table->metadata.rw_lock, thread_id);
      return true;
    }
  }

  bool ret = iceberg_lv2_remove(table, key, index, thread_id, mask);

  read_unlock(&table->metadata.rw_lock, thread_id);
  return ret;
}

bool iceberg_remove_lv1_resize(iceberg_table * table, KeyType key, uint8_t thread_id) {

  if (unlikely(!read_lock(&table->metadata.rw_lock, WAIT_FOR_LOCK, thread_id)))
    return false;

  iceberg_metadata * metadata = &table->metadata;
  iceberg_lv1_block * blocks = table->level1;
  uint64_t mask = ~(1ULL << (metadata->block_bits + FPRINT_BITS - 1));

  uint8_t fprint;
  uint64_t index;

  split_hash(lv1_hash(key) & mask, &fprint, &index, metadata);

  __mmask64 md_mask = slot_mask_64(metadata->lv1_md[index].block_md, fprint);
  uint8_t popct = __builtin_popcountll(md_mask);

  for(uint8_t i = 0; i < popct; ++i) {

    uint8_t slot = word_select(md_mask, i);

    if (blocks[index].slots[slot].key == key) {

      metadata->lv1_md[index].block_md[slot] = 0;
      pc_add(&metadata->lv1_balls, -1, thread_id);
      read_unlock(&table->metadata.rw_lock, thread_id);
      return true;
    }
  }

  read_unlock(&table->metadata.rw_lock, thread_id);
  return false;
}

static inline bool iceberg_lv3_get_value(iceberg_table * table, KeyType key, ValueType **value, uint64_t lv3_index) {

  iceberg_metadata * metadata = &table->metadata;
  iceberg_lv3_list * lists = table->level3;

  while(__sync_lock_test_and_set(metadata->lv3_locks + lv3_index, 1));

  if(likely(!metadata->lv3_sizes[lv3_index])) {
    metadata->lv3_locks[lv3_index] = 0;
    return false;
  }

  iceberg_lv3_node * current_node = lists[lv3_index].head;

  for(uint8_t i = 0; i < metadata->lv3_sizes[lv3_index]; ++i) {

    if(current_node->key == key) {

      *value = &current_node->val;
      metadata->lv3_locks[lv3_index] = 0;
      return true;
    }

    current_node = current_node->next_node;
  }

  metadata->lv3_locks[lv3_index] = 0;
  return false;
}


static inline bool iceberg_lv2_get_value(iceberg_table * table, KeyType key, ValueType **value, uint64_t lv3_index) {

  iceberg_metadata * metadata = &table->metadata;
  iceberg_lv2_block * blocks = table->level2;

  for(uint8_t i = 0; i < D_CHOICES; ++i) {

    uint8_t fprint;
    uint64_t index;

    split_hash(lv2_hash_inline(key, i), &fprint, &index, metadata);

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

  if (unlikely(!read_lock(&table->metadata.rw_lock, WAIT_FOR_LOCK, thread_id)))
    return false;

  iceberg_metadata * metadata = &table->metadata;
  iceberg_lv1_block * blocks = table->level1;

  uint8_t fprint;
  uint64_t index;

  split_hash(lv1_hash_inline(key), &fprint, &index, metadata);

  __mmask64 md_mask = slot_mask_64(metadata->lv1_md[index].block_md, fprint);

  while (md_mask != 0) {
    int slot = __builtin_ctzll(md_mask);
    md_mask = md_mask & ~(1ULL << slot);

    if (blocks[index].slots[slot].key == key) {
      *value = &blocks[index].slots[slot].val;
      read_unlock(&table->metadata.rw_lock, thread_id);
      return true;
    }
  }

  bool ret = iceberg_lv2_get_value(table, key, value, index);

  read_unlock(&table->metadata.rw_lock, thread_id);
  return ret;
}

static bool iceberg_lv1_move_block(iceberg_table * table, uint8_t thread_id) {
  // grab a block 
  uint64_t bnum = __atomic_fetch_add(&table->metadata.lv1_resize_block_ctr, 1, __ATOMIC_SEQ_CST);
  if (bnum >= (table->metadata.nblocks >> 1))
    return true;

  // relocate items in level1
  for (uint64_t j = 0; j < (1 << SLOT_BITS); ++j) {
    KeyType key = table->level1[bnum].slots[j].key;
    if (key == 0)
      continue;
    ValueType value = table->level1[bnum].slots[j].val;
    uint8_t fprint;
    uint64_t index;

    split_hash(lv1_hash(key), &fprint, &index, &table->metadata);
    // move to new location
    if (index != bnum) {
      if (!iceberg_remove_lv1_resize(table, key, thread_id)) {
        printf("Failed remove during resize lv1. key: %" PRIu64 ", block: %ld\n", key, bnum);
        exit(0);
      }
      if (!iceberg_insert(table, key, value, thread_id)) {
        printf("Failed insert during resize lv1\n");
        exit(0);
      }
      ValueType *val;
      if (!iceberg_get_value(table, key, &val, thread_id)) {
        printf("Key not found during resize lv1: %ld\n", key);
        exit(0);
      }
    }
  }

  return false;
}

static bool iceberg_lv2_move_block(iceberg_table * table, uint8_t thread_id) {
  // grab a block 
  uint64_t bnum = __atomic_fetch_add(&table->metadata.lv2_resize_block_ctr, 1, __ATOMIC_SEQ_CST);
  if (bnum >= (table->metadata.nblocks >> 1))
    return true;

  // relocate items in level2
  uint64_t mask = ~(1ULL << (table->metadata.block_bits + FPRINT_BITS - 1));
  for (uint64_t j = 0; j < C_LV2 + MAX_LG_LG_N / D_CHOICES; ++j) {
    KeyType key = table->level2[bnum].slots[j].key;
    if (key == 0)
      continue;
    ValueType value = table->level2[bnum].slots[j].val;

    ValueType *val;
    // move to new location
    if (!iceberg_get_value(table, key, &val, thread_id)) {
      if (!iceberg_lv2_remove(table, key, bnum, thread_id, mask)) {
        printf("Failed remove during resize lv2\n");
        exit(0);
      }
      if (!iceberg_insert(table, key, value, thread_id)) {
        printf("Failed insert during resize lv2\n");
        exit(0);
      }
      if (!iceberg_get_value(table, key, &val, thread_id)) {
        printf("Key not found during resize lv2: %ld\n", key);
        exit(0);
      }
    }
  }

  return false;
}

static bool iceberg_lv3_move_block(iceberg_table * table, uint8_t thread_id) {
  // grab a block 
  uint64_t bnum = __atomic_fetch_add(&table->metadata.lv3_resize_block_ctr, 1, __ATOMIC_SEQ_CST);
  if (bnum >= (table->metadata.nblocks >> 1))
    return true;

  // relocate items in level3
  if(unlikely(table->metadata.lv3_sizes[bnum])) {
    iceberg_lv3_node * current_node = table->level3[bnum].head;

    while (current_node != NULL) {
      KeyType key = current_node->key;
      ValueType value = current_node->val;

      uint8_t fprint;
      uint64_t index;

      split_hash(lv1_hash(key), &fprint, &index, &table->metadata);
      // move to new location
      ValueType *val;
      if (index != bnum) {
        current_node = current_node->next_node;
        if (!iceberg_lv3_remove(table, key, bnum, thread_id)) {
          printf("Failed remove during resize lv3: %" PRIu64 "\n", key);
          exit(0);
        }
        if (!iceberg_insert(table, key, value, thread_id)) {
          printf("Failed insert during resize lv3\n");
          exit(0);
        }
        if (!iceberg_get_value(table, key, &val, thread_id)) {
          printf("Key not found during resize lv3: %ld\n", key);
          exit(0);
        }
      }
      else
        current_node = current_node->next_node;
    }
  }

  return false;
}

typedef struct resize_str {
  iceberg_table * table;
  uint8_t level;
  uint8_t thread_id;
} resize_str;

void * iceberg_move(void * arg) {
  resize_str * str = (resize_str *)arg;

  switch (str->level) {
    case 1:
      while (!iceberg_lv1_move_block(str->table, str->thread_id))
        ;
      break;
    case 2:
      while (!iceberg_lv2_move_block(str->table, str->thread_id))
        ;
      break;
    case 3:
      while (!iceberg_lv3_move_block(str->table, str->thread_id))
        ;
      break;
  }

  pthread_exit(NULL);
}

static void * iceberg_resize(void * t) {
  iceberg_table * table = (iceberg_table *)t;
  pthread_t thr_list[RESIZE_THREADS];

  while (!table->metadata.end_flag) {
    pthread_mutex_lock(&resize_mutex);
    pthread_cond_wait(&resize_cond, &resize_mutex);
    if (table->metadata.end_flag) {
      pthread_mutex_unlock(&resize_mutex);
      break;
    }
    // move levels
    for (uint64_t i = 1; i <= 3; ++i) {
      for (uint64_t j = 0; j < RESIZE_THREADS; ++j) {
        resize_str str = {table, i, j};
        if (pthread_create(&thr_list[j], NULL, iceberg_move, &str) != 0) {
          perror("Error creating resize thread\n");
          exit(1);
        }
      }
      for (uint64_t i = 0; i < RESIZE_THREADS; ++i) {
        pthread_join(thr_list[i], NULL);
      }
    }
    pthread_mutex_unlock(&resize_mutex);
  }

  pthread_exit(NULL);
}

