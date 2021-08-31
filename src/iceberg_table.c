#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <stdlib.h>
#include <immintrin.h>
#include <tmmintrin.h>
#include <sys/mman.h>

#include "hashutil.h"
#include "iceberg_precompute.h"
#include "iceberg_table.h"

#define SLOT_EMPTY ((uint8_t)0)
#define SLOT_RESERVED ((uint8_t)1)

#define ICEBERG_LF_TO_SPLIT 0.91l

#define likely(x)   __builtin_expect((x),1)
#define unlikely(x) __builtin_expect((x),0)

uint64_t seed[5] = { 12351327692179052ll, 23246347347385899ll, 35236262354132235ll, 13604702930934770ll, 57439820692984798ll };

uint64_t nonzero_fprint(uint64_t hash) {
	return hash & ((1 << FPRINT_BITS) - 2) ? hash : hash | 2;
}

uint64_t lv1_hash(KeyType key) {
	return nonzero_fprint(MurmurHash64A(&key, FPRINT_BITS, seed[0]));
}

uint64_t lv1_hash_inline(KeyType key) {
	return nonzero_fprint(MurmurHash64A_inline(&key, FPRINT_BITS, seed[0]));
}

uint64_t lv2_hash(KeyType key, uint8_t i) {
	return nonzero_fprint(MurmurHash64A(&key, FPRINT_BITS, seed[i + 1]));
}

uint64_t lv2_hash_inline(KeyType key, uint8_t i) {
	return nonzero_fprint(MurmurHash64A_inline(&key, FPRINT_BITS, seed[i + 1]));
}

static inline uint8_t word_select(uint64_t val, int rank) {
	val = _pdep_u64(one[rank], val);
	return _tzcnt_u64(val);
}

uint64_t lv1_balls(iceberg_table * table) {
	//pc_sync(table->metadata->lv1_balls);
	return *(table->metadata->lv1_balls->global_counter);
}

uint64_t lv2_balls(iceberg_table * table) {
	//pc_sync(table->metadata->lv2_balls);
	return *(table->metadata->lv2_balls->global_counter);
}

uint64_t lv3_balls(iceberg_table * table) {
	//pc_sync(table->metadata->lv3_balls);
	return *(table->metadata->lv3_balls->global_counter);
}

uint64_t tot_balls(iceberg_table * table) {
	return lv1_balls(table) + lv2_balls(table) + lv3_balls(table);
}

static inline uint64_t num_blocks(iceberg_metadata *md) {
	return 1ULL << md->log_nblocks;
}

static inline uint64_t lv2_num_blocks(iceberg_metadata *md) {
	return 1ULL << md->lv2_log_nblocks;
}

__attribute__ ((unused))
static inline uint64_t lv3_num_blocks(iceberg_metadata *md) {
	return 1ULL << md->lv3_log_nblocks;
}

static inline uint64_t lv1_block_capacity() {
	return 1 << SLOT_BITS;
}

static inline uint64_t lv2_block_capacity() {
	return C_LV2 + MAX_LG_LG_N / D_CHOICES;
}

uint64_t total_capacity(iceberg_table * table) {
	uint64_t lv1_capacity = num_blocks(table->metadata) * lv1_block_capacity();
	uint64_t lv2_capacity = lv2_num_blocks(table->metadata) * lv2_block_capacity();
	uint64_t lv3_capacity = lv3_balls(table);
	return lv1_capacity + lv2_capacity + lv3_capacity;
}

double iceberg_load_factor(iceberg_table * table) {
	return (double)tot_balls(table) / (double)total_capacity(table);
}

void split_hash(uint64_t hash, uint8_t *fprint, uint64_t *index, uint8_t log_nblocks) {	
	*fprint = hash & ((1 << FPRINT_BITS) - 1);
	*index = (hash >> FPRINT_BITS) & ((1 << log_nblocks) - 1);
}

void lv2_split_hash(uint64_t hash, uint8_t *fprint, uint64_t *index, iceberg_metadata * metadata) {	
	*fprint = hash & ((1 << FPRINT_BITS) - 1);
	*index = (hash >> FPRINT_BITS) & ((1 << metadata->lv2_log_nblocks) - 1);
}

uint32_t slot_mask_32(uint8_t * metadata, uint8_t fprint) {
	__m256i bcast = _mm256_set1_epi8(fprint);
	__m256i block = _mm256_loadu_si256((const __m256i *)(metadata));
	return _mm256_cmp_epi8_mask(bcast, block, _MM_CMPINT_EQ);
}

uint64_t slot_mask_64(uint8_t * metadata, uint8_t fprint) {
	__m512i bcast = _mm512_set1_epi8(fprint);
	__m512i block = _mm512_loadu_si512((const __m512i *)(metadata));
	return _mm512_cmp_epi8_mask(bcast, block, _MM_CMPINT_EQ);
}

size_t round_up(size_t n, size_t k) {
	size_t rem = n % k;
	if (rem == 0) {
		return n;
	}
	n += k - rem;
	return n;
}

iceberg_table * iceberg_init(uint64_t log_slots, uint64_t final_log_slots) {

	iceberg_table * table;

	uint64_t log_total_blocks = log_slots - SLOT_BITS;
	uint64_t total_blocks = 1 << log_total_blocks;
	uint64_t final_total_blocks = 1 << (final_log_slots - SLOT_BITS);
	uint64_t lv2_log_nblocks = final_log_slots - SLOT_BITS;
	uint64_t lv2_total_blocks = 1 << lv2_log_nblocks;
	uint64_t lv3_log_nblocks = final_log_slots - SLOT_BITS;
	uint64_t lv3_total_blocks = 1 << lv3_log_nblocks;

  	uint64_t total_size_in_bytes = (sizeof(iceberg_lv1_block) + sizeof(iceberg_lv2_block) + sizeof(iceberg_lv1_block_md) + sizeof(iceberg_lv2_block_md)) * total_blocks;

	table = (iceberg_table *)malloc(sizeof(iceberg_table));
	assert(table);

#if defined(HUGE_TLB)
	int mmap_flags = MAP_SHARED | MAP_ANONYMOUS | MAP_POPULATE | MAP_HUGETLB;
#else
	int mmap_flags = MAP_SHARED | MAP_ANONYMOUS | MAP_POPULATE;
#endif

        size_t level1_size = sizeof(iceberg_lv1_block) * final_total_blocks;
        //table->level1 = (iceberg_lv1_block *)malloc(level1_size);
	table->level1 = (iceberg_lv1_block *)mmap(NULL, level1_size, PROT_READ | PROT_WRITE, mmap_flags, 0, 0);
	if (!table->level1) {
		perror("level1 malloc failed");
		exit(1);
	}
        size_t level2_size = sizeof(iceberg_lv2_block) * lv2_total_blocks;
	printf("level2 size: %lu\n", level2_size);
        //table->level2 = (iceberg_lv2_block *)malloc(level2_size);
	table->level2 = (iceberg_lv2_block *)mmap(NULL, level2_size, PROT_READ | PROT_WRITE, mmap_flags, 0, 0);
	if (!table->level2) {
		perror("level2 malloc failed");
		exit(1);
	}
	table->level3 = (iceberg_lv3_list *)malloc(sizeof(iceberg_lv3_list) * lv3_total_blocks);

	table->metadata = (iceberg_metadata *)malloc(sizeof(iceberg_metadata));
	table->metadata->total_size_in_bytes = total_size_in_bytes;
	table->metadata->nslots = 1 << log_slots;
	table->metadata->log_nblocks = log_total_blocks;
	table->metadata->initial_log_nblocks = log_total_blocks;
	table->metadata->lv2_log_nblocks = lv2_log_nblocks;
	table->metadata->lv3_log_nblocks = lv3_log_nblocks;

	table->metadata->lv1_balls = (pc_t *)malloc(sizeof(pc_t));
	int64_t * lv1_ctr = (int64_t *)malloc(sizeof(int64_t));
	* lv1_ctr = 0;
	pc_init(table->metadata->lv1_balls, lv1_ctr, 64, 1000);
	
	table->metadata->lv2_balls = (pc_t *)malloc(sizeof(pc_t));
	int64_t * lv2_ctr = (int64_t *)malloc(sizeof(int64_t));
	* lv2_ctr = 0;
	pc_init(table->metadata->lv2_balls, lv2_ctr, 64, 1000);

	table->metadata->lv3_balls = (pc_t *)malloc(sizeof(pc_t));
	int64_t * lv3_ctr = (int64_t *)malloc(sizeof(int64_t));
	* lv3_ctr = 0;
	pc_init(table->metadata->lv3_balls, lv3_ctr, 64, 1000);

	size_t lv1_md_size = sizeof(iceberg_lv1_block_md) * final_total_blocks + 64;
	table->metadata->lv1_md = (iceberg_lv1_block_md *)mmap(NULL, lv1_md_size, PROT_READ | PROT_WRITE, mmap_flags, 0, 0);
	size_t lv2_md_size = sizeof(iceberg_lv2_block_md) * lv2_total_blocks + 32;
	table->metadata->lv2_md = (iceberg_lv2_block_md *)mmap(NULL, lv2_md_size, PROT_READ | PROT_WRITE, mmap_flags, 0, 0);
	table->metadata->lv3_sizes = (uint64_t *)malloc(sizeof(uint64_t) * lv3_total_blocks);
	table->metadata->lv3_locks = (uint8_t *)malloc(sizeof(uint8_t) * lv3_total_blocks);

	for (uint64_t i = 0; i < final_total_blocks; ++i) {

		for (uint64_t j = 0; j < (1 << SLOT_BITS); ++j) {
			table->metadata->lv1_md[i].block_md[j] = 0;
			table->level1[i].slots[j].key = table->level1[i].slots[j].val = 0;
		}
	}

	for (uint64_t i = 0; i < lv2_total_blocks; ++i) {
		for (uint64_t j = 0; j < C_LV2 + MAX_LG_LG_N / D_CHOICES; ++j) {
			table->metadata->lv2_md[i].block_md[j] = 0;
			table->level2[i].slots[j].key = table->level2[i].slots[j].val = 0;
		}
	}

	for (uint64_t i = 0; i < lv3_total_blocks; ++i) {
		table->metadata->lv3_sizes[i] = table->metadata->lv3_locks[i] = 0;
	}

	return table;
}

void iceberg_lv1_print(iceberg_table *table, uint64_t index) {
	return;
	iceberg_lv1_block_md md = table->metadata->lv1_md[index];
	iceberg_lv1_block block = table->level1[index];
		printf("-----------------------------------------------------------------------------------------------------------------------------------------------------------------\n");
		printf("| index: %-12lu                                                                                                                                           |\n", index);
	for (uint64_t i = 0; i < 4; i++) {
		printf("|---------------------------------------------------------------------------------------------------------------------------------------------------------------|\n");
		for (uint64_t j = 0; j < 16; j++) {
			uint64_t pos = i * 16 + j;
			printf("| %2lu 0x%02x ", pos, md.block_md[pos]);
		}
		printf("|\n");
		for (uint64_t j = 0; j < 16; j++) {
			uint64_t pos = i * 16 + j;
			uint16_t key = (uint16_t)block.slots[pos].key;
			printf("|  0x%04x ", key);
		}
		printf("|\n");
		for (uint64_t j = 0; j < 16; j++) {
			uint64_t pos = i * 16 + j;
			uint16_t val = (uint16_t)block.slots[pos].val;
			printf("|  0x%04x ", val);
		}
		printf("|\n");
	}
	printf("-----------------------------------------------------------------------------------------------------------------------------------------------------------------\n");
	printf("\n");
	fflush(stdout);
}

bool iceberg_lv3_insert(iceberg_table * table, KeyType key, ValueType value, uint64_t lv3_index, uint8_t thread_id) {

	iceberg_metadata * metadata = table->metadata;
	iceberg_lv3_list * lists = table->level3;

	while(__sync_lock_test_and_set(metadata->lv3_locks + lv3_index, 1));

	iceberg_lv3_node * new_node = (iceberg_lv3_node *)malloc(sizeof(iceberg_lv3_node));
	new_node->key = key;
	new_node->val = value;
	new_node->next_node = lists[lv3_index].head;
	lists[lv3_index].head = new_node;

	metadata->lv3_sizes[lv3_index]++;
	pc_add(metadata->lv3_balls, 1, thread_id);
	metadata->lv3_locks[lv3_index] = 0;

	return true;
}

bool iceberg_lv2_insert(iceberg_table * table, KeyType key, ValueType value, uint64_t lv3_index, uint8_t thread_id) {

	iceberg_metadata * metadata = table->metadata;
	iceberg_lv2_block * blocks = table->level2;

	uint8_t fprint1, fprint2;
	uint64_t index1, index2;

	lv2_split_hash(lv2_hash(key, 0), &fprint1, &index1, metadata);
	lv2_split_hash(lv2_hash(key, 1), &fprint2, &index2, metadata);

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

			pc_add(metadata->lv2_balls, 1, thread_id);
			blocks[index1].slots[slot].key = key;
			blocks[index1].slots[slot].val = value;

			metadata->lv2_md[index1].block_md[slot] = fprint1;
			return true;
		}
	}

	return iceberg_lv3_insert(table, key, value, lv3_index, thread_id);
}

static inline kv_pair *iceberg_lv1_get_entry(iceberg_table *table, uint64_t index, uint64_t slot) {
	return &table->level1[index].slots[slot];
}

static inline void iceberg_lv1_set_entry(iceberg_table *table, uint64_t index, uint64_t slot, kv_pair kv, uint8_t fprint) {
	table->level1[index].slots[slot] = kv;
	//assert(table->metadata->lv1_md[index].block_md[slot] == SLOT_RESERVED);
	table->metadata->lv1_md[index].block_md[slot] = fprint;
}

static inline void iceberg_lv1_remove_entry(iceberg_table *table, uint64_t index, uint64_t slot) {
	table->metadata->lv1_md[index].block_md[slot] = SLOT_RESERVED;
	table->level1[index].slots[slot].key = 0;
	table->level1[index].slots[slot].val = 0;
	table->metadata->lv1_md[index].block_md[slot] = SLOT_EMPTY;
}

static inline bool iceberg_lv1_try_reserve_slot(iceberg_metadata *md, uint64_t index, uint64_t slot) {
	uint8_t slot_empty = SLOT_EMPTY;
	bool ret =  __atomic_compare_exchange_n(&md->lv1_md[index].block_md[slot],
				&slot_empty, SLOT_RESERVED, false,
				__ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST);
	return ret;
}

static inline uint8_t iceberg_lv1_entry_fprint(iceberg_metadata *md, uint64_t index, uint64_t slot) {
	return md->lv1_md[index].block_md[slot];
}
static inline void iceberg_lv1_entry_set_fprint(iceberg_metadata *md, uint64_t index, uint64_t slot, uint8_t fprint) {
	md->lv1_md[index].block_md[slot] = fprint;
}

static inline uint8_t iceberg_generation(iceberg_metadata *md, uint8_t log_nblocks) {
	return log_nblocks - md->initial_log_nblocks;
}

static inline uint8_t iceberg_index_generation(iceberg_metadata *md, uint64_t index) {
	return md->lv1_md[index].block_md[0];
}

static inline bool iceberg_first_half(uint64_t index, uint8_t log_nblocks) {
	return index < (1ULL << (log_nblocks - 1));
}

static inline bool iceberg_second_half(uint64_t index, uint8_t log_nblocks) {
	return !iceberg_first_half(index, log_nblocks);
}

static inline uint64_t iceberg_split_from(uint64_t index, uint8_t log_nblocks) {
	return index & ~(1ULL << (log_nblocks - 1));
}

static inline uint64_t iceberg_split_to(uint64_t index, uint8_t log_nblocks) {
	return index | (1ULL << (log_nblocks - 1));
}

static inline bool iceberg_needs_split(iceberg_metadata *md, uint8_t log_nblocks, uint64_t index) {
	//assert(iceberg_index_generation(md, index) <= iceberg_generation(md, log_nblocks));
	bool needs_split = iceberg_index_generation(md, index) != iceberg_generation(md, log_nblocks);

	return needs_split;
}

static inline bool iceberg_try_start_split(iceberg_metadata *md, uint8_t log_nblocks, uint64_t index) {
	uint64_t split_from = iceberg_split_from(index, log_nblocks);
	uint8_t from_gen = iceberg_index_generation(md, split_from);
	uint8_t to_gen = iceberg_generation(md, log_nblocks);
	//assert(from_gen != to_gen);
	return __atomic_compare_exchange_n(&md->lv1_md[split_from].block_md[0],
			&from_gen, to_gen, false, __ATOMIC_SEQ_CST,
			__ATOMIC_SEQ_CST);
}

static inline void iceberg_maybe_split(iceberg_table *table, uint64_t index) {
	iceberg_metadata *md = table->metadata;
	uint8_t log_nblocks = md->log_nblocks;
	if (likely(!iceberg_needs_split(md, log_nblocks, index))) {
		return;
	}
	if (!iceberg_try_start_split(md, log_nblocks, index)) {
		return;
	}

	uint64_t split_from = iceberg_split_from(index, log_nblocks);
	uint64_t split_to = iceberg_split_to(index, log_nblocks);
	iceberg_lv1_print(table, split_from);
	uint64_t to_idx = 1;
	for (uint64_t from_idx = 1; from_idx < lv1_block_capacity(); from_idx++) {
		//assert(to_idx < lv1_block_capacity());
		// FIXME: [aconway 2021-08-30]
		// Revisit these for consistency semantics
		while (iceberg_lv1_entry_fprint(md, split_from, from_idx) == SLOT_RESERVED);
		if (iceberg_lv1_entry_fprint(md, split_from, from_idx) == SLOT_EMPTY) {
			continue;
		}
		kv_pair kv = *iceberg_lv1_get_entry(table, split_from, from_idx);
		uint8_t fprint;
		uint64_t new_index;
		split_hash(lv1_hash_inline(kv.key), &fprint, &new_index, log_nblocks);
		if (new_index != split_from) {
			//assert(new_index == split_to);
			while (to_idx < lv1_block_capacity()) {
				if (iceberg_lv1_try_reserve_slot(md, split_to, to_idx)) {
					//assert(to_idx < lv1_block_capacity());
					iceberg_lv1_set_entry(table, split_to, to_idx, kv, fprint);
					break;
				}
				to_idx++;
			}
			iceberg_lv1_remove_entry(table, split_from, from_idx);
		}
	}
	iceberg_lv1_entry_set_fprint(md, split_to, 0, iceberg_generation(md, log_nblocks));
	iceberg_lv1_print(table, split_from);
	iceberg_lv1_print(table, split_to);
}

static inline void iceberg_maybe_resize(iceberg_table *table) {
	uint8_t log_nblocks = table->metadata->log_nblocks;
	if (iceberg_load_factor(table) >= ICEBERG_LF_TO_SPLIT) {
		printf("Needs resize\n");
	}
	if (unlikely(iceberg_load_factor(table) >= ICEBERG_LF_TO_SPLIT)) {
		__atomic_compare_exchange_n(&table->metadata->log_nblocks,
				&log_nblocks, log_nblocks + 1, false,
				__ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST);
	}
}

bool iceberg_insert(iceberg_table * table, KeyType key, ValueType value, uint8_t thread_id) {

	iceberg_metadata * metadata = table->metadata;
	iceberg_lv1_block * blocks = table->level1;	

	iceberg_maybe_resize(table);

	uint8_t fprint;
	uint64_t index;
	uint8_t log_nblocks = metadata->log_nblocks;

	split_hash(lv1_hash(key), &fprint, &index, log_nblocks);

	iceberg_maybe_split(table, index);

	uint8_t gen = iceberg_index_generation(metadata, index);
	if (unlikely(gen != iceberg_generation(metadata, metadata->log_nblocks))) {
		return iceberg_insert(table, key, value, thread_id);
	}

	__mmask64 md_mask = slot_mask_64(metadata->lv1_md[index].block_md, 0);
	md_mask &= ~1;

	uint8_t popct = __builtin_popcountll(md_mask);

	for(uint8_t i = 0; i < popct; ++i) {
	
		uint8_t slot = word_select(md_mask, i);
		//assert(slot != 0);
		//assert(slot < 64);

		if(__sync_bool_compare_and_swap(metadata->lv1_md[index].block_md + slot, 0, 1)) {
			
			pc_add(metadata->lv1_balls, 1, thread_id);
			blocks[index].slots[slot].key = key;
			blocks[index].slots[slot].val = value;

			if (unlikely(gen != iceberg_generation(metadata, metadata->log_nblocks))) {
				pc_add(metadata->lv1_balls, -1, thread_id);
				blocks[index].slots[slot].key = 0;
				blocks[index].slots[slot].val = 0;
				return iceberg_insert(table, key, value, thread_id);
			}


			metadata->lv1_md[index].block_md[slot] = fprint;
			return true;
		}
	}

	return iceberg_lv2_insert(table, key, value, index, thread_id);
}

bool iceberg_lv3_remove(iceberg_table * table, KeyType key, uint64_t lv3_index, uint8_t thread_id) {

	iceberg_metadata * metadata = table->metadata;
	iceberg_lv3_list * lists = table->level3;

	while(__sync_lock_test_and_set(metadata->lv3_locks + lv3_index, 1));

	if(metadata->lv3_sizes[lv3_index] == 0) return false;

	if(lists[lv3_index].head->key == key) {

		iceberg_lv3_node * old_head = lists[lv3_index].head;
		lists[lv3_index].head = lists[lv3_index].head->next_node;
		free(old_head);

		metadata->lv3_sizes[lv3_index]--;
		pc_add(metadata->lv3_balls, -1, thread_id);
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
			pc_add(metadata->lv3_balls, -1, thread_id);
			metadata->lv3_locks[lv3_index] = 0;

			return true;
		}

		current_node = current_node->next_node;
	}

	metadata->lv3_locks[lv3_index] = 0;
	return false;
}

bool iceberg_lv2_remove(iceberg_table * table, KeyType key, uint64_t lv3_index, uint8_t thread_id) {

	iceberg_metadata * metadata = table->metadata;
	iceberg_lv2_block * blocks = table->level2;

	for(int i = 0; i < D_CHOICES; ++i) {

		uint8_t fprint;
		uint64_t index;

		lv2_split_hash(lv2_hash(key, i), &fprint, &index, metadata);

		__mmask32 md_mask = slot_mask_32(metadata->lv2_md[index].block_md, fprint) & ((1 << (C_LV2 + MAX_LG_LG_N / D_CHOICES)) - 1);
		uint8_t popct = __builtin_popcount(md_mask);

		for(uint8_t i = 0; i < popct; ++i) {

			uint8_t slot = word_select(md_mask, i);

			if (blocks[index].slots[slot].key == key) {

				metadata->lv2_md[index].block_md[slot] = 0;
				pc_add(metadata->lv2_balls, -1, thread_id);
				blocks[index].slots[slot].key = key;
				return true;
			}
		}
	}

	return iceberg_lv3_remove(table, key, lv3_index, thread_id);
}

bool iceberg_remove(iceberg_table * table, KeyType key, uint8_t thread_id) {

	iceberg_metadata * metadata = table->metadata;
	iceberg_lv1_block * blocks = table->level1;

	uint8_t fprint;
	uint64_t index;
	uint8_t log_nblocks = metadata->log_nblocks;

	split_hash(lv1_hash(key), &fprint, &index, log_nblocks);

	__mmask64 md_mask = slot_mask_64(metadata->lv1_md[index].block_md, fprint);
	uint8_t popct = __builtin_popcountll(md_mask);

	for(uint8_t i = 0; i < popct; ++i) {

		uint8_t slot = word_select(md_mask, i);

		if (blocks[index].slots[slot].key == key) {

			metadata->lv1_md[index].block_md[slot] = 0;
			pc_add(metadata->lv1_balls, -1, thread_id);
			blocks[index].slots[slot].key = key;
			return true;
		}
	}

	return iceberg_lv2_remove(table, key, index, thread_id);
}

bool iceberg_lv3_get_value(iceberg_table * table, KeyType key, ValueType **value, uint64_t lv3_index) {

	iceberg_metadata * metadata = table->metadata;
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


bool iceberg_lv2_get_value(iceberg_table * table, KeyType key, ValueType **value, uint64_t lv3_index) {

	iceberg_metadata * metadata = table->metadata;
	iceberg_lv2_block * blocks = table->level2;

	for(uint8_t i = 0; i < D_CHOICES; ++i) {

		uint8_t fprint;
		uint64_t index;

		lv2_split_hash(lv2_hash_inline(key, i), &fprint, &index, metadata);

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

bool iceberg_lv1_get_value(iceberg_table * table, uint64_t index, uint8_t fprint, KeyType key, ValueType **value) {

	iceberg_metadata * metadata = table->metadata;
	iceberg_lv1_block * blocks = table->level1;

	__mmask64 md_mask = slot_mask_64(metadata->lv1_md[index].block_md, fprint);
	
	while (md_mask != 0) {
		int slot = __builtin_ctzll(md_mask);
		md_mask = md_mask & ~(1ULL << slot);

		if (blocks[index].slots[slot].key == key) {
			*value = &blocks[index].slots[slot].val;
			return true;
		}
	}

	return false;
}

bool iceberg_get_value(iceberg_table * table, KeyType key, ValueType **value) {

	iceberg_metadata * md = table->metadata;
	uint8_t log_nblocks = md->log_nblocks;
	uint8_t fprint;
	uint64_t index;
	bool found;
	split_hash(lv1_hash_inline(key), &fprint, &index, log_nblocks);
	if (unlikely(iceberg_second_half(index, log_nblocks) && iceberg_needs_split(md, log_nblocks, index))) {
		uint64_t split_from = iceberg_split_from(index, log_nblocks);

		if (iceberg_lv1_get_value(table, split_from, fprint, key, value)) {
			found = true;
			goto out;
		}
	}

	if (iceberg_lv1_get_value(table, index, fprint, key, value)) {
		found = true;
	} else {
		found = iceberg_lv2_get_value(table, key, value, index);
	}
	
out:
	if (unlikely(log_nblocks != md->log_nblocks)) {
		return iceberg_get_value(table, key, value);
	} else {
		return found;
	}
}

