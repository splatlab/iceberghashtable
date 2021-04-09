#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <stdlib.h>
#include <immintrin.h>
#include <tmmintrin.h>
#include <sys/mman.h>
#include <libpmem.h>

#include "hashutil.h"
#include "iceberg_precompute.h"
#include "iceberg_table.h"

#define likely(x)   __builtin_expect((x),1)
#define unlikely(x) __builtin_expect((x),0)

#define PMEM_PATH "/mnt/pmem0"
#define FILENAME_LEN 1024
#define NUM_LEVEL3_NODES 2048

#define likely(x)   __builtin_expect((x),1)
#define unlikely(x) __builtin_expect((x),0)

uint64_t seed[5] = { 12351327692179052ll, 23246347347385899ll, 35236262354132235ll, 13604702930934770ll, 57439820692984798ll };

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
	pc_sync(table->metadata->lv3_balls);
	return *(table->metadata->lv3_balls->global_counter);
}

uint64_t tot_balls(iceberg_table * table) {
	return lv1_balls(table) + lv2_balls(table) + lv3_balls(table);
}

uint64_t total_capacity(iceberg_table * table) {
	return lv3_balls(table) + table->metadata->nblocks * ((1 << SLOT_BITS) + C_LV2 + MAX_LG_LG_N / D_CHOICES);
}

double iceberg_load_factor(iceberg_table * table) {
	return (double)tot_balls(table) / (double)total_capacity(table);
}

uint32_t slot_mask_32(const uint8_t * metadata, uint8_t fprint) {
	__m256i bcast = _mm256_set1_epi8(fprint);
	__m256i block = _mm256_loadu_si256((const __m256i *)(metadata));
	return _mm256_cmp_epi8_mask(bcast, block, _MM_CMPINT_EQ);
}

uint64_t slot_mask_64(const uint8_t * metadata, uint8_t fprint) {
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

iceberg_table * iceberg_init(uint64_t log_slots) {

	iceberg_table * table;

	uint64_t total_blocks = 1 << (log_slots - SLOT_BITS);
  	uint64_t total_size_in_bytes = (sizeof(iceberg_lv1_block) + sizeof(iceberg_lv2_block) + sizeof(iceberg_lv1_block_md) + sizeof(iceberg_lv2_block_md)) * total_blocks;

	table = (iceberg_table *)malloc(sizeof(iceberg_table));
	assert(table);

        size_t mapped_len;
        int is_pmem;

        size_t level1_size = round_up(sizeof(iceberg_lv1_block) * total_blocks, 2 * 1024 * 1024);
        char level1_filename[FILENAME_LEN];
        sprintf(level1_filename, "%s/level1", PMEM_PATH);
        if ((table->level1 = (iceberg_lv1_block *)pmem_map_file(level1_filename,
                    level1_size, PMEM_FILE_CREATE, 0666, &mapped_len,
                    &is_pmem)) == NULL) {
		perror("pmem_map_file");
		exit(1);
	}
	assert(is_pmem);
	assert(mapped_len == level1_size);

        size_t level2_size = round_up(sizeof(iceberg_lv2_block) * total_blocks, 2 * 1024 * 1024);
        char level2_filename[FILENAME_LEN];
        sprintf(level2_filename, "%s/level2", PMEM_PATH);
        if ((table->level2 = (iceberg_lv2_block *)pmem_map_file(level2_filename,
                    level2_size, PMEM_FILE_CREATE, 0666, &mapped_len,
                    &is_pmem)) == NULL) {
		perror("pmem_map_file");
		exit(1);
	}
	assert(is_pmem);
	assert(mapped_len == level2_size);

	size_t level3_size = sizeof(iceberg_lv3_list) * total_blocks;
        char level3_filename[FILENAME_LEN];
        sprintf(level3_filename, "%s/level3", PMEM_PATH);
        if ((table->level3 = (iceberg_lv3_list *)pmem_map_file(level3_filename,
                    level3_size, PMEM_FILE_CREATE, 0666, &mapped_len,
                    &is_pmem)) == NULL) {
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
				level3_nodes_size, PMEM_FILE_CREATE, 0666,
				&mapped_len, &is_pmem);
	if (table->level3_nodes == NULL) {
		perror("pmem_map_file");
		exit(1);
	}
	assert(is_pmem);
	assert(mapped_len == level3_nodes_size);

	table->metadata = (iceberg_metadata *)malloc(sizeof(iceberg_metadata));
	table->metadata->total_size_in_bytes = total_size_in_bytes;
	table->metadata->nslots = 1 << log_slots;
	table->metadata->nblocks = total_blocks;
	table->metadata->block_bits = log_slots - SLOT_BITS;

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

	int mmap_flags = MAP_SHARED | MAP_ANONYMOUS | MAP_POPULATE | MAP_HUGETLB;
	size_t lv1_md_size = sizeof(iceberg_lv1_block_md) * total_blocks + 64;
	table->metadata->lv1_md = (iceberg_lv1_block_md *)mmap(NULL, lv1_md_size, PROT_READ | PROT_WRITE, mmap_flags, 0, 0);
	size_t lv2_md_size = sizeof(iceberg_lv2_block_md) * total_blocks + 32;
	table->metadata->lv2_md = (iceberg_lv2_block_md *)mmap(NULL, lv2_md_size, PROT_READ | PROT_WRITE, mmap_flags, 0, 0);
	table->metadata->lv3_sizes = (uint64_t *)malloc(sizeof(uint64_t) * total_blocks);
	table->metadata->lv3_locks = (uint8_t *)malloc(sizeof(uint8_t) * total_blocks);

	pmem_memset_persist(table->level1, 0, level1_size);
	pmem_memset_persist(table->level2, 0, level2_size);
	memset(table->level3, 0, level3_size);

	for (uint64_t i = 0; i < total_blocks; i++) {
		table->level3[i].head_idx = -1;
	}
	pmem_persist(table->level3, level3_size);
	pmem_memset_persist(table->level3_nodes, 0, level3_nodes_size);

	memset((char *)table->metadata->lv1_md, 0, lv1_md_size);
	memset((char *)table->metadata->lv2_md, 0, lv2_md_size);

	memset(table->metadata->lv3_sizes, 0, total_blocks * sizeof(uint64_t));
	memset((char *)table->metadata->lv3_locks, 0, total_blocks * sizeof(uint8_t));

	return table;
}

void iceberg_dismount(iceberg_table * table) {

	size_t total_blocks = table->metadata->nblocks;

	size_t lv1_md_size = sizeof(iceberg_lv1_block_md) * total_blocks + 64;
	munmap(table->metadata->lv1_md, lv1_md_size);
	size_t lv2_md_size = sizeof(iceberg_lv2_block_md) * total_blocks + 32;
	munmap(table->metadata->lv2_md, lv2_md_size);

	free(table->metadata->lv3_sizes);
	free(table->metadata->lv3_locks);

	pc_destructor(table->metadata->lv1_balls);
	free(table->metadata->lv1_balls->global_counter);
	free(table->metadata->lv1_balls);

	pc_destructor(table->metadata->lv2_balls);
	free(table->metadata->lv2_balls->global_counter);
	free(table->metadata->lv2_balls);

	pc_destructor(table->metadata->lv3_balls);
	free(table->metadata->lv3_balls->global_counter);
	free(table->metadata->lv3_balls);

	free(table->metadata);

        size_t level1_size = round_up(sizeof(iceberg_lv1_block) * total_blocks, 2 * 1024 * 1024);
	pmem_unmap(table->level1, level1_size);

        size_t level2_size = round_up(sizeof(iceberg_lv2_block) * total_blocks, 2 * 1024 * 1024);
	pmem_unmap(table->level2, level2_size);

	size_t level3_size = sizeof(iceberg_lv3_list) * total_blocks;
	pmem_unmap(table->level3, level3_size);

	size_t level3_nodes_size = NUM_LEVEL3_NODES * sizeof(iceberg_lv3_node);
	pmem_unmap(table->level3_nodes, level3_nodes_size);
}

iceberg_table * iceberg_mount(uint64_t log_slots) {

	iceberg_table * table;

	uint64_t total_blocks = 1 << (log_slots - SLOT_BITS);
	uint64_t total_size_in_bytes = (sizeof(iceberg_lv1_block) + sizeof(iceberg_lv2_block) + sizeof(iceberg_lv1_block_md) + sizeof(iceberg_lv2_block_md)) * total_blocks;

	table = (iceberg_table *)malloc(sizeof(iceberg_table));
	assert(table);

        size_t mapped_len;
        int is_pmem;

        size_t level1_size = round_up(sizeof(iceberg_lv1_block) * total_blocks, 2 * 1024 * 1024);
        char level1_filename[FILENAME_LEN];
        sprintf(level1_filename, "%s/level1", PMEM_PATH);
        if ((table->level1 = (iceberg_lv1_block *)pmem_map_file(level1_filename,
                    0, 0, 0666, &mapped_len,
                    &is_pmem)) == NULL) {
		perror("pmem_map_file");
		exit(1);
	}
	assert(is_pmem);
	assert(mapped_len == level1_size);

        size_t level2_size = round_up(sizeof(iceberg_lv2_block) * total_blocks, 2 * 1024 * 1024);
        char level2_filename[FILENAME_LEN];
        sprintf(level2_filename, "%s/level2", PMEM_PATH);
        if ((table->level2 = (iceberg_lv2_block *)pmem_map_file(level2_filename,
		    0, 0, 0666, &mapped_len, &is_pmem)) == NULL) {
		perror("pmem_map_file");
		exit(1);
	}
	assert(is_pmem);
	assert(mapped_len == level2_size);

	size_t level3_size = sizeof(iceberg_lv3_list) * total_blocks;
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
	table->level3_nodes = (iceberg_lv3_node *)pmem_map_file(level3_nodes_filename,
				0, 0, 0666, &mapped_len, &is_pmem);
	if (table->level3_nodes == NULL) {
		perror("pmem_map_file");
		exit(1);
	}
	assert(is_pmem);
	assert(mapped_len == level3_nodes_size);

	table->metadata = (iceberg_metadata *)malloc(sizeof(iceberg_metadata));
	table->metadata->total_size_in_bytes = total_size_in_bytes;
	table->metadata->nslots = 1 << log_slots;
	table->metadata->nblocks = total_blocks;
	table->metadata->block_bits = log_slots - SLOT_BITS;

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

	int mmap_flags = MAP_SHARED | MAP_ANONYMOUS | MAP_POPULATE | MAP_HUGETLB;
	size_t lv1_md_size = sizeof(iceberg_lv1_block_md) * total_blocks + 64;
	table->metadata->lv1_md = (iceberg_lv1_block_md *)mmap(NULL, lv1_md_size, PROT_READ | PROT_WRITE, mmap_flags, 0, 0);
	size_t lv2_md_size = sizeof(iceberg_lv2_block_md) * total_blocks + 32;
	table->metadata->lv2_md = (iceberg_lv2_block_md *)mmap(NULL, lv2_md_size, PROT_READ | PROT_WRITE, mmap_flags, 0, 0);
	table->metadata->lv3_sizes = (uint64_t *)malloc(sizeof(uint64_t) * total_blocks);
	table->metadata->lv3_locks = (uint8_t *)malloc(sizeof(uint8_t) * total_blocks);

	for (uint64_t i = 0; i < total_blocks; i++) {
		iceberg_lv1_block *block = &table->level1[i];
		uint8_t *block_md = table->metadata->lv1_md[i].block_md;
		for (uint64_t slot = 0; slot < 1 << SLOT_BITS; slot++) {
			KeyType key = block->slots[slot].key;
			if (key != 0) {
				uint8_t slot_choice;
				uint8_t fprint;
				uint64_t index;

				split_hash(key, &slot_choice, &fprint, &index, table->metadata, 0);
				assert(index == i);
				block_md[slot] = fprint;
				pc_add(table->metadata->lv1_balls, 1, 0);
			}
		}
	}

	for (uint64_t i = 0; i < total_blocks; i++) {
		iceberg_lv2_block *block = &table->level2[i];
		uint8_t *block_md = table->metadata->lv2_md[i].block_md;
		for (uint64_t slot = 0; slot < C_LV2 + MAX_LG_LG_N / D_CHOICES; slot++) {
			KeyType key = block->slots[slot].key;
			if (key != 0) {
				uint8_t slot_choice;
				uint8_t fprint;
				uint64_t index;

				split_hash(key, &slot_choice, &fprint, &index, table->metadata, 1);
				if (index != i) {
					split_hash(key, &slot_choice, &fprint, &index, table->metadata, 2);
				}

				assert(index == i);
				block_md[slot] = fprint;
				pc_add(table->metadata->lv2_balls, 1, 0);
			}
		}
	}

	memset(table->metadata->lv3_sizes, 0, sizeof(uint64_t) * total_blocks);
	for (uint64_t i = 0; i < total_blocks; i++) {
		ptrdiff_t idx = table->level3[i].head_idx;
		while (idx != -1) {
			idx = table->level3_nodes[idx].next_idx;
			table->metadata->lv3_sizes[i]++;
			pc_add(table->metadata->lv3_balls, 1, 0);
		}
	}

	memset(table->metadata->lv3_locks, 0, sizeof(uint8_t) * total_blocks);

	return table;
}

bool iceberg_lv3_insert(iceberg_table * table, KeyType key, ValueType value, uint64_t lv3_index, uint8_t thread_id) {

	iceberg_metadata * metadata = table->metadata;
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
	pc_add(metadata->lv3_balls, 1, thread_id);
	metadata->lv3_locks[lv3_index] = 0;

	return true;
}

bool iceberg_lv2_insert(iceberg_table * table, KeyType key, ValueType value, uint64_t lv3_index, uint8_t thread_id) {

	iceberg_metadata * metadata = table->metadata;
	iceberg_lv2_block * blocks = table->level2;

	if (*(table->metadata->lv2_balls->global_counter) == (int64_t)(C_LV2 * metadata->nblocks)) {
		return iceberg_lv3_insert(table, key, value, lv3_index, thread_id);
	}

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

	uint8_t start = popct1 == 0 ? 0 : slot_choice % popct1;
	for(uint8_t i = start; i != start + popct1; ++i) {

		uint8_t slot = word_select(md_mask1, i % popct1);

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

bool iceberg_insert(iceberg_table * table, KeyType key, ValueType value, uint8_t thread_id) {

	iceberg_metadata * metadata = table->metadata;
	iceberg_lv1_block * blocks = table->level1;	

	uint8_t fprint;
	uint8_t slot_choice;
	uint64_t index;

	split_hash(key, &slot_choice, &fprint, &index, metadata, 0);

	__mmask64 md_mask = slot_mask_64(metadata->lv1_md[index].block_md, 0);

	uint8_t popct = __builtin_popcountll(md_mask);

	uint8_t start = popct == 0 ? 0 : slot_choice % popct;
	for(uint8_t i = start; i != start + popct; ++i) {

		uint8_t slot = word_select(md_mask, i % popct);

		if(__sync_bool_compare_and_swap(metadata->lv1_md[index].block_md + slot, 0, 1)) {

			pc_add(metadata->lv1_balls, 1, thread_id);
			blocks[index].slots[slot].key = key;
			blocks[index].slots[slot].val = value;
			metadata->lv1_md[index].block_md[slot] = fprint;
                        pmem_persist(&blocks[index].slots[slot], sizeof(kv_pair));
			return true;
		}
	}

	return iceberg_lv2_insert(table, key, value, index, thread_id);
}

bool iceberg_lv3_remove(iceberg_table * table, KeyType key, uint64_t lv3_index, uint8_t thread_id) {

	iceberg_metadata * metadata = table->metadata;
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
		pc_add(metadata->lv3_balls, -1, thread_id);
		metadata->lv3_locks[lv3_index] = 0;

		return true;
	}

	iceberg_lv3_node * current_node = head;

	for(uint64_t i = 0; i < metadata->lv3_sizes[lv3_index] - 1; ++i) {

		assert(current_node->next_idx != -1);
		iceberg_lv3_node *next_node = &lv3_nodes[current_node->next_idx];
		if(next_node->key == key) {


			current_node->next_idx = next_node->next_idx;
			pmem_memset_persist(next_node, 0, sizeof(*next_node));

			metadata->lv3_sizes[lv3_index]--;
			pc_add(metadata->lv3_balls, -1, thread_id);
			metadata->lv3_locks[lv3_index] = 0;

			return true;
		}

		current_node = next_node;
	}

	metadata->lv3_locks[lv3_index] = 0;
	return false;
}

bool iceberg_lv2_remove(iceberg_table * table, KeyType key, uint64_t lv3_index, uint8_t thread_id) {

	iceberg_metadata * metadata = table->metadata;
	iceberg_lv2_block * blocks = table->level2;

	for(int i = 0; i < D_CHOICES; ++i) {

		uint8_t fprint;
		uint8_t slot_choice; // ununsed
		uint64_t index;

		split_hash(key, &slot_choice, &fprint, &index, metadata, 1 + i);

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
	uint8_t slot_choice; // unused
	uint64_t index;

	split_hash(key, &slot_choice, &fprint, &index, metadata, 0);

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


bool iceberg_lv2_get_value(iceberg_table * table, KeyType key, ValueType **value, uint64_t lv3_index) {

	iceberg_metadata * metadata = table->metadata;
	iceberg_lv2_block * blocks = table->level2;

	for(uint8_t i = 0; i < D_CHOICES; ++i) {

		uint8_t fprint;
		uint8_t slot_choice; // unused
		uint64_t index;

		split_hash(key, &slot_choice, &fprint, &index, metadata, i + 1);

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

bool iceberg_get_value(iceberg_table * table, KeyType key, ValueType **value) {

	iceberg_metadata * metadata = table->metadata;
	iceberg_lv1_block * blocks = table->level1;

	uint8_t fprint;
	uint8_t slot_choice;
	uint64_t index;

	split_hash(key, &slot_choice, &fprint, &index, metadata, 0);

	uint64_t md_mask = slot_mask_64(metadata->lv1_md[index].block_md, fprint);

	while (md_mask != 0) {
		int slot = __builtin_ctzll(md_mask);
		md_mask = md_mask & ~(1ULL << slot);

		if (blocks[index].slots[slot].key == key) {
			*value = &blocks[index].slots[slot].val;
			return true;
		}
	}

	return iceberg_lv2_get_value(table, key, value, index);
}

