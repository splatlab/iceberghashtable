#include <cstdio>
#include <cstring>
#include <cassert>
#include <stdlib.h>
#include <immintrin.h>
#include <tmmintrin.h>

#include "hashutil.h"
#include "iceberg_precompute.h"
#include "iceberg_table.h"


uint64_t seed[5] = { 12351327692179052ll, 23246347347385899ll, 35236262354132235ll, 13604702930934770ll, 57439820692984798ll };

uint64_t nonzero_fprint(uint64_t hash) {
	return hash & ((1 << FPRINT_BITS) - 2) ? hash : hash | 2;
}

uint64_t lv1_hash(KeyType key) {
	return nonzero_fprint(MurmurHash64A(&key, 8, seed[0]));
}

uint64_t lv2_hash(KeyType key, uint8_t i) {
	return nonzero_fprint(MurmurHash64A(&key, 8, seed[i + 1]));
}

static inline uint8_t word_select(uint64_t val, int rank) {
	val = _pdep_u64(one[rank], val);
	return _tzcnt_u64(val);
}

uint64_t lv1_balls(iceberg_table * restrict table) {
	//pc_sync(table->metadata->lv1_balls);
	return *(table->metadata->lv1_balls->global_counter);
}

uint64_t lv2_balls(iceberg_table * restrict table) {
	//pc_sync(table->metadata->lv2_balls);
	return *(table->metadata->lv2_balls->global_counter);
}

uint64_t lv3_balls(iceberg_table * restrict table) {
	pc_sync(table->metadata->lv3_balls);
	return *(table->metadata->lv3_balls->global_counter);
}

uint64_t tot_balls(iceberg_table * restrict table) {
	return lv1_balls(table) + lv2_balls(table) + lv3_balls(table);
}

uint64_t total_capacity(iceberg_table * restrict table) {
	return lv3_balls(table) + table->metadata->nblocks * ((1 << SLOT_BITS) + C_LV2 + MAX_LG_LG_N / D_CHOICES);
}

double iceberg_load_factor(iceberg_table * restrict table) {
	return (double)tot_balls(table) / (double)total_capacity(table);
}

void split_hash(uint64_t hash, uint8_t& fprint, uint64_t& index, iceberg_metadata * metadata) {	
	fprint = hash & ((1 << FPRINT_BITS) - 1);
	index = (hash >> FPRINT_BITS) & ((1 << metadata->block_bits) - 1);
}

uint32_t slot_mask_32(uint8_t * metadata, uint8_t fprint) {
	__m256i bcast = _mm256_set1_epi8(fprint);
	__m256i block = _mm256_loadu_si256(reinterpret_cast<__m256i*>(metadata));
	return _mm256_cmp_epi8_mask(bcast, block, _MM_CMPINT_EQ);
}

uint64_t slot_mask_64(uint8_t * metadata, uint8_t fprint) {
	return slot_mask_32(metadata, fprint) | (((uint64_t)slot_mask_32(metadata + 32, fprint)) << 32);
}

iceberg_table * iceberg_init(uint64_t log_slots) {

	iceberg_table * table;

	uint64_t total_blocks = 1 << (log_slots - SLOT_BITS);
  	uint64_t total_size_in_bytes = (sizeof(iceberg_lv1_block) + sizeof(iceberg_lv2_block) + sizeof(iceberg_lv1_block_md) + sizeof(iceberg_lv2_block_md)) * total_blocks;

	table = (iceberg_table *)malloc(sizeof(iceberg_table));
	assert(table);

	table->level1 = (iceberg_lv1_block *)malloc(sizeof(iceberg_lv1_block) * total_blocks);
	table->level2 = (iceberg_lv2_block *)malloc(sizeof(iceberg_lv2_block) * total_blocks);
	table->level3 = (iceberg_lv3_list *)malloc(sizeof(iceberg_lv3_list) * total_blocks);

	table->metadata = (iceberg_metadata *)malloc(sizeof(iceberg_metadata));
	table->metadata->total_size_in_bytes = total_size_in_bytes;
	table->metadata->nslots = 1 << log_slots;
	table->metadata->nblocks = total_blocks;
	table->metadata->block_bits = log_slots - SLOT_BITS;

	table->metadata->lv1_balls = (pc_t *)malloc(sizeof(pc_t));
	int64_t * lv1_ctr = (int64_t *)malloc(sizeof(int64_t));
	* lv1_ctr = 0;
	pc_init(table->metadata->lv1_balls, lv1_ctr, 8, 1000);
	
	table->metadata->lv2_balls = (pc_t *)malloc(sizeof(pc_t));
	int64_t * lv2_ctr = (int64_t *)malloc(sizeof(int64_t));
	* lv2_ctr = 0;
	pc_init(table->metadata->lv2_balls, lv2_ctr, 8, 1000);

	table->metadata->lv3_balls = (pc_t *)malloc(sizeof(pc_t));
	int64_t * lv3_ctr = (int64_t *)malloc(sizeof(int64_t));
	* lv3_ctr = 0;
	pc_init(table->metadata->lv3_balls, lv3_ctr, 8, 1000);

	table->metadata->lv1_md = (iceberg_lv1_block_md *)malloc(sizeof(iceberg_lv1_block_md) * total_blocks);
	table->metadata->lv2_md = (iceberg_lv2_block_md *)malloc(sizeof(iceberg_lv2_block_md) * total_blocks);
	table->metadata->lv3_sizes = (uint64_t *)malloc(sizeof(uint64_t) * total_blocks);
	table->metadata->lv3_locks = (uint8_t *)malloc(sizeof(uint8_t) * total_blocks);

	for (uint64_t i = 0; i < total_blocks; ++i) {

		for (uint64_t j = 0; j < (1 << SLOT_BITS); ++j) {
			table->metadata->lv1_md[i].block_md[j] = 0;
			table->level1[i].slots[j].key = table->level1[i].slots[j].val = 0;
		}

		for (uint64_t j = 0; j < C_LV2 + MAX_LG_LG_N / D_CHOICES; ++j) {
			table->metadata->lv2_md[i].block_md[j] = 0;
			table->level2[i].slots[j].key = table->level2[i].slots[j].val = 0;
		}

		table->metadata->lv3_sizes[i] = table->metadata->lv3_locks[i] = 0;
	}

	return table;
}

bool iceberg_lv3_insert(iceberg_table * restrict table, KeyType key, ValueType value, uint64_t lv3_index, uint8_t thread_id) {

	iceberg_metadata * restrict metadata = table->metadata;
	iceberg_lv3_list * restrict lists = table->level3;

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

bool iceberg_lv2_insert(iceberg_table * restrict table, KeyType key, ValueType value, uint64_t lv3_index, uint8_t thread_id) {

	iceberg_metadata * restrict metadata = table->metadata;
	iceberg_lv2_block * restrict blocks = table->level2;

	if(*(table->metadata->lv2_balls->global_counter) == (int64_t)(C_LV2 * metadata->nblocks)) return iceberg_lv3_insert(table, key, value, lv3_index, thread_id);

	uint8_t fprint1, fprint2;
	uint64_t index1, index2;

	split_hash(lv2_hash(key, 0), fprint1, index1, metadata);
	split_hash(lv2_hash(key, 1), fprint2, index2, metadata);

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

bool iceberg_insert(iceberg_table * restrict table, KeyType key, ValueType value, uint8_t thread_id) {

	iceberg_metadata * restrict metadata = table->metadata;
	iceberg_lv1_block * restrict blocks = table->level1;	

	uint8_t fprint;
	uint64_t index;

	split_hash(lv1_hash(key), fprint, index, metadata);

	__mmask64 md_mask = slot_mask_64(metadata->lv1_md[index].block_md, 0);

	uint8_t popct = __builtin_popcountll(md_mask);

	for(uint8_t i = 0; i < popct; ++i) {
	
		uint8_t slot = word_select(md_mask, i);

		if(__sync_bool_compare_and_swap(metadata->lv1_md[index].block_md + slot, 0, 1)) {
			
			pc_add(metadata->lv1_balls, 1, thread_id);
			blocks[index].slots[slot].key = key;
			blocks[index].slots[slot].val = value;

			metadata->lv1_md[index].block_md[slot] = fprint;
			return true;
		}
	}

	return iceberg_lv2_insert(table, key, value, index, thread_id);
}

bool iceberg_lv3_remove(iceberg_table * restrict table, KeyType key, uint64_t lv3_index, uint8_t thread_id) {

	iceberg_metadata * restrict metadata = table->metadata;
	iceberg_lv3_list * restrict lists = table->level3;

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

bool iceberg_lv2_remove(iceberg_table * restrict table, KeyType key, uint64_t lv3_index, uint8_t thread_id) {

	iceberg_metadata * restrict metadata = table->metadata;
	iceberg_lv2_block * restrict blocks = table->level2;

	for(int i = 0; i < D_CHOICES; ++i) {

		uint8_t fprint;
		uint64_t index;

		split_hash(lv2_hash(key, i), fprint, index, metadata);

		__mmask32 md_mask = slot_mask_32(metadata->lv2_md[index].block_md, fprint) & ((1 << (C_LV2 + MAX_LG_LG_N / D_CHOICES)) - 1);
		uint8_t popct = __builtin_popcount(md_mask);

		for(uint8_t i = 0; i < popct; ++i) {

			uint8_t slot = word_select(md_mask, i);

			while(1) {

				if(__sync_bool_compare_and_swap(&blocks[index].slots[slot].key, key, UINT64_MAX)) {

					metadata->lv2_md[index].block_md[slot] = 0;
					pc_add(metadata->lv2_balls, -1, thread_id);
					blocks[index].slots[slot].key = key;
					return true;
				
				} else {
					KeyType slot_key = blocks[index].slots[slot].key;
					if(slot_key != UINT64_MAX && slot_key != key) break; 
				}
			}
		}
	}

	return iceberg_lv3_remove(table, key, lv3_index, thread_id);
}

bool iceberg_remove(iceberg_table * restrict table, KeyType key, uint8_t thread_id) {

	iceberg_metadata * restrict metadata = table->metadata;
	iceberg_lv1_block * restrict blocks = table->level1;

	uint8_t fprint;
	uint64_t index;

	split_hash(lv1_hash(key), fprint, index, metadata);

	__mmask64 md_mask = slot_mask_64(metadata->lv1_md[index].block_md, fprint);
	uint8_t popct = __builtin_popcountll(md_mask);

	for(uint8_t i = 0; i < popct; ++i) {

		uint8_t slot = word_select(md_mask, i);

		while(1) {

			if(__sync_bool_compare_and_swap(&blocks[index].slots[slot].key, key, UINT64_MAX)) {

				metadata->lv1_md[index].block_md[slot] = 0;
				pc_add(metadata->lv1_balls, -1, thread_id);
				blocks[index].slots[slot].key = key;
				return true;

			} else {
				KeyType slot_key = blocks[index].slots[slot].key;
				if(slot_key != UINT64_MAX && slot_key != key) break;
			}
		}
	}

	return iceberg_lv2_remove(table, key, index, thread_id);
}

bool iceberg_lv3_get_value(iceberg_table * restrict table, KeyType key, ValueType& value, uint64_t lv3_index) {

	iceberg_metadata * restrict metadata = table->metadata;
	iceberg_lv3_list * restrict lists = table->level3;

	while(__sync_lock_test_and_set(metadata->lv3_locks + lv3_index, 1));

	if(!metadata->lv3_sizes[lv3_index]) {
		metadata->lv3_locks[lv3_index] = 0;
		return false;
	}

	iceberg_lv3_node * current_node = lists[lv3_index].head;

	for(uint8_t i = 0; i < metadata->lv3_sizes[lv3_index]; ++i) {

		if(current_node->key == key) {

			value = current_node->val;
			metadata->lv3_locks[lv3_index] = 0;
			return true;
		}

		current_node = current_node->next_node;
	}

	metadata->lv3_locks[lv3_index] = 0;
	return false;
}


bool iceberg_lv2_get_value(iceberg_table * restrict table, KeyType key, ValueType& value, uint64_t lv3_index) {

	iceberg_metadata * restrict metadata = table->metadata;
	iceberg_lv2_block * restrict blocks = table->level2;

	for(uint8_t i = 0; i < D_CHOICES; ++i) {

		uint8_t fprint;
		uint64_t index;

		split_hash(lv2_hash(key, i), fprint, index, metadata);

		__mmask32 md_mask = slot_mask_32(metadata->lv2_md[index].block_md, fprint) & ((1 << (C_LV2 + MAX_LG_LG_N / D_CHOICES)) - 1);
		uint8_t popct = __builtin_popcount(md_mask);

		for(uint8_t i = 0; i < popct; ++i) {

			uint8_t slot = word_select(md_mask, i);

			while(1) {

				if(__sync_bool_compare_and_swap(&blocks[index].slots[slot].key, key, UINT64_MAX)) {

					value = blocks[index].slots[slot].val;
					blocks[index].slots[slot].key = key;
					return true;

				} else  {
					KeyType slot_key = blocks[index].slots[slot].key;
					if(slot_key != UINT64_MAX && slot_key != key) break;
				}
			}
		}
	}

	return iceberg_lv3_get_value(table, key, value, lv3_index);
}

bool iceberg_get_value(iceberg_table * restrict table, KeyType key, ValueType& value) {

	iceberg_metadata * restrict metadata = table->metadata;
	iceberg_lv1_block * restrict blocks = table->level1;

	uint8_t fprint;
	uint64_t index;

	split_hash(lv1_hash(key), fprint, index, metadata);

	__mmask64 md_mask = slot_mask_64(metadata->lv1_md[index].block_md, fprint);

	uint8_t popct = __builtin_popcountll(md_mask);

	for(uint8_t i = 0; i < popct; ++i) {

		uint8_t slot = word_select(md_mask, i);

		while(1) {

			if(__sync_bool_compare_and_swap(&blocks[index].slots[slot].key, key, UINT64_MAX)) {

				value = blocks[index].slots[slot].val;
				blocks[index].slots[slot].key = key;
				return true;

			} else {
				KeyType slot_key = blocks[index].slots[slot].key;
				if(slot_key != UINT64_MAX && slot_key != key) break;
			}
		}
	}

	return iceberg_lv2_get_value(table, key, value, index);
}

