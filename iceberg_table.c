#include <cstdio>
#include <cstring>
#include <cassert>
#include <stdlib.h>
#include <immintrin.h>
#include <tmmintrin.h>

#include "iceberg_precompute.h"
#include "iceberg_table.h"

uint64_t get_hash(KeyType key) { //placeholder - will replace with MurmurHash
	return key;
}

static inline int word_rank(uint64_t val) {
	return __builtin_popcountll(val);
}

// Returns the position of the rank'th 1.  (rank = 0 returns the 1st 1)
// Returns 64 if there are fewer than rank+1 1s.
static inline uint64_t word_select(uint64_t val, int rank) {
	val = _pdep_u64(one[rank], val);
	return _tzcnt_u64(val);
}

iceberg_table * iceberg_init(uint64_t log_slots) {

	iceberg_table * table;

	uint64_t total_blocks = 1 << (log_slots - BITS_PER_BUCKET);
  	uint64_t total_size_in_bytes = sizeof(iceberg_block) * total_blocks;

    	table = (iceberg_table *)malloc(sizeof(*table) + total_size_in_bytes);

    	printf("Size: %ld\n",total_size_in_bytes);
	assert(table);
    
	table->metadata.total_size_in_bytes = total_size_in_bytes;
	table->metadata.nslots = total_blocks * BUCKETS_PER_BLOCK;
	table->metadata.nblocks = total_blocks;
	table->metadata.nelts = 0;
	table->metadata.block_bits = log_slots - BITS_PER_BUCKET;

	for (uint64_t i = 0; i < total_blocks; ++i)
		table->blocks[i].metadata = UINT64_MAX;

	return table;
}

static inline void update_tags(iceberg_block * restrict block, uint64_t index, uint64_t tag, ValueType value) {
	uint64_t sz = BUCKETS_PER_BLOCK - index - 1;
	memmove(&block->tags[index + 1], &block->tags[index], sz);
	memmove(&block->vals[index + 1], &block->vals[index], sz);
	block->tags[index] = tag;
	block->vals[index] = value;
}

static inline void update_md(uint64_t& md, uint64_t index) {
	md = _pdep_u64(md, pdep_table[index]);
}

bool iceberg_insert(iceberg_table * restrict table, KeyType key, ValueType value) {

	iceberg_metadata * restrict metadata = &table->metadata;
	iceberg_block * restrict blocks = table->blocks;

	uint64_t hash = get_hash(key);
	uint64_t index = hash & ((1 << metadata->block_bits) - 1);
	uint64_t offset = (hash >> metadata->block_bits) & (BUCKETS_PER_BLOCK - 1);
	uint64_t tag = hash >> (metadata->block_bits + BITS_PER_BUCKET);

	uint64_t select_index = word_select(blocks[index].metadata, offset);
	uint64_t slot_index = select_index - offset;

	if(word_rank(blocks[index].metadata) == BUCKETS_PER_BLOCK) {
		fprintf(stderr, "No space in Level 1.\n");
		exit(EXIT_FAILURE);
	}

	update_tags(&blocks[index], slot_index, tag, value);
	update_md(blocks[index].metadata, select_index);

	return true;
}

bool iceberg_remove(iceberg_table * restrict table, KeyType key) {
	
	return true;
}

bool iceberg_get_value(iceberg_table * restrict table, KeyType key, ValueType& value) {

	iceberg_metadata * restrict metadata = &table->metadata;
	iceberg_block * restrict blocks = table->blocks;

	uint64_t hash = get_hash(key);
	uint64_t index = hash & ((1 << metadata->block_bits) - 1);
	uint64_t offset = (hash >> metadata->block_bits) & (BUCKETS_PER_BLOCK - 1);
	uint64_t tag = hash >> (metadata->block_bits + BITS_PER_BUCKET);

	uint64_t start = offset ? word_select(blocks[index].metadata, offset - 1) + 1 - offset : 0;
	uint64_t end = word_select(blocks[index].metadata, offset) - offset;

	for(int i = start; i < end; ++i)
		if(blocks[index].tags[i] == tag) {
			value = blocks[index].vals[i];
			return true;
		}

	return false;
}
