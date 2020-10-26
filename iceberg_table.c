#include <cstdio>
#include <cstring>
#include <cassert>
#include <stdlib.h>
#include <immintrin.h>
#include <tmmintrin.h>

#include "iceberg_precompute.h"
#include "iceberg_table.h"

uint64_t get_hash(KeyType key) { //placeholder - will replace with MurmurHash
	return (key % 256) ? key : key + 1;
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

	table->metadata = (iceberg_metadata *)malloc(sizeof(*(table->metadata)) + (1 << log_slots) * sizeof(uint8_t));

	table->metadata->total_size_in_bytes = total_size_in_bytes;
	table->metadata->nslots = 1 << log_slots;
	table->metadata->nblocks = total_blocks;
	table->metadata->block_bits = log_slots - BITS_PER_BUCKET;

	for (uint64_t i = 0; i < (1 << log_slots); ++i)
		table->metadata->block_md[i] = 0;

	return table;
}

bool iceberg_insert(iceberg_table * restrict table, KeyType key, ValueType value) {
	
	iceberg_metadata * restrict metadata = table->metadata;
	iceberg_block * restrict blocks = table->blocks;

	uint64_t hash = get_hash(key);

	uint64_t index = hash & ((1 << metadata->block_bits) - 1);
	uint8_t fprint = (hash >> metadata->block_bits) & ((1 << FPRINT_BITS) - 1);
	uint64_t tag = hash >> (metadata->block_bits + FPRINT_BITS);
	
	__m256i bcast = _mm256_set1_epi8(0);
	__m256i block = _mm256_loadu_si256(reinterpret_cast<__m256i*>(&metadata->block_md[index * BUCKETS_PER_BLOCK]));
	__mmask32 result = _mm256_cmp_epi8_mask(bcast, block, _MM_CMPINT_EQ);
	
	if(!result) return false;

	uint8_t slot = word_select(result, 0);

	metadata->block_md[index * BUCKETS_PER_BLOCK + slot] = fprint;
	blocks[index].tags[slot] = tag;
	blocks[index].vals[slot] = value;
	
	return true;
}

bool iceberg_remove(iceberg_table * restrict table, KeyType key, ValueType value) {

	iceberg_metadata * restrict metadata = table->metadata;
	iceberg_block * restrict blocks = table->blocks;

	uint64_t hash = get_hash(key);

	uint64_t index = hash & ((1 << metadata->block_bits) - 1);
	uint8_t fprint = (hash >> metadata->block_bits) & ((1 << FPRINT_BITS) - 1);
	uint64_t tag = hash >> (metadata->block_bits + FPRINT_BITS);

	__m256i bcast = _mm256_set1_epi8(fprint);
	__m256i block = _mm256_loadu_si256(reinterpret_cast<__m256i*>(&metadata->block_md[index * BUCKETS_PER_BLOCK]));
	__mmask32 result = _mm256_cmp_epi8_mask(bcast, block, _MM_CMPINT_EQ);

	if(!result) return false;
	
	uint8_t idx = 0;
	while(1) {
		
		uint8_t slot = word_select(result, idx);
		if(slot == 64) return false;

		if(blocks[index].tags[slot] == tag && blocks[index].vals[slot] == value) {
			metadata->block_md[index * BUCKETS_PER_BLOCK + slot] = 0;
			return true;
		}
		
		idx++;
	}
}

bool iceberg_get_value(iceberg_table * restrict table, KeyType key, ValueType& value) {

	iceberg_metadata * restrict metadata = table->metadata;
	iceberg_block * restrict blocks = table->blocks;

	uint64_t hash = get_hash(key);

	uint64_t index = hash & ((1 << metadata->block_bits) - 1);
	uint8_t fprint = (hash >> metadata->block_bits) & ((1 << FPRINT_BITS) - 1);
	uint64_t tag = hash >> (metadata->block_bits + FPRINT_BITS);

	__m256i bcast = _mm256_set1_epi8(fprint);
	__m256i block = _mm256_loadu_si256(reinterpret_cast<__m256i*>(&metadata->block_md[index * BUCKETS_PER_BLOCK]));
	__mmask32 result = _mm256_cmp_epi8_mask(bcast, block, _MM_CMPINT_EQ);

	if(!result) return false;
	
	uint8_t idx = 0;
	while(1) {
		
		uint8_t slot = word_select(result, idx);
		if(slot == 64) return false;

		if(blocks[index].tags[slot] == tag) {
			value = blocks[index].vals[slot];
			return true;
		}
		
		idx++;
	}
}
