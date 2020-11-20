#include <cstdio>
#include <cstring>
#include <cassert>
#include <stdlib.h>
#include <immintrin.h>
#include <tmmintrin.h>

#include "hashutil.h"
#include "iceberg_precompute.h"
#include "iceberg_table.h"


uint64_t seed[6] = { 12351327692179052ll, 23246347347385899ll, 35236262354132235ll, 13604702930934770ll, 57439820692984798ll, 23602879368919876ll };

uint64_t nonzero_fprint(uint64_t hash) {
	return hash & ((1 << FPRINT_BITS) - 1) ? hash : hash | 1;
}

uint64_t lv1_hash(KeyType key) {
	return nonzero_fprint(MurmurHash64A(&key, 8, seed[0]));
}

uint64_t lv2_hash(KeyType key, uint8_t i) {
	return nonzero_fprint(MurmurHash64A(&key, 8, seed[i + 1]));
}

uint64_t lv3_hash(KeyType key) {
	return nonzero_fprint(MurmurHash64A(&key, 8, seed[5]));
}

// Returns the position of the rank'th 1.  (rank = 0 returns the 1st 1)
// Returns 64 if there are fewer than rank+1 1s.
static inline uint64_t word_select(uint64_t val, int rank) {
	val = _pdep_u64(one[rank], val);
	return _tzcnt_u64(val);
}

iceberg_table * iceberg_init(uint64_t log_slots) {

	iceberg_table * table;

	uint64_t total_blocks = 1 << (log_slots - SLOT_BITS);
  	uint64_t total_size_in_bytes = (sizeof(iceberg_lv1_block) + sizeof(iceberg_lv2_block) + sizeof(iceberg_lv1_block_md) + sizeof(iceberg_lv2_block_md)) * total_blocks;
	
	table = (iceberg_table *)malloc(sizeof(iceberg_table));
	assert(table);

	table->level1 = (iceberg_lv1_block *)malloc(sizeof(iceberg_lv1_block) * total_blocks);
	table->level2 = (iceberg_lv2_block *)malloc(sizeof(iceberg_lv2_block) * total_blocks);
	table->level3 = (iceberg_lv3 *)malloc(sizeof(iceberg_lv3));

	table->metadata = (iceberg_metadata *)malloc(sizeof(iceberg_metadata));
	table->metadata->total_size_in_bytes = total_size_in_bytes;
	table->metadata->nslots = 1 << log_slots;
	table->metadata->nelts = 0;
	table->metadata->nblocks = total_blocks;
	table->metadata->block_bits = log_slots - SLOT_BITS;
	table->metadata->lv2_balls = 0;
	table->metadata->lv3_bits = 0;
	table->metadata->lv3_balls = 0;
	table->metadata->lv3_pslsum = 0;

	table->metadata->lv1_md = (iceberg_lv1_block_md *)malloc(sizeof(iceberg_lv1_block_md) * total_blocks);
	table->metadata->lv2_md = (iceberg_lv2_block_md *)malloc(sizeof(iceberg_lv2_block_md) * total_blocks);
	table->metadata->lv3_md = (uint64_t *)malloc(sizeof(uint64_t));

	for (uint64_t i = 0; i < total_blocks; ++i) {
		for (uint64_t j = 0; j < (1 << SLOT_BITS); ++j)
			table->metadata->lv1_md[i].block_md[j] = 0;
		for (uint64_t j = 0; j < C_LV2 + MAX_LG_LG_N / D_CHOICES; ++j)
			table->metadata->lv2_md[i].block_md[j] = 0;
	}
	table->metadata->lv3_md[0] = 0;

	return table;
}

uint64_t total_capacity(iceberg_table * restrict table) {
	return (1 << table->metadata->lv3_bits) + table->metadata->nblocks * ((1 << SLOT_BITS) + C_LV2 + MAX_LG_LG_N / D_CHOICES);
}

double iceberg_load_factor(iceberg_table * restrict table) {
	return (double)table->metadata->nelts / (double)total_capacity(table);
}

void split_hash(uint64_t hash, uint8_t& fprint, uint64_t& index, uint64_t& tag, iceberg_metadata * metadata) {	
	fprint = hash & ((1 << FPRINT_BITS) - 1);
	index = (hash >> FPRINT_BITS) & ((1 << metadata->block_bits) - 1);
	tag = hash >> (metadata->block_bits + FPRINT_BITS);
}

__mmask32 slot_mask_32(uint8_t * metadata, uint8_t fprint) {
	__m256i bcast = _mm256_set1_epi8(fprint);
	__m256i block = _mm256_loadu_si256(reinterpret_cast<__m256i*>(metadata));
	return _mm256_cmp_epi8_mask(bcast, block, _MM_CMPINT_EQ);
}

__mmask64 slot_mask_64(uint8_t * metadata, uint8_t fprint) {
	__m512i bcast = _mm512_set1_epi8(fprint);
	__m512i block = _mm512_loadu_si512(reinterpret_cast<__m512i*>(metadata));
	return _mm512_cmp_epi8_mask(bcast, block, _MM_CMPINT_EQ);
}

void expand_lv3(iceberg_table * restrict table) {
	
	table->metadata->lv3_bits++;
	uint8_t lv3_size = (1 << table->metadata->lv3_bits);


}

bool iceberg_lv3_insert(iceberg_table * restrict table, KeyType key, ValueType value) {

	table->metadata->lv3_balls++;
/*
	iceberg_metadata * restrict metadata = table->metadata;
	iceberg_lv3 * restrict lv3 = table->level3;
	
	if(metadata->lv3_balls >= LV3_MAX_LF * (1 << metadata->lv3_bits)) expand_lv3(table);

	uint64_t hash = lv3_hash(key);
	uint8_t fprint = hash & ((1 << FPRINT_BITS) - 1);
	uint64_t index = (hash >> FPRINT_BITS) & ((1 << LV3_BITS) - 1);
	uint64_t tag = hash >> (LV3_BITS + FPRINT_BITS);

	uint64_t psl = 0;

	while (1) {
		
		if(!lv3->lv3_md[index]) {
			lv3->lv3_md[index] = fprint;
			lv3->vals[index] = value;
			lv3->tags[index] = tag;
			lv3->psl[index] = psl;
			metadata->lv3_balls++;
//			metadata->lv3_pslsum += psl;
			return true;
		}
		
		if(psl > lv3->psl[index]) {
			uint8_t temp_fprint = fprint;
			fprint = lv3->lv3_md[index];
			lv3->lv3_md[index] = temp_fprint;

			uint64_t temp_psl = psl;
			psl = lv3->psl[index];
			lv3->psl[index] = temp_psl;

			ValueType temp_val = value;
			value = lv3->vals[index];
			lv3->vals[index] = temp_val;
			
			uint64_t temp_tag = tag;
			tag = lv3->tags[index];
			lv3->tags[index] = temp_tag;
		}

		index = (index + 1) % (1 << LV3_BITS);
		psl++;
	}

*/	return false;
}

bool iceberg_lv2_insert(iceberg_table * restrict table, KeyType key, ValueType value) {

	iceberg_metadata * restrict metadata = table->metadata;
	iceberg_lv2_block * restrict blocks = table->level2;

	if(metadata->lv2_balls == C_LV2 * metadata->nblocks) return iceberg_lv3_insert(table, key, value);

	uint8_t best_fprint, most_spots = 0, slot;
	uint64_t best_index, best_tag;

	for(uint8_t i = 0; i < D_CHOICES; ++i) {
		
		uint8_t fprint;
		uint64_t index, tag;

		split_hash(lv2_hash(key, i), fprint, index, tag, metadata);

		__mmask32 md_mask = slot_mask_32(metadata->lv2_md[index].block_md, 0) & ((1 << (C_LV2 + MAX_LG_LG_N / D_CHOICES)) - 1);
		uint8_t spots = __builtin_popcount(md_mask);

		if(md_mask && (spots > most_spots || (spots == most_spots && index < best_index))) {
			best_fprint = fprint;
			best_index = index;
			best_tag = tag;
			most_spots = spots;
			slot = word_select(md_mask, 0);
		}
	}

	if(most_spots) {
		
		metadata->lv2_md[best_index].block_md[slot] = best_fprint;
		blocks[best_index].tags[slot] = best_tag;
		blocks[best_index].vals[slot] = value;
		metadata->nelts++;
		metadata->lv2_balls++;
		
		return true;
	}
	
	return iceberg_lv3_insert(table, key, value);
}

bool iceberg_insert(iceberg_table * restrict table, KeyType key, ValueType value) {
	
	iceberg_metadata * restrict metadata = table->metadata;
	iceberg_lv1_block * restrict blocks = table->level1;	
	
	uint8_t fprint;
	uint64_t index, tag;
	
	split_hash(lv1_hash(key), fprint, index, tag, metadata);
	
	__mmask64 md_mask = slot_mask_64(metadata->lv1_md[index].block_md, 0);
	
	if(!md_mask) return iceberg_lv2_insert(table, key, value);
	
	uint8_t slot = word_select(md_mask, 0);
	
	metadata->lv1_md[index].block_md[slot] = fprint;
	blocks[index].tags[slot] = tag;
	blocks[index].vals[slot] = value;
	metadata->nelts++;

	return true;
}

bool iceberg_lv2_remove(iceberg_table * restrict table, KeyType key, ValueType value) {

	iceberg_metadata * restrict metadata = table->metadata;
	iceberg_lv2_block * restrict blocks = table->level2;
	
	for(int i = 0; i < D_CHOICES; ++i) {
		
		uint8_t fprint;
		uint64_t index, tag;

		split_hash(lv2_hash(key, i), fprint, index, tag, metadata);

		__mmask32 md_mask = slot_mask_32(metadata->lv2_md[index].block_md, fprint) & ((1 << (C_LV2 + MAX_LG_LG_N / D_CHOICES)) - 1);
		uint8_t popct = __builtin_popcount(md_mask);

		for(uint8_t i = 0; i < popct; ++i) {
			
			uint8_t slot = word_select(md_mask, i);

			if(blocks[index].tags[slot] == tag && blocks[index].vals[slot] == value) {
				metadata->lv2_md[index].block_md[slot] = 0;
				metadata->nelts--;
				metadata->lv2_balls--;
				
				return true;
			}
		}
	}

	return false;
}

bool iceberg_remove(iceberg_table * restrict table, KeyType key, ValueType value) {

	iceberg_metadata * restrict metadata = table->metadata;
	iceberg_lv1_block * restrict blocks = table->level1;
	
	uint8_t fprint;
	uint64_t index, tag;
	
	split_hash(lv1_hash(key), fprint, index, tag, metadata);
	
	__mmask64 md_mask = slot_mask_64(metadata->lv1_md[index].block_md, fprint);
	uint8_t popct = __builtin_popcountll(md_mask);
	
	for(uint8_t i = 0; i < popct; ++i) {
		
		uint8_t slot = word_select(md_mask, i);

		if(blocks[index].tags[slot] == tag && blocks[index].vals[slot] == value) {
			metadata->lv1_md[index].block_md[slot] = 0;
			metadata->nelts--;

			return true;
		}
	}

	return iceberg_lv2_remove(table, key, value);
}

bool iceberg_lv2_get_value(iceberg_table * restrict table, KeyType key, ValueType& value) {
	
	iceberg_metadata * restrict metadata = table->metadata;
	iceberg_lv2_block * restrict blocks = table->level2;

	for(uint8_t i = 0; i < D_CHOICES; ++i) {
		
		uint8_t fprint;
		uint64_t index, tag;

		split_hash(lv2_hash(key, i), fprint, index, tag, metadata);

		__mmask32 md_mask = slot_mask_32(metadata->lv2_md[index].block_md, fprint) & ((1 << (C_LV2 + MAX_LG_LG_N / D_CHOICES)) - 1);
		uint8_t popct = __builtin_popcount(md_mask);

		for(uint8_t i = 0; i < popct; ++i) {

			uint8_t slot = word_select(md_mask, i);

			if(blocks[index].tags[slot] == tag) {
				value = blocks[index].vals[slot];
				return true;
			}
		}
	}
	
	return false;
}

bool iceberg_get_value(iceberg_table * restrict table, KeyType key, ValueType& value) {

	iceberg_metadata * restrict metadata = table->metadata;
	iceberg_lv1_block * restrict blocks = table->level1;
	
	uint8_t fprint;
	uint64_t index, tag;
	
	split_hash(lv1_hash(key), fprint, index, tag, metadata);
	
	__mmask64 md_mask = slot_mask_64(metadata->lv1_md[index].block_md, fprint);
	
	uint8_t popct = __builtin_popcountll(md_mask);

	for(uint8_t i = 0; i < popct; ++i) {

		uint8_t slot = word_select(md_mask, i);

		if(blocks[index].tags[slot] == tag) {
			value = blocks[index].vals[slot];
			return true;
		}
	}

	return iceberg_lv2_get_value(table, key, value);
}

