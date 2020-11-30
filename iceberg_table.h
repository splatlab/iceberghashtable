#ifndef _POTC_TABLE_H_
#define _POTC_TABLE_H_

#include <inttypes.h>
#include <stdbool.h>

#ifdef __cplusplus
#define restrict __restrict__
extern "C" {
#endif

	#define SLOT_BITS 6
	#define FPRINT_BITS 8
	#define D_CHOICES 2
	#define MAX_LG_LG_N 4
	#define C_LV2 6

	typedef uint64_t KeyType;
	typedef uint64_t ValueType;

	typedef struct __attribute__ ((__packed__)) iceberg_lv1_block {
		uint64_t keys[1 << SLOT_BITS];
		ValueType vals[1 << SLOT_BITS];
	} iceberg_lv1_block;

	typedef struct __attribute__ ((__packed__)) iceberg_lv1_block_md {
		uint8_t block_md[1 << SLOT_BITS];
	} iceberg_lv1_block_md;

	typedef struct __attribute__ ((__packed__)) iceberg_lv2_block {
		uint64_t keys[C_LV2 + MAX_LG_LG_N / D_CHOICES];
		ValueType vals[C_LV2 + MAX_LG_LG_N / D_CHOICES];
	} iceberg_lv2_block;

	typedef struct __attribute__ ((__packed__)) iceberg_lv2_block_md {
		uint8_t block_md[C_LV2 + MAX_LG_LG_N / D_CHOICES];
	} iceberg_lv2_block_md;

	typedef struct iceberg_lv3_node {
		KeyType key;
		ValueType val;
		iceberg_lv3_node * next_node;
	} iceberg_lv3_node;

	typedef struct iceberg_lv3_list {
		iceberg_lv3_node * head;
		iceberg_lv3_node * tail;
	} iceberg_lv3_list;

	typedef struct iceberg_metadata {
		uint64_t total_size_in_bytes;
		uint64_t nblocks;
		uint64_t nslots;
		uint64_t block_bits;
		uint64_t total_balls;
		uint64_t lv2_balls;
		uint64_t lv3_balls;
		iceberg_lv1_block_md * lv1_md;
		iceberg_lv2_block_md * lv2_md;
		uint64_t * lv3_sizes;
	} iceberg_metadata;

	typedef struct iceberg_table {
		iceberg_metadata * metadata;
		iceberg_lv1_block * level1;
		iceberg_lv2_block * level2;
		iceberg_lv3_list * level3;
	} iceberg_table;

	iceberg_table * iceberg_init(uint64_t nslots);

	double iceberg_load_factor(iceberg_table * restrict table);

	bool iceberg_insert(iceberg_table * restrict table, KeyType key, ValueType value);

	bool iceberg_remove(iceberg_table * restrict table, KeyType key, ValueType value);

	bool iceberg_get_value(iceberg_table * restrict table, KeyType key, ValueType& value);

#ifdef __cplusplus
}
#endif

#endif	// _POTC_TABLE_H_

