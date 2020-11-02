#ifndef _POTC_TABLE_H_
#define _POTC_TABLE_H_

#include <inttypes.h>
#include <stdbool.h>

#ifdef __cplusplus
#define restrict __restrict__
extern "C" {
#endif

	#define SLOT_BITS 5
	#define FPRINT_BITS 8
	#define D_CHOICES 4
	#define MAX_LG_LG_N 8
	#define C_LV2 2

	typedef uint64_t KeyType;
	typedef uint64_t ValueType;

	typedef struct __attribute__ ((__packed__)) iceberg_lv1_block {
		uint64_t tags[1 << SLOT_BITS];
		ValueType vals[1 << SLOT_BITS];
	} iceberg_lv1_block;

	typedef struct __attribute__ ((__packed__)) iceberg_lv1_block_md {
		uint8_t block_md[1 << SLOT_BITS];
	} iceberg_lv1_block_md;

	typedef struct __attribute__ ((__packed__)) iceberg_lv2_block {
		uint64_t tags[C_LV2 + MAX_LG_LG_N / D_CHOICES];
		ValueType vals[C_LV2 + MAX_LG_LG_N / D_CHOICES];
	} iceberg_lv2_block;

	typedef struct __attribute__ ((__packed__)) iceberg_lv2_block_md {
		uint8_t block_md[C_LV2 + MAX_LG_LG_N / D_CHOICES];
	} iceberg_lv2_block_md;

	typedef struct iceberg_metadata {
		uint64_t total_size_in_bytes;
		uint64_t nblocks;
		uint64_t nslots;
		uint64_t block_bits;
		uint64_t lv2_balls;
		iceberg_lv1_block_md * lv1_md;
		iceberg_lv2_block_md * lv2_md;
	} iceberg_metadata;

	typedef struct iceberg_table {
		iceberg_metadata * metadata;
		iceberg_lv1_block * level1;
		iceberg_lv2_block * level2;
	} iceberg_table;

	iceberg_table * iceberg_init(uint64_t nslots);

	bool iceberg_insert(iceberg_table * restrict table, KeyType key, ValueType value);

	bool iceberg_remove(iceberg_table * restrict table, KeyType key, ValueType value);

	bool iceberg_get_value(iceberg_table * restrict table, KeyType key, ValueType& value);

#ifdef __cplusplus
}
#endif

#endif	// _POTC_TABLE_H_

