#ifndef _POTC_TABLE_H_
#define _POTC_TABLE_H_

#include <inttypes.h>
#include <stdbool.h>

#ifdef __cplusplus
#define restrict __restrict__
extern "C" {
#endif

	#define BUCKETS_PER_BLOCK 32
	#define BITS_PER_BUCKET 5

	typedef uint64_t KeyType;
	typedef uint64_t ValueType;

	typedef struct iceberg_block {
		uint64_t tags[BUCKETS_PER_BLOCK];
		ValueType vals[BUCKETS_PER_BLOCK];
	} iceberg_block;

	typedef struct iceberg_metadata {
		uint64_t total_size_in_bytes;
		uint64_t nblocks;
		uint64_t nelts;
		uint64_t nslots;
		uint64_t block_bits;
		uint64_t block_metadata[];
	} iceberg_metadata;

	typedef struct iceberg_table {
		iceberg_metadata * metadata;
		iceberg_block blocks[];
	} iceberg_table;

	iceberg_table * iceberg_init(uint64_t nslots);

	bool iceberg_insert(iceberg_table * restrict table, KeyType key, ValueType value);

	bool iceberg_remove(iceberg_table * restrict table, KeyType key, ValueType value);

	bool iceberg_get_value(iceberg_table * restrict table, KeyType key, ValueType& value);

#ifdef __cplusplus
}
#endif

#endif	// _POTC_TABLE_H_

