#ifndef _POTC_TABLE_H_
#define _POTC_TABLE_H_

#include <inttypes.h>
#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
#  define __restrict__
extern "C" {
#endif

typedef uint64_t iceberg_key_t;
typedef uint64_t iceberg_value_t;

typedef struct iceberg_table iceberg_table;

int iceberg_create(iceberg_table **table, uint64_t log_slots);

void iceberg_destroy(iceberg_table **table);

bool iceberg_insert(iceberg_table  *table,
                    iceberg_key_t   key,
                    iceberg_value_t value,
                    uint64_t        tid);

bool iceberg_delete(iceberg_table *table, iceberg_key_t key, uint64_t tid);

bool iceberg_query(iceberg_table   *table,
                   iceberg_key_t    key,
                   iceberg_value_t *value,
                   uint64_t         tid);


uint64_t level1_load(iceberg_table *table);
uint64_t level2_load(iceberg_table *table);
uint64_t level3_load(iceberg_table *table);
uint64_t iceberg_load(iceberg_table *table);

uint64_t iceberg_capacity(iceberg_table *table);
double   iceberg_load_factor(iceberg_table *table);

#ifdef ENABLE_RESIZE
void iceberg_end(iceberg_table *table, uint64_t tid);
#endif

bool iceberg_scan_for_key(iceberg_table *t, iceberg_key_t key);

#ifdef __cplusplus
}
#endif

#endif // _POTC_TABLE_H_
