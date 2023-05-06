#include <assert.h>
#include <errno.h>
#include <stdarg.h>
#include <stdio.h>
#include <string.h>

#include "iceberg_table.h"

#define TEST_LOG_SLOTS 20

int
open(iceberg_table **table, uint64_t *capacity)
{
  int rc    = iceberg_create(table, TEST_LOG_SLOTS);
  *capacity = iceberg_capacity(*table);
  return rc;
}

void
close(iceberg_table **table)
{
  iceberg_destroy(table);
}

void
print_start_message(const char *test_name)
{
  size_t len = strnlen(test_name, 24);
  assert(len <= 24);
  printf("Starting test %s...%*s", test_name, (int)(24 - len), "");
}

void
print_fail_message(const char *fail_format, ...)
{
  printf("FAILED: ");
  va_list args;
  va_start(args, fail_format);
  vprintf(fail_format, args);
  va_end(args);
  printf("\n");
}

void
print_success()
{
  printf("SUCCEEDED.\n");
}

void
run_open_close()
{
  print_start_message("Open and Close");
  iceberg_table *table;
  uint64_t       capacity;
  int            rc = open(&table, &capacity);
  if (rc) {
    print_fail_message(
      "iceberg_create failed with error: %d -- %s", rc, strerror(rc));
    return;
  }
  close(&table);
  print_success();
}

void
run_basic()
{
  print_start_message("Basic Operations");
  iceberg_table *table;
  uint64_t       capacity;
  int            rc = open(&table, &capacity);
  if (rc) {
    print_fail_message(
      "iceberg_create failed with error: %d -- %s", rc, strerror(rc));
    return;
  }

  // Insert a kv pair
  iceberg_key_t   key      = 1ULL;
  iceberg_value_t value    = 1ULL;
  bool            inserted = iceberg_insert(table, key, value, 0);
  if (!inserted) {
    print_fail_message("iceberg_insert failed to insert key: %" PRIx64, key);
    goto out;
  }

  // Query for the kv pair, should be found
  iceberg_value_t returned_value;
  bool            found = iceberg_query(table, key, &returned_value, 0);
  if (!found) {
    print_fail_message("iceberg_query failed to find inserted key: %" PRIx64,
                       key);
    goto out;
  }
  if (returned_value != value) {
    print_fail_message("iceberg_query returned incorrect value: %" PRIx64
                       ", expected %" PRIx64,
                       returned_value,
                       value);
    goto out;
  }

  // Query for a different key, should not be found
  iceberg_key_t another_key = 2ULL;
  found = iceberg_query(table, another_key, &returned_value, 0);
  if (found) {
    print_fail_message("iceberg_query found a non-inserted key: %" PRIx64,
                       another_key);
    goto out;
  }

  // Try to reinsert the key, should fail
  inserted = iceberg_insert(table, key, value, 0);
  if (!inserted) {
    print_fail_message("iceberg_insert overwrote inserted key: %" PRIx64, key);
    goto out;
  }

  // Check the load
  uint64_t load = iceberg_load(table);
  if (load != 1) {
    print_fail_message(
      "iceberg_load reported incorrect load: %" PRIx64 ", expected 1", load);
    goto out;
  }

  // Delete the key
  bool deleted = iceberg_delete(table, key, 0);
  if (!deleted) {
    print_fail_message("iceberg_insert failed to delete key: %" PRIx64, key);
    goto out;
  }

  // Query for the key, should not be fouund
  found = iceberg_query(table, key, &returned_value, 0);
  if (found) {
    print_fail_message("iceberg_query found the deleted key: %" PRIx64, key);
    goto out;
  }

out:
  close(&table);
  print_success();
}


int
main(int argc, char *argv[])
{
  run_open_close();

  run_basic();
}
