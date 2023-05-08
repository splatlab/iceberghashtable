#include <assert.h>
#include <errno.h>
#include <stdarg.h>
#include <stdatomic.h>
#include <stdio.h>
#include <string.h>
#include <threads.h>

#include "iceberg_table.h"

#define TEST_LOG_SLOTS 20

#define TEST_OP_NAME_LENGTH 26

#define BATCH_SIZE 1024

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
  size_t len = strnlen(test_name, TEST_OP_NAME_LENGTH);
  assert(len <= TEST_OP_NAME_LENGTH);
  printf("Starting test %s...%*s", test_name, (int)(TEST_OP_NAME_LENGTH - len), "");
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
    print_fail_message("iceberg_insert failed to insert key: 0x%" PRIx64, key);
    goto out;
  }

  // Query for the kv pair, should be found
  iceberg_value_t returned_value;
  bool            found = iceberg_query(table, key, &returned_value, 0);
  if (!found) {
    print_fail_message("iceberg_query failed to find inserted key: 0x%" PRIx64,
                       key);
    goto out;
  }
  if (returned_value != value) {
    print_fail_message("iceberg_query returned incorrect value: 0x%" PRIx64
                       ", expected %" PRIx64,
                       returned_value,
                       value);
    goto out;
  }

  // Query for a different key, should not be found
  iceberg_key_t another_key = 2ULL;
  found = iceberg_query(table, another_key, &returned_value, 0);
  if (found) {
    print_fail_message("iceberg_query found a non-inserted key: 0x%" PRIx64,
                       another_key);
    goto out;
  }

  // Try to reinsert the key, should fail
  inserted = iceberg_insert(table, key, value, 0);
  if (inserted) {
    print_fail_message("iceberg_insert overwrote inserted key: 0x%" PRIx64,
                       key);
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
    print_fail_message("iceberg_insert failed to delete key: 0x%" PRIx64, key);
    goto out;
  }

  // Query for the key, should not be fouund
  found = iceberg_query(table, key, &returned_value, 0);
  if (found) {
    print_fail_message("iceberg_query found the deleted key: 0x%" PRIx64, key);
    goto out;
  }

  print_success();
out:
  close(&table);
}

bool
insert_keys_in_range(iceberg_table *table,
                     iceberg_key_t  start_key,
                     iceberg_key_t  end_key,
                     uint64_t       tid)
{
  for (iceberg_key_t key = start_key; key < end_key; key++) {
    iceberg_value_t value    = key;
    bool            inserted = iceberg_insert(table, key, value, tid);
    if (!inserted) {
      print_fail_message("iceberg_insert failed to insert key: 0x%" PRIx64,
                         key);
      return false;
    }
  }
  return true;
}

bool
query_keys_in_range(iceberg_table *table,
                    iceberg_key_t  start_key,
                    iceberg_key_t  end_key,
                    bool           expect_found,
                    uint64_t       tid)
{
  for (iceberg_key_t key = start_key; key < end_key; key++) {
    iceberg_value_t value;
    bool            found = iceberg_query(table, key, &value, tid);
    if (found != expect_found) {
      if (expect_found) {
        print_fail_message("iceberg_query failed to find key: 0x%" PRIx64, key);
      } else {
        print_fail_message("iceberg_query found unexpected key: 0x%" PRIx64,
                           key);
      }
      return false;
    }
  }
  return true;
}

void
run_resize()
{
  print_start_message("Resize");
  iceberg_table *table;
  uint64_t       initial_capacity;
  int            rc = open(&table, &initial_capacity);
  if (rc) {
    print_fail_message(
      "iceberg_create failed with error: %d -- %s", rc, strerror(rc));
    return;
  }

  // Insert keys so that the initial capacity is filled
  iceberg_key_t start_key = 1;
  iceberg_key_t end_key   = initial_capacity + 1;
  if (!insert_keys_in_range(table, start_key, end_key, 0)) {
    goto out;
  }

  // Check that all keys are found
  if (!query_keys_in_range(table, start_key, end_key, true, 0)) {
    goto out;
  }

  // Check that a resize hasn't happened, i.e. capacity is the same
  uint64_t capacity = iceberg_capacity(table);
  if (initial_capacity != capacity) {
    print_fail_message("Premature resize after %" PRIu64 " insertions",
                       initial_capacity);
    goto out;
  }

  uint64_t load = iceberg_load(table);
  if (load != initial_capacity) {
    print_fail_message("Unexpected load reported: %" PRIu64
                       ", expected %" PRIu64,
                       load,
                       initial_capacity);
    goto out;
  }

  // Insert one more key, should cause a resize
  iceberg_key_t one_more_key = end_key;
  end_key                    = end_key + 1;
  if (!insert_keys_in_range(table, one_more_key, end_key, 0)) {
    goto out;
  }

  load = iceberg_load(table);
  if (load != initial_capacity + 1) {
    print_fail_message("Unexpected load reported: %" PRIu64
                       ", expected %" PRIu64,
                       load,
                       initial_capacity + 1);
    goto out;
  }

  // Check that the resized occured, i.e. capacity has doubled
  capacity = iceberg_capacity(table);
  if (capacity != initial_capacity * 2) {
    print_fail_message("Unexpected initial_capacity after resize: %" PRIu64
                       ", expected %" PRIu64,
                       capacity,
                       initial_capacity);
    goto out;
  }

  // Check that all keys are found
  if (!query_keys_in_range(table, start_key, end_key, true, 0)) {
    goto out;
  }

  print_success();
out:
  close(&table);
}

typedef struct {
  iceberg_table             *table;
  volatile _Atomic uint64_t *key_batch_num;
  uint64_t                   num_batches;
  uint64_t                   batch_size;
  uint64_t                   tid;
  bool                       expect_found;
  volatile bool             *stop;
} thread_params;

int
insert_thread(void *arg)
{
  thread_params *params = arg;

  uint64_t batch_num = atomic_fetch_add(params->key_batch_num, 1);
  while (batch_num < params->num_batches) {
    iceberg_key_t start_key = batch_num * params->batch_size + 1;
    iceberg_key_t end_key   = start_key + params->batch_size;
    bool          succeeded =
      insert_keys_in_range(params->table, start_key, end_key, params->tid);
    if (!succeeded) {
      return -1;
    }
    batch_num = atomic_fetch_add(params->key_batch_num, 1);
  }

  return 0;
}

int
query_thread(void *arg)
{
  thread_params *params = arg;

  do {
    uint64_t batch_num = atomic_load(params->key_batch_num);
    if (batch_num != 0) {
      batch_num--;
      iceberg_key_t start_key = batch_num * params->batch_size + 1;
      iceberg_key_t end_key   = start_key + params->batch_size;
      bool          succeeded = query_keys_in_range(
        params->table, start_key, end_key, params->expect_found, params->tid);
      if (!succeeded) {
        return -1;
      }
      batch_num = atomic_load(params->key_batch_num);
    }
  } while (!*params->stop);

  return 0;
}

void
run_multithreaded_inserts(uint64_t num_threads)
{
  char op_name[128];
  snprintf(op_name, 128, "Multithreaded Inserts (%" PRIu64 ")", num_threads);
  print_start_message(op_name);
  iceberg_table *table;
  uint64_t       initial_capacity;
  int            rc = open(&table, &initial_capacity);
  if (rc) {
    print_fail_message(
      "iceberg_create failed with error: %d -- %s", rc, strerror(rc));
    return;
  }

  uint64_t      num_batches = initial_capacity / BATCH_SIZE;
  uint64_t      num_inserts = num_batches * BATCH_SIZE;
  thread_params params[num_threads];
  thrd_t        threads[num_threads];
  volatile _Atomic uint64_t batch_num = 0;
  volatile bool stop = false;
  for (uint64_t i = 0; i < num_threads; i++) {
    params[i].table         = table;
    params[i].key_batch_num = &batch_num;
    params[i].num_batches   = num_batches;
    params[i].batch_size    = BATCH_SIZE;
    params[i].tid           = i;
    params[i].expect_found  = false;
    params[i].stop          = &stop;
    int rc = thrd_create(&threads[i], insert_thread, &params[i]);
    assert(rc == thrd_success);
  }

  for (uint64_t i = 0; i < num_threads; i++) {
    int res;
    int rc = thrd_join(threads[i], &res);
    assert(rc == thrd_success);
  }

  uint64_t load = iceberg_load(table);
  if (load != num_inserts) {
    print_fail_message("Unexpected load reported: %" PRIu64
                       ", expected %" PRIu64,
                       load,
                       num_inserts);
    goto out;
  }

  if (!query_keys_in_range(table, 1, num_inserts + 1, true, 0)) {
    goto out;
  }

  print_success();
out:
  close(&table);
}

int
main(int argc, char *argv[])
{
  run_open_close();

  run_basic();

  run_resize();

  run_multithreaded_inserts(2);
  run_multithreaded_inserts(4);
  run_multithreaded_inserts(8);
}
