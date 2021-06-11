/*
 * ============================================================================
 *
 *         Author:  Prashant Pandey (), ppandey@cs.stonybrook.edu
 *   Organization:  Stony Brook University
 *
 * ============================================================================
 */

#if !defined(_GNU_SOURCE)
#define _GNU_SOURCE
#endif

#include "lock.h"

void rw_lock_init(ReaderWriterLock *rwlock) {
  rwlock->readers = 0;
  rwlock->writer = 0;
  rwlock->pc_counter = (pc_t *)malloc(sizeof(pc_t));
  pc_init(rwlock->pc_counter, &rwlock->readers, 8, 8);
}

/**
 * Try to acquire a lock and spin until the lock is available.
 */
bool read_lock(ReaderWriterLock *rwlock, uint8_t thread_id) {
  while (rwlock->writer != 0);
  pc_add(rwlock->pc_counter, 1, thread_id);
  return true;

  return false;
}

void read_unlock(ReaderWriterLock *rwlock, uint8_t thread_id) {
  pc_add(rwlock->pc_counter, -1, thread_id);
  return;
}

/**
 * Try to acquire a write lock and spin until the lock is available.
 * Then wait till reader count is 0.
 */
bool write_lock(ReaderWriterLock *rwlock) {
  // try to acquire write lock.
  if (__sync_lock_test_and_set(&rwlock->writer, 1))
    return false;
  // wait for readers to finish
  do {
    pc_sync(rwlock->pc_counter);
  } while(rwlock->readers);

  return true;
}

void write_unlock(ReaderWriterLock *rwlock) {
  __sync_lock_release(&rwlock->writer);
  return;
}


