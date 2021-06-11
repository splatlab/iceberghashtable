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
  pc_init(&rwlock->pc_counter, &rwlock->readers, 8, 8);
}

/**
 * Try to acquire a lock and spin until the lock is available.
 */
bool read_lock(ReaderWriterLock *rwlock, uint8_t flag, uint8_t thread_id) {
  __atomic_add_fetch(&rwlock->pc_counter.local_counters[thread_id].counter, 1, __ATOMIC_SEQ_CST);

  if (GET_WAIT_FOR_LOCK(flag) != WAIT_FOR_LOCK) {
    if (rwlock->writer) {
      __atomic_add_fetch(&rwlock->pc_counter.local_counters[thread_id].counter, -1, __ATOMIC_SEQ_CST);
      return false;
    }
    return true;
  }

  while (rwlock->writer) {
    __atomic_add_fetch(&rwlock->pc_counter.local_counters[thread_id].counter, -1, __ATOMIC_SEQ_CST);
    while (rwlock->writer)
      ;
    __atomic_add_fetch(&rwlock->pc_counter.local_counters[thread_id].counter, 1, __ATOMIC_SEQ_CST);
  }

  return true;
}

void read_unlock(ReaderWriterLock *rwlock, uint8_t thread_id) {
  __atomic_add_fetch(&rwlock->pc_counter.local_counters[thread_id].counter, -1, __ATOMIC_SEQ_CST);
  return;
}

/**
 * Try to acquire a write lock and spin until the lock is available.
 * Then wait till reader count is 0.
 */
bool write_lock(ReaderWriterLock *rwlock, uint8_t flag) {
  // acquire write lock.
  if (GET_WAIT_FOR_LOCK(flag) != WAIT_FOR_LOCK) {
    if (__sync_lock_test_and_set(&rwlock->writer, 1))
      return false;
  } else {
    while (__sync_lock_test_and_set(&rwlock->writer, 1))
      while (rwlock->writer != 0)
        ;
  }
  // wait for readers to finish
  for (int i = 0; i < 8; i++)
    while (rwlock->pc_counter.local_counters[i].counter)
      ;

  return true;
}

void write_unlock(ReaderWriterLock *rwlock) {
  __sync_lock_release(&rwlock->writer);
  return;
}


