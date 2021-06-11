/*
 * ============================================================================
 *
 *         Author:  Prashant Pandey (), ppandey@cs.stonybrook.edu
 *   Organization:  Stony Brook University
 *
 * ============================================================================
 */

#ifndef _LOCK_H_
#define _LOCK_H_

#include <stdlib.h>
#include <inttypes.h>
#include <stdio.h>
#include <unistd.h>

#include "partitioned_counter.h"

#ifdef __cplusplus
#define __restrict__
extern "C" {
#endif

#define NO_LOCK (0x01)
#define TRY_ONCE_LOCK (0x02)
#define WAIT_FOR_LOCK (0x04)

#define GET_NO_LOCK(flag) (flag & NO_LOCK)
#define GET_TRY_ONCE_LOCK(flag) (flag & TRY_ONCE_LOCK)
#define GET_WAIT_FOR_LOCK(flag) (flag & WAIT_FOR_LOCK)


  typedef struct ReaderWriterLock {
    int64_t readers;
    volatile int writer;
    pc_t * pc_counter;
  } ReaderWriterLock;

  void rw_lock_init(ReaderWriterLock *rwlock) {
    rwlock->readers = 0;
    rwlock->writer = 0;
    rwlock->pc_counter = (pc_t *)malloc(sizeof(pc_t));
    pc_init(rwlock->pc_counter, &rwlock->readers, 8, 8);
  }

  /**
   * Try to acquire a lock and spin until the lock is available.
   */
  bool read_lock(ReaderWriterLock *rwlock, uint8_t flag, uint8_t thread_id) {
    if (GET_WAIT_FOR_LOCK(flag) != WAIT_FOR_LOCK) {
      if (rwlock->writer == 0) {
        pc_add(rwlock->pc_counter, 1, thread_id);
        return true;
      }
    } else {
      while (rwlock->writer != 0);
      pc_add(rwlock->pc_counter, 1, thread_id);
      return true;
    }

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
  bool write_lock(ReaderWriterLock *rwlock, uint8_t flag) {
    // acquire write lock.
    if (GET_WAIT_FOR_LOCK(flag) != WAIT_FOR_LOCK) {
      if (__sync_lock_test_and_set(&rwlock->writer, 1))
        return false;
    } else {
      while (__sync_lock_test_and_set(&rwlock->writer, 1))
        while (rwlock->writer != 0);
    }
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

#ifdef __cplusplus
}
#endif



#endif
