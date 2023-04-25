#pragma once

#include <stdbool.h>

/*
 * Try to acquire a lock once and return even if the lock is busy.
 */
static inline bool
lock(volatile bool *lk)
{
  return !__atomic_test_and_set(lk, __ATOMIC_SEQ_CST);
}

void
unlock(volatile bool *lk)
{
  __atomic_clear(lk, __ATOMIC_SEQ_CST);
  return;
}
