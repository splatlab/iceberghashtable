#pragma once

#include "util.h"
#include <inttypes.h>

#define NUM_COUNTERS 64
#define THRESHOLD    1024

typedef __attribute__((aligned(64))) struct {
  volatile int64_t count[8];
} local_counter;

typedef __attribute__((aligned(64))) struct counter {
  int64_t        global[8];
  local_counter *local_counters;
} counter;

static inline void
counter_init(counter *cntr)
{
  size_t local_counter_bytes = NUM_COUNTERS * sizeof(local_counter);
  cntr->local_counters       = util_mmap(local_counter_bytes);
}

static inline void
counter_increment(counter *cntr, uint64_t batch, uint8_t tid)
{
  int64_t local = ++cntr->local_counters[tid].count[batch];
  if (local == THRESHOLD) {
    cntr->local_counters[tid].count[batch] = 0;
    __atomic_fetch_add(&cntr->global[batch], local, __ATOMIC_SEQ_CST);
  }
}

static inline void
counter_decrement(counter *cntr, uint64_t batch, uint8_t tid)
{
  int64_t local = --cntr->local_counters[tid].count[batch];
  if (local == -THRESHOLD) {
    cntr->local_counters[tid].count[batch] = 0;
    __atomic_fetch_add(&cntr->global[batch], local, __ATOMIC_SEQ_CST);
  }
}

/*
 * NOT THREAD SAFE
 */
static inline void
counter_sync(counter *cntr, uint64_t batch)
{
  for (uint32_t tid = 0; tid < NUM_COUNTERS; tid++) {
    int64_t local_count = cntr->local_counters[tid].count[batch];
    cntr->local_counters[tid].count[batch] = 0;
    __atomic_fetch_add(&cntr->global[batch], local_count, __ATOMIC_SEQ_CST);
  }
}

static inline uint64_t
counter_get(counter *cntr, uint64_t batch)
{
  return cntr->global[batch];
}
