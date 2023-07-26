#pragma once

#include "util.h"
#include <inttypes.h>

#define NUM_COUNTERS 64
#define THRESHOLD    1024

typedef __attribute__((aligned(64))) struct {
  volatile int64_t count[8];
} local_counter;

typedef __attribute__((aligned(64))) struct counter {
  int64_t       global[8];
  local_counter local_counters[NUM_COUNTERS];
} counter;

static inline void
counter_init(counter *cntr)
{
  memset(cntr, 0, sizeof(*cntr));
}

static inline void
counter_increment(counter *cntr, uint64_t batch, uint64_t tid)
{
  int64_t local = ++cntr->local_counters[tid].count[batch];
  if (local == THRESHOLD) {
    cntr->local_counters[tid].count[batch] = 0;
    __atomic_fetch_add(&cntr->global[batch], local, __ATOMIC_SEQ_CST);
  }
}

static inline void
counter_decrement(counter *cntr, uint64_t batch, uint64_t tid)
{
  int64_t local = --cntr->local_counters[tid].count[batch];
  if (local == -THRESHOLD) {
    cntr->local_counters[tid].count[batch] = 0;
    __atomic_fetch_add(&cntr->global[batch], local, __ATOMIC_SEQ_CST);
  }
}

static inline int64_t
counter_get_local(counter *cntr, uint64_t batch, uint64_t tid)
{
  int64_t local = cntr->local_counters[tid].count[batch];
  if (local < 0) {
    local += THRESHOLD;
  }
  return local;
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

static inline int64_t
counter_local_sum(counter *cntr, uint64_t batch_start, uint64_t batch_end, uint64_t tid)
{
  int64_t sum = 0;
  for (uint64_t i = batch_start; i < batch_end; i++) {
    sum += cntr->local_counters[tid].count[i];
  }
  while (sum < 0) {
    sum += THRESHOLD;
  }
  sum %= THRESHOLD;

  return sum;
}
