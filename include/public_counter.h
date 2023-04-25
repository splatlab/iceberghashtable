#pragma once

#include <stdint.h>

typedef __attribute__((aligned(64))) struct {
  volatile int64_t count[8];
} local_counter;

typedef __attribute__((aligned(64))) struct counter {
  int64_t        global[8];
  local_counter *local_counters;
} counter;
