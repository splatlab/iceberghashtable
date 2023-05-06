#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>

#define likely(x)   __builtin_expect((x), 1)
#define unlikely(x) __builtin_expect((x), 0)

#if __linux__
#  include <linux/version.h>
#  if LINUX_VERSION_CODE > KERNEL_VERSION(2, 6, 22)
#    define _MAP_POPULATE_AVAILABLE
#  endif
#endif

#ifdef _MAP_POPULATE_AVAILABLE
#  ifdef _MAP_HUGETLB_AVAILABLE
#    define MMAP_FLAGS                                                         \
      (MAP_PRIVATE | MAP_ANONYMOUS | MAP_POPULATE | MAP_HUGE_TLB)
#  else // _MAP_HUGETLB_AVAILABLE
#    define MMAP_FLAGS (MAP_PRIVATE | MAP_ANONYMOUS | MAP_POPULATE)
#  endif // _MAP_HUGETLB_AVAILABLE
#else    // _MAP_POPULATE_AVAILABLE
#  define MMAP_FLAGS (MAP_PRIVATE | MAP_ANONYMOUS)
#endif

static inline void *
util_mmap(size_t length)
{
  void *ret = mmap(NULL, length, PROT_READ | PROT_WRITE, MMAP_FLAGS, 0, 0);
  if (ret == MAP_FAILED) {
    perror("mmap failed");
    exit(1);
  }
  return ret;
}

static inline void
util_munmap(volatile void *addr, size_t length)
{
  __attribute__((unused)) int ret = munmap((void *)addr, length);
  assert(ret == 0);
}
