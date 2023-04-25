#pragma once

#include <stdint.h>
#include <stdio.h>
#include <string.h>

static inline void
verbose_print_hash(uint64_t level1_block,
                   uint64_t level2_block1,
                   uint64_t level2_block2,
                   uint8_t  fp)
{
#ifdef VERBOSE
  printf("hash: level1 block: %8" PRIx64 ", level2 block1: %8" PRIx64
         ", level2 block2: %8" PRIx64 ", fingerprint: %02" PRIx8 "\n",
         level1_block,
         level2_block1,
         level2_block2,
         fp);
#endif
}

static inline void
verbose_print_location(uint64_t level,
                       uint64_t partition,
                       uint64_t block,
                       uint64_t slot,
                       void    *kv)
{
#ifdef VERBOSE
  printf("level: %" PRIx64 ", partition: %" PRIx64 ", block: %8" PRIx64
         ", slot: %2" PRIu64 ", kv: %p\n",
         level,
         partition,
         block,
         slot,
         kv);
#endif
}

static inline void
verbose_print_sketch(uint8_t *sketch, uint64_t sketch_size)
{
#ifdef VERBOSE
  printf("[");
  for (uint64_t i = 0; i < sketch_size; i++) {
    printf("%02x", sketch[i]);
    if (i + 1 < sketch_size) {
      printf("|");
    } else {
      printf("]\n");
    }
  }
#endif
}

#define BYTE_TO_BINARY_PATTERN "%c%c%c%c%c%c%c%c"
#define BYTE_TO_BINARY(byte)                                                   \
  ((byte)&0x80 ? '1' : '0'), ((byte)&0x40 ? '1' : '0'),                        \
    ((byte)&0x20 ? '1' : '0'), ((byte)&0x10 ? '1' : '0'),                      \
    ((byte)&0x08 ? '1' : '0'), ((byte)&0x04 ? '1' : '0'),                      \
    ((byte)&0x02 ? '1' : '0'), ((byte)&0x01 ? '1' : '0')

static inline void
verbose_print_mask_8(uint8_t mask)
{
#ifdef VERBOSE
  printf(BYTE_TO_BINARY_PATTERN, BYTE_TO_BINARY(mask));
  printf("\n");
#endif
}

static inline void
verbose_print_double_mask_8(uint16_t mask)
{
#ifdef VERBOSE
  verbose_print_mask_8(mask);
  verbose_print_mask_8(mask >> 8);
#endif
}

static inline void
verbose_print_mask_64(uint64_t mask)
{
#ifdef VERBOSE
  for (uint64_t i = 0; i < 8; i++) {
    printf(BYTE_TO_BINARY_PATTERN, BYTE_TO_BINARY((uint8_t)mask));
    mask >>= 8;
  }
  printf("\n");
#endif
}

static inline void
verbose_print_operation(char *op_name, uint64_t key, uint64_t value)
{
#ifdef VERBOSE
  printf("%-15s key: 0x%016" PRIx64 ", value: 0x%016" PRIx64 "\n",
         op_name,
         key,
         value);
#endif
}

static inline void
verbose_end(char *op_name, bool internal)
{
#ifdef VERBOSE
  printf("END %s\n", op_name);
  if (!internal) {
    printf("\n");
  }
#endif
}
