ifdef D
   OPT= -g
else
   OPT= -g -flto -Ofast
endif

ifdef H
   HUGE= -DHUGE_TLB
else
   HUGE=
endif

RESIZE_POLICY = -DENABLE_RESIZE
BLOCK_LOCKING = -DENABLE_BLOCK_LOCKING

ifdef NBL
	BLOCK_LOCKING =
endif

ifdef NORESIZE
	RESIZE_POLICY =
endif

ifdef INST
	THRPT_POLICY = -DINSTAN_THRPT
endif

ifdef LATENCY
	LATENCY_POLICY = -DLATENCY
endif

ifdef PMEM
	PMEM_POLICY = -DPMEM
endif

OPT += $(RESIZE_POLICY) $(BLOCK_LOCKING) $(THRPT_POLICY) $(LATENCY_POLICY) $(PMEM_POLICY)

CC = clang
CPP = clang++
CFLAGS = $(OPT) -Wall -march=native -pthread -Werror -Wfatal-errors $(HUGE)
CPPFLAGS = $(OPT) -Wall -march=native -pthread -Werror -Wfatal-errors $(HUGE) -std=c++11
INCLUDE = -I ./include
SOURCES = src/iceberg_table.c src/hashutil.c src/partitioned_counter.c src/lock.c
OBJECTS = $(subst src/,obj/,$(subst .c,.o,$(SOURCES)))

ifdef PMEM
INCLUDE += -I ./pmdk/src/PMDK/src/include
LIBS +=  -L ./pmdk/src/PMDK/src/nondebug -lpmem -lpmemobj
endif

all: main ycsb

obj/%.o: src/%.c
	@ mkdir -p obj
	$(CC) $(CFLAGS) $(INCLUDE) -c $< -o $@

obj/main.o: main.cc
	@ mkdir -p obj
	$(CPP) $(CPPFLAGS) $(INCLUDE) -c $< -o $@

obj/ycsb.o: ycsb.cc
	@ mkdir -p obj
	$(CPP) $(CPPFLAGS) $(INCLUDE) -c $< -o $@

main: $(OBJECTS) obj/main.o
	$(CPP) $(CFLAGS) $^ -o $@ $(LIBS)

ycsb: $(OBJECTS) obj/ycsb.o
	$(CPP) $(CFLAGS) $^ -o $@ $(LIBS)

.PHONY: clean directories

clean:
	rm -f main ycsb $(OBJECTS) obj/main.o obj/ycsb.o
