ifdef D
   OPT= -g
else
   OPT= -flto -Ofast
endif

ifdef H
   HUGE= -DHUGE_TLB
else
   HUGE=
endif

RESIZE_POLICY = -DENABLE_RESIZE

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

OPT += $(RESIZE_POLICY) $(THRPT_POLICY) $(LATENCY_POLICY) $(PMEM_POLICY)

CC = clang
CPP = clang++
CFLAGS = $(OPT) -Wall -march=native -pthread $(HUGE)
INCLUDE = -I ./include
SOURCES = src/iceberg_table.c src/hashutil.c src/partitioned_counter.c src/lock.c
OBJECTS = $(subst src/,obj/,$(subst .c,.o,$(SOURCES)))
LIBS = -lssl -lcrypto -lpmem -ltbb

ifdef PMEM
INCLUDE += ./pmdk/src/PMDK/src/include
LIBS +=  ./pmdk/src/PMDK/src/nondebug -lpmem -lpmemobj
endif

all: main ycsb

obj/%.o: src/%.c
	@ mkdir -p obj
	$(CC) $(CFLAGS) $(INCLUDE) -c $< -o $@

obj/main.o: main.cc
	@ mkdir -p obj
	$(CPP) $(CFLAGS) $(INCLUDE) -c $< -o $@

obj/ycsb.o: ycsb.cc
	@ mkdir -p obj
	$(CPP) $(CFLAGS) $(INCLUDE) -c $< -o $@

main: $(OBJECTS) obj/main.o
	$(CPP) $(CFLAGS) $(INCLUDE) $^ -o $@ $(LIBS)

ycsb: $(OBJECTS) obj/ycsb.o
	$(CPP) $(CFLAGS) $(INCLUDE) $^ -o $@ $(LIBS)

.PHONY: clean directories

clean:
	rm -f main ycsb $(OBJECTS) obj/main.o obj/ycsb.o
