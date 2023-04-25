ifdef DEBUG
   OPT= -g
else
   OPT= -g -flto -Ofast
endif

ifdef VERBOSE
   OPT += -DVERBOSE
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

OPT += $(RESIZE_POLICY) $(THRPT_POLICY) $(LATENCY_POLICY)

CC = clang
CPP = clang++
CFLAGS = $(OPT) -Wall -march=native -pthread -Werror -Wfatal-errors $(HUGE) -DXXH_INLINE_ALL
CPPFLAGS = $(OPT) -Wall -march=native -pthread -Werror -Wfatal-errors $(HUGE) -std=c++11
INCLUDE = -I ./include -I ./src -I ./xxhash
SOURCES = src/iceberg_table.c
HEADERS = include/iceberg_table.h src/iceberg_precompute.h src/lock.h src/counter.h src/util.h src/verbose.h
OBJECTS = $(subst src/,obj/,$(subst .c,.o,$(SOURCES)))

all: micro ycsb

obj/iceberg_table.o: src/iceberg_table.c $(HEADERS)
	@ mkdir -p obj
	$(CC) $(CFLAGS) $(INCLUDE) -c $< -o $@

obj/micro.o: micro.cc obj/iceberg_table.o
	@ mkdir -p obj
	$(CPP) $(CPPFLAGS) $(INCLUDE) -c $< -o $@

obj/ycsb.o: ycsb.cc obj/iceberg_table.o
	@ mkdir -p obj
	$(CPP) $(CPPFLAGS) $(INCLUDE) -c $< -o $@

micro: $(OBJECTS) obj/micro.o
	$(CPP) $(CFLAGS) $^ -o $@ $(LIBS)

ycsb: $(OBJECTS) obj/ycsb.o
	$(CPP) $(CFLAGS) $^ -o $@ $(LIBS)

.PHONY: clean format

clean:
	rm -f micro ycsb $(OBJECTS) obj/micro.o obj/ycsb.o

format:
	clang-format-13 -style=file -i $(SOURCES) $(HEADERS)
