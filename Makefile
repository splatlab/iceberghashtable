ifdef D
   OPT=
else
   OPT= -flto -O3
endif

ifdef H
   HUGE= -DHUGE_TLB
else
   HUGE=
endif

RESIZE_POLICY = -DENABLE_FGR

ifdef BGR
	RESIZE_POLICY = -DENABLE_BGR
endif

OPT += $(RESIZE_POLICY) -DPMEM

CC = clang
CPP = clang++ -std=c++17
CFLAGS = -g $(OPT) -Wall -march=native -pthread $(HUGE) -Wfatal-errors
INCLUDE = -I ./include -I ./dash/src -I ./dash/build/_deps/epoch_reclaimer-src -I ./dash/build/pmdk/src/PMDK/src/include
SOURCES = src/iceberg_table.c src/hashutil.c src/partitioned_counter.c src/lock.c
OBJECTS = $(subst src/,obj/,$(subst .c,.o,$(SOURCES)))
LIBS = -L ./dash/build/pmdk/src/PMDK/src/nondebug -Wl,-rpath=/dash/build/pmdk/src/PMDK/src/nondebug -lpmem -lpmemobj -lssl -lcrypto -ltbb

all: main ycsb dash_micro

obj/%.o: src/%.c
	@ mkdir -p obj
	$(CC) $(CFLAGS) $(INCLUDE) -c $< -o $@

obj/main.o: main.cc
	@ mkdir -p obj
	$(CPP) $(CFLAGS) $(INCLUDE) -c $< -o $@

obj/ycsb.o: ycsb.cc
	@ mkdir -p obj
	$(CPP) $(CFLAGS) $(INCLUDE) -c $< -o $@

obj/dash_micro.o: dash_micro.cc
	@ mkdir -p obj
	$(CPP) $(CFLAGS) $(INCLUDE) -c $< -o $@

main: $(OBJECTS) obj/main.o
	$(CPP) $(CFLAGS) $(INCLUDE) $^ -o $@ $(LIBS)

ycsb: $(OBJECTS) obj/ycsb.o
	$(CPP) $(CFLAGS) $(INCLUDE) $^ -o $@ $(LIBS)

dash_micro: $(OBJECTS) obj/dash_micro.o
	$(CPP) $(CFLAGS) $(INCLUDE) $^ -o $@ $(LIBS)

.PHONY: clean directories

clean:
	rm -f main ycsb dash_micro $(OBJECTS) obj/main.o obj/ycsb.o obj/dash_micro.o
