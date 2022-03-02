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
OPT += $(RESIZE_POLICY)

ifdef INST
	THRPT_POLICY = -DINSTAN_THRPT
endif
OPT += $(THRPT_POLICY)

ifdef LATENCY
	LATENCY_POLICY = -DLATENCY
endif
OPT += $(LATENCY_POLICY)

CC = clang
CPP = clang++
CFLAGS = $(OPT) -Wall -march=native -pthread $(HUGE)
INCLUDE = -I ./include
SOURCES = src/iceberg_table.c src/hashutil.c src/partitioned_counter.c src/lock.c
OBJECTS = $(subst src/,obj/,$(subst .c,.o,$(SOURCES)))
LIBS = -lssl -lcrypto -lpmem -ltbb

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
