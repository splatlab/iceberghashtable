CC = clang
CPP = clang++
CFLAGS = -g -flto -O3 -Wall -Werror -Wfatal-errors -march=native -pthread -DHUGE_TLB
#CFLAGS = -g -march=native -pthread -Wall -Werror -Wfatal-errors
INCLUDE = -I ./include
SOURCES = src/iceberg_table.c src/hashutil.c src/partitioned_counter.c
OBJECTS = $(subst src/,obj/,$(subst .c,.o,$(SOURCES)))
LIBS = -lssl -lcrypto -lpmem

all: main directories

obj/%.o: src/%.c
	@ mkdir -p obj
	$(CC) $(CFLAGS) $(INCLUDE) -c $< -o $@

obj/main.o: main.cc
	@ mkdir -p obj
	$(CPP) $(CFLAGS) $(INCLUDE) -c $< -o $@

main: $(OBJECTS) obj/main.o
	$(CPP) $(CFLAGS) $(INCLUDE) $^ -o $@ $(LIBS)

.PHONY: clean directories

clean:
	rm -f main $(OBJECTS) obj/main.o
