#include <iostream>
#include <chrono>
#include <random>
#include <cstring>
#include <vector>
#include <fstream>
#include <iostream>
#include <stdlib.h>
#include <atomic>
#include <thread>

#include "iceberg_table.h"

using namespace std;

// index types
enum {
    TYPE_ICEBERG,
    TYPE_CUCKOO,
    TYPE_DASH,
};

enum {
    OP_INSERT,
    OP_UPDATE,
    OP_READ,
    OP_SCAN,
    OP_DELETE,
};

enum {
    WORKLOAD_A,
    WORKLOAD_B,
    WORKLOAD_C,
    WORKLOAD_D,
    WORKLOAD_E,
};

enum {
    RANDINT_KEY,
    STRING_KEY,
};

enum {
    UNIFORM,
    ZIPFIAN,
};

////////////////////////////////////////////////////////////////////////////////

////////////////////////Helper functions for Icerberg HashTable/////////////////
typedef struct thread_data {
    uint32_t id;
    iceberg_table *ht;
} thread_data_t;

/////////////////////////////////////////////////////////////////////////////////

static uint64_t LOAD_SIZE = 64000000;
static uint64_t RUN_SIZE = 1280000000;

void ycsb_load_run_randint(int index_type, int wl, int kt, int ap, int num_thread,
        std::vector<uint64_t> &init_keys,
        std::vector<uint64_t> &keys,
        std::vector<int> &ranges,
        std::vector<int> &ops)
{
    std::string init_file;
    std::string txn_file;

    if (ap == UNIFORM) {
        if (kt == RANDINT_KEY && wl == WORKLOAD_A) {
            init_file = "/mnt/nvme3/RECIPE/index-microbench/workloads/loada_unif_int.dat";
            txn_file = "/mnt/nvme3/RECIPE/index-microbench/workloads/txnsa_unif_int.dat";
        } else if (kt == RANDINT_KEY && wl == WORKLOAD_B) {
            init_file = "/mnt/nvme3/RECIPE/index-microbench/workloads/loadb_unif_int.dat";
            txn_file = "/mnt/nvme3/RECIPE/index-microbench/workloads/txnsb_unif_int.dat";
        } else if (kt == RANDINT_KEY && wl == WORKLOAD_C) {
            init_file = "/mnt/nvme3/RECIPE/index-microbench/workloads/loadc_unif_int.dat";
            txn_file = "/mnt/nvme3/RECIPE/index-microbench/workloads/txnsc_unif_int.dat";
        } else if (kt == RANDINT_KEY && wl == WORKLOAD_D) {
            init_file = "/mnt/nvme3/RECIPE/index-microbench/workloads/loadd_unif_int.dat";
            txn_file = "/mnt/nvme3/RECIPE/index-microbench/workloads/txnsd_unif_int.dat";
        } else if (kt == RANDINT_KEY && wl == WORKLOAD_E) {
            init_file = "/mnt/nvme3/RECIPE/index-microbench/workloads/loade_unif_int.dat";
            txn_file = "/mnt/nvme3/RECIPE/index-microbench/workloads/txnse_unif_int.dat";
        }
    } else {
        if (kt == RANDINT_KEY && wl == WORKLOAD_A) {
            init_file = "/mnt/nvme3/RECIPE/index-microbench/workloads/loada_unif_int.dat";
            txn_file = "/mnt/nvme3/RECIPE/index-microbench/workloads/txnsa_unif_int.dat";
        } else if (kt == RANDINT_KEY && wl == WORKLOAD_B) {
            init_file = "/mnt/nvme3/RECIPE/index-microbench/workloads/loadb_unif_int.dat";
            txn_file = "/mnt/nvme3/RECIPE/index-microbench/workloads/txnsb_unif_int.dat";
        } else if (kt == RANDINT_KEY && wl == WORKLOAD_C) {
            init_file = "/mnt/nvme3/RECIPE/index-microbench/workloads/loadc_unif_int.dat";
            txn_file = "/mnt/nvme3/RECIPE/index-microbench/workloads/txnsc_unif_int.dat";
        } else if (kt == RANDINT_KEY && wl == WORKLOAD_D) {
            init_file = "/mnt/nvme3/RECIPE/index-microbench/workloads/loadd_unif_int.dat";
            txn_file = "/mnt/nvme3/RECIPE/index-microbench/workloads/txnsd_unif_int.dat";
        } else if (kt == RANDINT_KEY && wl == WORKLOAD_E) {
            init_file = "/mnt/nvme3/RECIPE/index-microbench/workloads/loade_unif_int.dat";
            txn_file = "/mnt/nvme3/RECIPE/index-microbench/workloads/txnse_unif_int.dat";
        }
    }

    std::ifstream infile_load(init_file);

    std::string op;
    uint64_t key;
    int range;

    std::string insert("INSERT");
    std::string update("UPDATE");
    std::string read("READ");
    std::string scan("SCAN");

    int count = 0;
    while (infile_load.good()) {
        infile_load >> op >> key;
        if (op.compare(insert) != 0) {
            std::cout << "READING LOAD FILE FAIL!\n";
            return ;
        }
        init_keys.push_back(key);
        count++;
    }
    count--;

    fprintf(stderr, "Loaded %d keys\n", count);

    std::ifstream infile_txn(txn_file);

    uint64_t txn_count = 0;
    while (infile_txn.good()) {
        infile_txn >> op >> key;
        if (op.compare(insert) == 0) {
            ops.push_back(OP_INSERT);
            keys.push_back(key);
            ranges.push_back(1);
        } else if (op.compare(update) == 0) {
            ops.push_back(OP_UPDATE);
            keys.push_back(key);
            ranges.push_back(1);
        } else if (op.compare(read) == 0) {
            ops.push_back(OP_READ);
            keys.push_back(key);
            ranges.push_back(1);
        } else if (op.compare(scan) == 0) {
            infile_txn >> range;
            ops.push_back(OP_SCAN);
            keys.push_back(key);
            ranges.push_back(range);
        } else {
            std::cout << "UNRECOGNIZED CMD!\n";
            return;
        }
        txn_count++;
    }
    txn_count--;
    fprintf(stderr, "Loaded %" PRIu64 " txn keys\n", txn_count);

    std::atomic<int> range_complete, range_incomplete;
    range_complete.store(0);
    range_incomplete.store(0);

    if (index_type == TYPE_ICEBERG) {
        iceberg_table hashtable;
        iceberg_init(&hashtable, 24);

        thread_data_t *tds = (thread_data_t *) malloc(num_thread * sizeof(thread_data_t));

        std::atomic<int> next_thread_id;

        {
            // Load
            auto starttime = std::chrono::system_clock::now();
            next_thread_id.store(0);
            auto func = [&]() {
                int thread_id = next_thread_id.fetch_add(1);
                tds[thread_id].id = thread_id;
                tds[thread_id].ht = &hashtable;

                uint64_t start_key = LOAD_SIZE / num_thread * (uint64_t)thread_id;
                uint64_t end_key = start_key + LOAD_SIZE / num_thread;

                for (uint64_t i = start_key; i < end_key; i++) {
                  if(!iceberg_insert(tds[thread_id].ht, init_keys[i],
                        init_keys[i], thread_id)) {
                    printf("Failed insert\n");
                    exit(0);
                  }
                  //printf("\rInsert %ld", i);
                  //fflush(stdout);
                }
            };

            std::vector<std::thread> thread_group;

            for (int i = 0; i < num_thread; i++)
                thread_group.push_back(std::thread{func});

            for (int i = 0; i < num_thread; i++)
                thread_group[i].join();

            iceberg_end(&hashtable);
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
                    std::chrono::system_clock::now() - starttime);
            printf("Throughput: load, %f ,ops/us\n", (LOAD_SIZE * 1.0) / duration.count());

#if 0
#if PMEM
            iceberg_dismount(&hashtable);

            starttime = std::chrono::system_clock::now();
            iceberg_mount(&hashtable, 24, 2);
            duration = std::chrono::duration_cast<std::chrono::microseconds>(
                    std::chrono::system_clock::now() - starttime);
            printf("Throughput: mount, %f ,ops/us\n", (LOAD_SIZE * 1.0) / duration.count());
#endif
#endif
        }

        {
            // Run
            auto starttime = std::chrono::system_clock::now();
            next_thread_id.store(0);
            auto func = [&]() {
                int thread_id = next_thread_id.fetch_add(1);
                tds[thread_id].id = thread_id;
                tds[thread_id].ht = &hashtable;

                uint64_t start_key = txn_count / num_thread * (uint64_t)thread_id;
                uint64_t end_key = start_key + txn_count / num_thread;
#ifdef LATENCY
                std::vector<double> insert_times;
                std::vector<double> query_times;
#endif

                for (uint64_t i = start_key; i < end_key; i++) {
                    if (ops[i] == OP_INSERT) {
#ifdef LATENCY
                        std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
#endif
                      if(!iceberg_insert(tds[thread_id].ht, keys[i],
                            keys[i], thread_id)) {
                        printf("Failed insert\n");
                        exit(0);
                      }
#ifdef LATENCY
                        std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
                        insert_times.emplace_back(std::chrono::duration_cast<std::chrono::nanoseconds>(t2-t1).count());
#endif
                    } else if (ops[i] == OP_READ) {
                        iceberg_value_t val;
#ifdef LATENCY
                        std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
#endif
                        auto ret = iceberg_get_value(tds[thread_id].ht, keys[i], &val, thread_id);
                        if (val != keys[i]) {
                            std::cout << "[ICEBERG] wrong key read: " << val << " expected: " << keys[i] << " ret: " << ret << std::endl;
                            exit(1);
                        }
#ifdef LATENCY
                        std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
                        query_times.emplace_back(std::chrono::duration_cast<std::chrono::nanoseconds>(t2-t1).count());
#endif
                    } else if (ops[i] == OP_SCAN) {
                        std::cout << "NOT SUPPORTED CMD!\n";
                        exit(0);
                    } else if (ops[i] == OP_UPDATE) {
                        std::cout << "NOT SUPPORTED CMD!\n";
                        exit(0);
                    }
                }
#ifdef LATENCY
                std::ofstream f;
                f.open("ycsb_insert_times_" + std::to_string(thread_id) + ".log");
                for (auto time : insert_times) {
                    f << time << '\n';
                }
                f.close();

                std::ofstream g;
                g.open("ycsb_query_times_" + std::to_string(thread_id) + ".log");
                for (auto time : query_times) {
                    g << time << '\n';
                }
                g.close();
#endif
            };

            std::vector<std::thread> thread_group;

            for (int i = 0; i < num_thread; i++)
                thread_group.push_back(std::thread{func});

            for (int i = 0; i < num_thread; i++)
                thread_group[i].join();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
                    std::chrono::system_clock::now() - starttime);
            printf("Throughput: run, %f ,ops/us\n", (txn_count * 1.0) / duration.count());
        }
        // TODO: Add a iceberg destroy function
    }
}

int main(int argc, char **argv) {
    if (argc != 6) {
        std::cout << "Usage: ./ycsb [index type] [ycsb workload type] [key distribution] [access pattern] [number of threads]\n";
        std::cout << "1. index type: iceberg cuckoo dash\n";
        std::cout << "2. ycsb workload type: a, b, c, e\n";
        std::cout << "3. key distribution: randint\n";
        std::cout << "4. access pattern: uniform\n";
        std::cout << "5. number of threads (integer)\n";
        return 1;
    }

    printf("%s, workload%s, %s, %s, threads %s\n", argv[1], argv[2], argv[3], argv[4], argv[5]);

    int index_type;
    if (strcmp(argv[1], "iceberg") == 0)
        index_type = TYPE_ICEBERG;
    else if (strcmp(argv[1], "cuckoo") == 0)
        index_type = TYPE_CUCKOO;
    else if (strcmp(argv[1], "dash") == 0)
        index_type = TYPE_DASH;
    else {
        fprintf(stderr, "Unknown index type: %s\n", argv[1]);
        exit(1);
    }

    int wl;
    if (strcmp(argv[2], "a") == 0) {
        wl = WORKLOAD_A;
    } else if (strcmp(argv[2], "b") == 0) {
        wl = WORKLOAD_B;
    } else if (strcmp(argv[2], "c") == 0) {
        wl = WORKLOAD_C;
    } else if (strcmp(argv[2], "d") == 0) {
        wl = WORKLOAD_D;
    } else if (strcmp(argv[2], "e") == 0) {
        wl = WORKLOAD_E;
    } else {
        fprintf(stderr, "Unknown workload: %s\n", argv[2]);
        exit(1);
    }

    int kt;
    if (strcmp(argv[3], "randint") == 0) {
        kt = RANDINT_KEY;
    } else {
        fprintf(stderr, "Unknown key type: %s\n", argv[3]);
        exit(1);
    }

    int ap;
    if (strcmp(argv[4], "uniform") == 0) {
        ap = UNIFORM;
    } else if (strcmp(argv[4], "zipfian") == 0) {
        ap = ZIPFIAN;
    } else {
        fprintf(stderr, "Unknown access pattern: %s\n", argv[4]);
        exit(1);
    }

    int num_thread = atoi(argv[5]);

    if (kt != STRING_KEY) {
        std::vector<uint64_t> init_keys;
        std::vector<uint64_t> keys;
        std::vector<int> ranges;
        std::vector<int> ops;

        init_keys.reserve(LOAD_SIZE);
        keys.reserve(RUN_SIZE);
        ranges.reserve(RUN_SIZE);
        ops.reserve(RUN_SIZE);

        memset(&init_keys[0], 0x00, LOAD_SIZE * sizeof(uint64_t));
        memset(&keys[0], 0x00, RUN_SIZE * sizeof(uint64_t));
        memset(&ranges[0], 0x00, RUN_SIZE * sizeof(int));
        memset(&ops[0], 0x00, RUN_SIZE * sizeof(int));

        ycsb_load_run_randint(index_type, wl, kt, ap, num_thread, init_keys, keys, ranges, ops);
    }
    /*
    else {
        std::vector<Key *> init_keys;
        std::vector<Key *> keys;
        std::vector<int> ranges;
        std::vector<int> ops;

        init_keys.reserve(LOAD_SIZE);
        keys.reserve(RUN_SIZE);
        ranges.reserve(RUN_SIZE);
        ops.reserve(RUN_SIZE);

        memset(&init_keys[0], 0x00, LOAD_SIZE * sizeof(Key *));
        memset(&keys[0], 0x00, RUN_SIZE * sizeof(Key *));
        memset(&ranges[0], 0x00, RUN_SIZE * sizeof(int));
        memset(&ops[0], 0x00, RUN_SIZE * sizeof(int));

        ycsb_load_run_string(index_type, wl, kt, ap, num_thread, init_keys, keys, ranges, ops);
    }
    */

    return 0;
}
