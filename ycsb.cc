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
#include "tbb/tbb.h"

#include "iceberg_table.h"

// Dash
#include "Hash.h"
#include "allocator.h"
#include "ex_finger.h"

// pool path and name
static const char *pool_name = "/mnt/pmem1/pmem_hash.data";
// pool size
static const size_t pool_size = 1024ul * 1024ul * 1024ul * 10ul;

using namespace std;

thread_local double read_time = 0.0;
thread_local double insert_time = 0.0;
thread_local uint64_t read_count = 0;
thread_local uint64_t insert_count = 0;

std::atomic<uint64_t> global_read_time = 0.0;
std::atomic<uint64_t> global_insert_time = 0.0;
std::atomic<uint64_t> global_read_count = 0;
std::atomic<uint64_t> global_insert_count = 0;

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

#define EPOCH_SIZE 1024

typedef struct dash_thread_data {
    uint64_t id;
    Hash<uint64_t> *ht;
} dash_thread_data;

/////////////////////////////////////////////////////////////////////////////////

static uint64_t LOAD_SIZE = 64000000;
static uint64_t RUN_SIZE = 1280000000ULL;

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
    while ((count < LOAD_SIZE) && infile_load.good()) {
        infile_load >> op >> key;
        if (op.compare(insert) != 0) {
            std::cout << "READING LOAD FILE FAIL!\n";
            return ;
        }
        init_keys.push_back(key);
        count++;
    }

    //fprintf(stderr, "Loaded %d keys\n", count);

    std::ifstream infile_txn(txn_file);

    count = 0;
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
        count++;
    }
    count--;
    //printf("Count: %lu\n", count);

    std::atomic<int> range_complete, range_incomplete;
    range_complete.store(0);
    range_incomplete.store(0);

    if (index_type == TYPE_ICEBERG) {
        iceberg_table hashtable;
        iceberg_init(&hashtable, 24);

        thread_data_t *tds = (thread_data_t *) malloc(num_thread * sizeof(thread_data_t));

        std::atomic<int> next_thread_id;

        double load_throughput;
        double run_throughput;
        {
            // Load
            auto starttime = std::chrono::high_resolution_clock::now();
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
                }
            };

            std::vector<std::thread> thread_group;

            for (int i = 0; i < num_thread; i++)
                thread_group.push_back(std::thread{func});

            for (int i = 0; i < num_thread; i++)
                thread_group[i].join();

            //iceberg_end(&hashtable);
            //auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
            //        std::chrono::high_resolution_clock::now() - starttime);
            //printf("Throughput: load, %f ,ops/us\n", (LOAD_SIZE * 1.0) / duration.count());
            //printf("Blocks resolved: %lu\n", hashtable.metadata.lv1_resize_ctr);
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
                    std::chrono::high_resolution_clock::now() - starttime);
            load_throughput = (LOAD_SIZE * 1.0) / duration.count();
            //printf("Throughput: load, %f ,ops/us\n", (LOAD_SIZE * 1.0) / duration.count());
        }

        {
            // Run
            auto starttime = std::chrono::high_resolution_clock::now();
            next_thread_id.store(0);
            auto func = [&]() {
                int thread_id = next_thread_id.fetch_add(1);
                tds[thread_id].id = thread_id;
                tds[thread_id].ht = &hashtable;

                uint64_t start_key = count / num_thread * (uint64_t)thread_id;
                uint64_t end_key = start_key + count / num_thread;

                for (uint64_t i = start_key; i < end_key; i++) {
                    if (ops[i] == OP_INSERT) {
                        //auto opstarttime = std::chrono::high_resolution_clock::now();
                        if(!iceberg_insert(tds[thread_id].ht, keys[i],
                                    keys[i], thread_id)) {
                            printf("Failed insert\n");
                            exit(0);
                        }
                        //auto opendtime = std::chrono::high_resolution_clock::now();
                        //insert_time += std::chrono::duration_cast<std::chrono::nanoseconds>(opendtime - opstarttime).count();
                        insert_count++;
                    } else if (ops[i] == OP_READ) {
                        uintptr_t *val;
                        //auto opstarttime = std::chrono::high_resolution_clock::now();
                        auto ret = iceberg_get_value(tds[thread_id].ht, keys[i], &val, thread_id);
                        //auto opendtime = std::chrono::high_resolution_clock::now();
                        //read_time += std::chrono::duration_cast<std::chrono::nanoseconds>(opendtime - opstarttime).count();
                        read_count++;

                        if (*val != keys[i]) {
                            std::cout << "[ICEBERG] wrong key read: " << *val << " expected: " << keys[i] << " ret: " << ret << std::endl;
                            //exit(1);
                        }
                    } else if (ops[i] == OP_SCAN) {
                        std::cout << "NOT SUPPORTED CMD!\n";
                        exit(0);
                    } else if (ops[i] == OP_UPDATE) {
                        std::cout << "NOT SUPPORTED CMD!\n";
                        exit(0);
                    }
                }
                //global_read_time.fetch_add((uint64_t)read_time, std::memory_order_seq_cst);
                //global_insert_time.fetch_add((uint64_t)insert_time, std::memory_order_seq_cst);
                //global_read_count.fetch_add(read_count, std::memory_order_seq_cst);
                //global_insert_count.fetch_add(insert_count, std::memory_order_seq_cst);
            };

            std::vector<std::thread> thread_group;

            for (int i = 0; i < num_thread; i++)
                thread_group.push_back(std::thread{func});

            for (int i = 0; i < num_thread; i++)
                thread_group[i].join();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
                    std::chrono::high_resolution_clock::now() - starttime);
            run_throughput = (count * 1.0) / duration.count();
            printf("%lu %f %f\n", num_thread, load_throughput * 1000000, run_throughput * 1000000);
            //printf("Time: %luus\n", duration.count());
            //printf("Throughput: run, %f ,ops/us\n", (count * 1.0) / duration.count());
            //printf("Time: %fus\n", duration.count());
            //printf("Blocks resolved: %lu\n", hashtable.metadata.lv1_resize_ctr);
            //printf("Read Throughput: %f ops/us\n", global_read_count * 1000.0 / global_read_time);
            //printf("Insert Throughput: %f ops/us\n", global_insert_count * 1000.0 / global_insert_time);
        }
        // TODO: Add a iceberg destroy function
    } else if (index_type == TYPE_DASH) {
        // Step 1: create (if not exist) and open the pool
        bool file_exist = false;
        if (FileExists(pool_name)) file_exist = true;
        Allocator::Initialize(pool_name, pool_size);

        // Step 2: Allocate the initial space for the hash table on PM and get the
        // root; we use Dash-EH in this case.
        Hash<uint64_t> *hash_table = reinterpret_cast<Hash<uint64_t> *>(
                Allocator::GetRoot(sizeof(extendible::Finger_EH<uint64_t>)));

        // Step 3: Initialize the hash table
        if (!file_exist) {
            // During initialization phase, allocate 64 segments for Dash-EH
            size_t segment_number = 1 << 15;
            new (hash_table) extendible::Finger_EH<uint64_t>(
                    segment_number, Allocator::Get()->pm_pool_);
        }else{
            new (hash_table) extendible::Finger_EH<uint64_t>();
        }

        std::atomic<int> next_thread_id;

        dash_thread_data *dtd = (dash_thread_data *)malloc(num_thread * sizeof(dash_thread_data));

        double load_throughput;
        double run_throughput;

        {
            // Load
            auto starttime = std::chrono::high_resolution_clock::now();
            next_thread_id.store(0);
            auto func = [&]() {
                int thread_id = next_thread_id.fetch_add(1);
                dtd[thread_id].id = thread_id;
                dtd[thread_id].ht = hash_table;

                uint64_t start_key = LOAD_SIZE / num_thread * (uint64_t)thread_id;
                uint64_t end_key = start_key + LOAD_SIZE / num_thread;

                uint64_t num_epochs = ((end_key - start_key - 1) / EPOCH_SIZE) + 1;
                for (uint64_t epoch = 0; epoch < num_epochs; epoch++) {
                    auto epoch_guard = Allocator::AquireEpochGuard();
                    for (uint64_t i = 0; i < EPOCH_SIZE; i++) {
                        uint64_t curr_key = start_key + epoch * EPOCH_SIZE + i;
                        if (curr_key >= end_key) {
                            break;
                        }
                        auto ret = dtd[thread_id].ht->Insert(init_keys[curr_key], DEFAULT, true);
                        if (ret == -1) {
                            printf("Duplicate insert\n");
                            exit(0);
                        }
                    }
                }
            };

            std::vector<std::thread> thread_group;

            for (int i = 0; i < num_thread; i++)
                thread_group.push_back(std::thread{func});

            for (int i = 0; i < num_thread; i++)
                thread_group[i].join();

            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
                    std::chrono::high_resolution_clock::now() - starttime);
            load_throughput = (LOAD_SIZE * 1.0) / duration.count();
        }

        {
            // Run
            auto starttime = std::chrono::high_resolution_clock::now();
            next_thread_id.store(0);
            auto func = [&]() {
                int thread_id = next_thread_id.fetch_add(1);
                dtd[thread_id].id = thread_id;
                dtd[thread_id].ht = hash_table;

                uint64_t start_key = count / num_thread * (uint64_t)thread_id;
                uint64_t end_key = start_key + count / num_thread;

                uint64_t num_epochs = ((end_key - start_key - 1) / EPOCH_SIZE) + 1;
                for (uint64_t epoch = 0; epoch < num_epochs; epoch++) {
                    auto epoch_guard = Allocator::AquireEpochGuard();
                    for (uint64_t i = 0; i < EPOCH_SIZE; i++) {
                        uint64_t curr = start_key + epoch * EPOCH_SIZE + i;
                        if (curr >= end_key) {
                            break;
                        }
                        switch (ops[curr]) {
                            case OP_INSERT:
                                {
                                    auto ins_ret = dtd[thread_id].ht->Insert(keys[curr], DEFAULT, true);
                                    if (ins_ret == -1) {
                                        printf("Duplicate insert\n");
                                        exit(0);
                                    }
                                    break;
                                }
                            case OP_READ:
                                {
                                    auto read_ret = dtd->ht->Get(keys[curr], true);
                                    if (read_ret == NONE) {
                                        printf("false negative query %lu\n", keys[curr]);
                                    }
                                    break;
                                }
                            case OP_SCAN:
                                std::cout << "NOT SUPPORTED CMD!\n";
                                exit(0);
                            case OP_UPDATE:
                                std::cout << "NOT SUPPORTED CMD!\n";
                                exit(0);
                        }
                    }
                }
            };

            std::vector<std::thread> thread_group;

            for (int i = 0; i < num_thread; i++)
                thread_group.push_back(std::thread{func});

            for (int i = 0; i < num_thread; i++)
                thread_group[i].join();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
                    std::chrono::high_resolution_clock::now() - starttime);
            run_throughput = (count * 1.0) / duration.count();
            printf("%lu %f %f\n", num_thread, load_throughput * 1000000, run_throughput * 1000000);
        }
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

    //printf("%s, workload%s, %s, %s, threads %s\n", argv[1], argv[2], argv[3], argv[4], argv[5]);

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
    tbb::task_scheduler_init init(num_thread);

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
