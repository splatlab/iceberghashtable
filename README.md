# iceberghashtable
IcebergHT: High Performance Hash Tables Through Stability and Low Associativity

Overview
--------
 IcebergHT is a fast, concurrent, and resizeable hash table implementation. It supports
 insertions, deletions and queries for 64-bit keys and values.
 
API
--------
* 'iceberg_insert(KeyType key, ValueType value)': insert a key-value pair to the hash table
* 'iceberg_get_value(KeyType key)': return the value associated with the key.
* 'iceberg_remove(KeyType key)': remove the key. 

Build
-------
This library depends on libssl, libtbb, and libpmem. 

The code uses vector instructions to speed up operatons. 

```bash
 $ make main
 $ ./main -s 24 -t 4
```

The -s flag specifies the log of the number of slots, and the -t flag specifies the number of threads.

To build for PMEM,
```bash
 $ PMEM=1 make main
 $ ./main -p /path/to/pmem -s 24 -t 4
```

Contributing
------------
Contributions via GitHub pull requests are welcome.



