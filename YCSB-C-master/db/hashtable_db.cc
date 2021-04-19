//
//  hashtable_db.cc
//  YCSB-C
//
//  Created by Jinglei Ren on 12/24/14.
//  Copyright (c) 2014 Jinglei Ren <jinglei@ren.systems>.
//

#include "db/hashtable_db.h"

#include <string>
#include <vector>
#include "lib/string_hashtable.h"

using std::string;
using std::vector;
using vmp::StringHashtable;

namespace ycsbc {

int HashtableDB::Read(const string &table, const string &key,
    const vector<string> *fields, vector<KVPair> &result) {
  table.get_value(&key, &result);
  return DB::kOK;
}

int HashtableDB::Scan(iceberg_table * table, const string &key, int len,
    const vector<string> *fields, vector<vector<KVPair>> &result) {
  string key_index(table + key);
  result.clear();
  for(int i = 0; i < len; ++i) {
	  vector<KVPair> * value;
	  table.get_value((&key) + i, &value);
	  result.push_back(*value);
  }
  return DB::kOK;
}

int HashtableDB::Update(iceberg_table * table, const string &key,
    vector<KVPair> &values) {
  table.insert(&key, &values);
  return DB::kOK;
}

int HashtableDB::Insert(iceberg_table * table, const string &key,
    vector<KVPair> &values) {
  table.insert(&key, &values);
  return DB::kOK;
}

int HashtableDB::Delete(iceberg_table * table, const string &key) {
  table.remove(&key);
  return DB::kOK;
}

} // ycsbc
