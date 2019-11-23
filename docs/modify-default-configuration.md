---
id: modify-default-configuration
title: Modify Default Configuration
sidebar_label: Modify Default Configuration
---
AquilaDB is fully customizable with a single configuration file. AquilaDB when deployed, will by default look for [`src/DB_config.yml`](https://github.com/a-mma/AquilaDB/blob/develop/src/DB_config.yml) file to configure itself. This configuration file can be modified before building your docker image. Here is an example of a configuration file:

```
docs:
  vecount: 100 # minimum data required to start indexing
faiss:
  init:
    nlist: 1 # number of cells 
    nprobe: 1 # number of cells that are visited to perform a search
    bpv: 8 # bytes per vector
    bpsv: 8 # bytes per sub-vector
    vd: 784 # fixed vector dimension
annoy:
  init:
    vd: 784 # fixed vector dimension
    smetric: 'angular' # similarity metric to be used
    ntrees: 10 # no. of trees
couchDB:
  DBInstance: default # database namespace
  host: /data
  user: root
  password: 
vectorID:
  sync_t: 5000
```

You can override these configurations with docker `ENVIRONMENT VARIABLES` when deployed.

```
MIN_DOCS2INDEX - docs.vecount
MAX_CELLS - faiss.init.nlist
VISIT_CELLS - faiss.init.nprobe
BYTES_PER_VEC - faiss.init.bpv
BYTES_PER_SUB_VEC - faiss.init.bpsv
FIXED_VEC_DIMENSION - faiss.init.vd
DB_HOST - couchDB.host
FIXED_VEC_DIMENSION - annoy.init.vd
ANNOY_SIM_METRIC - annoy.init.smetric
ANNOY_NTREES - annoy.init.ntrees
```
