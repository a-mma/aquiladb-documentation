---
id: replication
title: Replication and Sharding
sidebar_label: Replication and Sharding
---

> This feature is not stable and is only available in AquilaDB `develop` branch and `bleeding` docker image.

## Couch Protocol

AquilaDB integrates [Couch Replication Protocol](https://docs.couchdb.org/en/2.3.1/replication/intro.html). With this design choice, AquilaDB is now being part of the whole Couch movement. It is able to communicate to any Couch variant (CouchDB, PouchDB, Cloudant etc.).

To connect AquilaDB to a Couch node and start data sync with it, you just need to configure [`DB_config.yml`](https://github.com/a-mma/AquilaDB/blob/develop/src/DB_config.yml) the following way,

```
couchDB:
  DBInstance: default # database namespace
  host: /data # this will store documents within AquilaDB volume. Changing this to a remote couchDB endpoint will use that DB instead (not recommended unless you know what's happening)
  user: root # username, if above host requires authentication
  password:  # password, if above host requires authentication
couchDBRemote:
  DBInstance: default # database namespace
  host: # Specify if data to be replicated from AquilaDB to a remote Couch Variant and vice versa
  user: # username, if above host requires authentication
  password: # password, if above host requires authentication
```
That's it. Everything is straight forward.

## Deployment patterns
Below are some possible deployment patterns that's being enabled by Couch Protocol Integration
![ADB1](https://user-images.githubusercontent.com/19545678/73383959-58587d80-42f0-11ea-860e-5572387652d4.jpg)
![ADB2](https://user-images.githubusercontent.com/19545678/73383961-58f11400-42f0-11ea-9d7f-f58755ebb2ca.jpg)
![ADB3](https://user-images.githubusercontent.com/19545678/73383970-5d1d3180-42f0-11ea-9ded-d2968eed0526.jpg)
