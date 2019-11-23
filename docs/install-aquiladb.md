---
id: install-aquiladb
title: Install AquilaDB
sidebar_label: Install AquilaDB
---

## AquilaDB Server
### from dockerhub
AquilaDB docker image builds are automated at [ammaorg/aquiladb dockerhub repository](https://hub.docker.com/r/ammaorg/aquiladb/tags). You will find three types of labels over there: latest, bleeding and release-v\*.

**latest** - automated builds from Github [master branch](https://github.com/a-mma/AquilaDB/tree/master)

**bleeding** - automated builds from Github [develop (default) branch](https://github.com/a-mma/AquilaDB/tree/develop)

**release-v\*** - automated builds for software [release versions](https://github.com/a-mma/AquilaDB/releases) on Github

So, it's necessary to have [docker installed](https://docs.docker.com/v17.09/engine/installation/#supported-platforms) on your system before proceeding. 

After you have installed docker in your system, you need to download the AquilaDB image to your system. To do that, run in terminal:

```
docker pull ammaorg/aquiladb:<LABEL>
```


### from the source
Once you have cloned the repository and checked out the branch you are interested in, you need to build a local docker image. To do that, run in terminal:


```
cd <AQUILADB SOURCE ROOT DIRECTORY>
docker build -t ammaorg/aquiladb:latest .
```

It is possible to configure the database default properties before building the image. To know more, go to [configuration](https://github.com/a-mma/AquilaDB/wiki/Default-Configuration).
## AquilaDB Client
To communicate with the AquilaDB server you can use a client library that abstracts gRPC API endpoint communications.
### python
To install python client, run in terminal:

```
pip install aquiladb
```

### Node JS
[(development in progress)](https://github.com/a-mma/AquilaDB-NodeJS)