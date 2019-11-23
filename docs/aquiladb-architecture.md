---
id: aquiladb-architecture
title: Technology 
sidebar_label: AquilaDB Architecture
---

# Technology
AquilaDB is not written from scratch thanks to Open Source community with a combination of awesome libraries out there AquilaDB provides a unique solution to data scientists and app developers. 

AquilaDB it is meant to be a well maintained a solution so that, you don't have to worry about the engineering, optimization and networking complexities of an Information retrieval system. Only focus on your idea, start building your Machine Learning application during the coffee time and we do the rest.

## High-level workflow with AquilaDB

![AquilaDB High-level](https://user-images.githubusercontent.com/19545678/68526888-b172cf80-0306-11ea-86cd-55a17050dfa3.png)

AquilaDB exposes a gRPC API endpoint to the outside world. Any application can connect to it and send latent vectors along with JSON formatted metadata to this endpoint. We are working on client-side libraries to abstract this communication complexity. Once you have indexed enough data to AquilaDB, you can send a query vector to the same endpoint to retrieve similar vectors along with their metadata.

You can connect AquilaDB to any other database which supports Couch protocol and replicate vectors from that database to your AquilaDB instance (eventual consistency). This replication procedure is silent and you don't have to worry about anything. 

## AquilaDB core components

![AquilaDB Architecture](https://user-images.githubusercontent.com/19545678/68530914-d2511a00-0332-11ea-9d35-8b8e045926dd.png)

Here is the AquilaDB v1.0 architecture (development in progress). It consists of three main components: Document store, Vector store and Peer Manager. 

Document store manages JSON metadata storage and retrieval along with Couch replication. We currently rely on the PouchDB library with Level backend to implement this module. 

The vector store manages the storage and fast k-NN retrieval of latent vectors. As of current implementation, it only supports the cosine similarity measure for k-NN retrieval. This module is supported by FAISS and Annoy libraries. 

Finally, the Peer manager manages the AquilaDB cluster in a decentralized network for index replication and load balancing. This module is in the very basic stage of development as of now. It would rely on IPFS and libp2p libraries for implementation.

**[Next >>](https://github.com/a-mma/AquilaDB/wiki/Client-API-reference)**