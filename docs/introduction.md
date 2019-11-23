---
id: introduction
title: Introduction
sidebar_label: Introduction
---

## AquilaDB
AquilaDB is a drop-in solution for data scientists and Machine Learning engineers to do Neural Information retrieval. It thus becomes the muscle memory of your Machine Learning applications. 

AquilaDB at its core is a vector and document database that is easy to set up with docker and start indexing your feature vectors within minutes. AquilaDB exposes GRPC API endpoint through which you can index vectors along with JSON metadata and perform k-NN retrieval of these documents later on. Use your favorite programming language and establish a fast communication channel with AquilaDB.

We currently have a Numpy friendly Python client to abstract GRPC communication with an AquilaDB instance away to help you not worry about the details of communication. We also have a Node JS client in progress.

As of the current development status, AquilaDB can be used as a standalone database. We currently have the development progressing to enable Couch Replication protocol to let AquilaDB clusters to be deployed in a network as decentralized nodes. This will also allow AquilaDB to talk and replicate with any other CouchDB and variants like IBM Cloudant and Couchbase.

In the future, we're planning to take AquilaDB to support IPFS storage and do protocol-agnostic replication with libp2p. We want Machine Learning and Information retrieval to be easy and native to the Web 3.0 as we progress.

For more details on the development of milestones and contribution guides, please visit the [contribute]() section.

## Media
Video: 
[<img alt="introduction to Neural Information retrieval with AquilaDB" src="http://img.youtube.com/vi/-VYpjpLXU5Q/0.jpg" width="300" />](http://www.youtube.com/watch?v=-VYpjpLXU5Q)

Slides: [<img alt="introduction to Neural Information retrieval with AquilaDB" src="https://user-images.githubusercontent.com/19545678/68539883-f56ddf00-03af-11ea-8ed6-6b6d43c6d510.png" width="300" />](http://bit.ly/faya-slideshare)

