---
id: get-started
title: Do your first vector search
sidebar_label: Get Started 
---
### Python Example:
This is a quite minimal code to give you an idea of how AquilaDB client-side code should be skeletoned. When building actual projects, all you need to do is beef up this code structure to include your real data and use case. In the following pages, we will show you multiple real-world use cases, where you will see how this same code structure is followed (gRPC and Python examples are included) to perform Neural Information retrieval tasks.

```
# import AquilaDB client
from aquiladb import AquilaClient as acl

# create DB instance
db = acl('localhost', 50051)

# convert a sample document
# convertDocument
sample = db.convertDocument([0.1,0.2,0.3,0.4], {"hello": "world"})

# add document to AquilaDB
db.addDocuments([sample])

# note that, depending on your default configuration, 
# you need to add docs.vecount number of documents 
# before k-NN search

# create a k-NN search vector
vector = db.convertMatrix([0.1,0.2,0.3,0.4])

# perform k-NN from AquilaDB
k = 10
result = db.getNearest(vector, k)
```
