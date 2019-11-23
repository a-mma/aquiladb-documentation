---
id: semantic-text-retrieval
title: Semantic Text Retrieval
sidebar_label: Search text by its meaning
---

# Efficient Semantic Text Retrieval with Google's Universal Sentence Encoder and AquilaDB

_**Download Jupyter notebook [here](https://github.com/a-mma/AquilaDB-Examples/tree/master/QnA_with_USE)**_


In this tutorial, we are going to look at how AquilaDB vector database can help in efficient Semantic Retrieval with Google's Universal Sentence Encoder (USE).

This tutorial is following the same idea as described [in latest post on Universal Sentence Encoder at Google AI blog](https://ai.googleblog.com/2019/07/multilingual-universal-sentence-encoder.html). We encourage you to read that blog before proceeding. Because it is very useful to get a context on what we are going to do below.

> One difference in this tutorial is that, we use `universal-sentence-encoder-large` which belongs to the same `USE` family for simplicity in explanation. The idea explained here is the very same for all models in `USE` family.

This is an image taken from that blog post. A recommended pipeline for textual similarity. `AquilaDB` will cover `pre-encoded Candidates` data store and `ANN search` modules in this pipeline. Cool.. Right?

![A prototypical semantic retrieval pipeline, used for textual similarity](https://1.bp.blogspot.com/-q1g13xLR-9E/XSi8ZewIXzI/AAAAAAAAETQ/Oek9K51ZrAQvbZL3t3rme5HcegzCNm98QCEwYBhgL/s640/image1.png)


```python
# Let's import required modules

import tensorflow as tf
import tensorflow_hub as hub
```

### Load pretrained encoder
We need to load pre-trained USE model from Tensorflow Hub. We use this model to encode our sentences before sending it to AquilaDB for indexing and querying.


```python
use_module_url = "https://tfhub.dev/google/universal-sentence-encoder-large/3"

# load Universal Sentence Encoder module from tensor hub
embed_module = hub.Module(use_module_url)
```

Let's test our loaded model with some random texts before proceeding.


```python
# let's create some test sentanaces
test_messages = ["AquilaDB is a Resillient, Replicated, Decentralized, Host neutral storage for Feature Vectors along with Document Metadata.", 
            "Do k-NN retrieval from anywhere, even from the darkest rifts of Aquila (in progress). It is easy to setup and scales as the universe expands."]
```

We feed our text array to model for embedding. Don't forget to wrap the embedding logic into a method to reuse it.


```python
# helper function to generate embedding for input array of sentances
def generate_embeddings (messages_in):
    # generate embeddings
    with tf.Session() as session:
        session.run([tf.global_variables_initializer(), tf.tables_initializer()])
        message_embeddings = session.run(embed_module(messages_in))
        
    return message_embeddings
```


```python
# print generated embeddings
print(generate_embeddings(test_messages))
```

    [[-0.00570544  0.01024008  0.04416275 ...  0.03282805 -0.01723128
       0.00956334]
     [ 0.0124177   0.09862255  0.06958324 ... -0.00700251  0.02332876
      -0.09377097]]


As you can see above, we were able to encode out random texts into corresponding sentance embedding with `USE`.

### Let's load actual data
We will be loading some text from a [text file](https://github.com/a-mma/AquilaDB-Examples/blob/master/QnA_with_USE/article_set.txt?raw=true). This is a small wiki article set in plain text format.


```python
with open('article_set.txt', 'r') as file_in:
    lines = file_in.readlines()
```

Let's write some helper functions. These functions will help us communicate with AquilaDB. You don't have to worry about this part now. Just keep it as is except for the IP address `192.168.1.100`. Replace that with the IP address where your AquilaDB installation is. Most probably, it is the same machine you are using now - then give `localhost` as address.


```python
# helper functions to generate documents

import grpc

import vecdb_pb2
import vecdb_pb2_grpc

channel = grpc.insecure_channel('192.168.1.100:50051')
stub = vecdb_pb2_grpc.VecdbServiceStub(channel)

# API interface to add documents to AquilaDB
def addDocuments (documents_in):
    response = stub.addDocuments(vecdb_pb2.addDocRequest(documents=documents_in))
    return response


import base64
import json

# helper function to convert native data to API friendly data
def convertDocuments(vector, document):
    return {
            "vector": {
                "e": vector
            },
            "b64data": json.dumps(document, separators=(',', ':')).encode('utf-8')
        }


# API interface to get nearest documents from AquilaDB
def getNearest (matrix_in, k_in):
    response = stub.getNearest(vecdb_pb2.getNearestRequest(matrix=matrix_in, k=k_in))
    return response


# helper function to convert native data to API friendly data
def convertMatrix(vector):
    return [{
            "e": vector
    }]
```

### Send documents to AquilaDB for indexing
As mentioned previously, we need to store pre encoded candidates in a vector database to perform semantic similarity retrieval later. So, what we are going to do here is to take each line from wiki articles, encode them with `USE` model, attach the original wiki text with the resulting vector as metadata and send them to AquilaDB for indexing.


```python
import time

# set a batch length
batch_len = 200
# counter to init batch sending of documents
counter = 0
# to keep generated documents
docs_gen = []
# to keep lines batch
lbatch = []

for line in lines:
    lbatch.append(line)
    if len(lbatch) == batch_len:
        counter = counter + 1
        # generate embeddings
        vectors = generate_embeddings(lbatch)
        for i in range(len(vectors)):
            docs_gen.append(convertDocuments(vectors[i], {"text": lbatch[i]}))
        # add documents to AquilaDB
        response = addDocuments(docs_gen)
        print("index: "+str(counter), "inserted: "+str(len(response._id)))
        docs_gen = []
        lbatch = []
```

    index: 1 inserted: 199
    index: 2 inserted: 178
    ...
    index: 77 inserted: 186


### Query the database
Now, we need to retrieve semantically similar sentance to our input query from the database. It is straight forward. Just encode the query text with the same `USE` model and then perform k-NN query on the database.


```python
# Method to query for nearest neighbours
def query_nn (query):
    query = [query]
    vector = generate_embeddings(query)[0]
    
    converted_vector = convertMatrix(vector)
    nearest_docs_result = getNearest(converted_vector, 1)
    nearest_docs_result = json.loads(nearest_docs_result.documents)
    
    return nearest_docs_result
```


```python
# Let's try an example query.

print(query_nn('what are the subfamilies of duck')[0]['doc']['text'])
```

    Swans are birds of the family Anatidae, which also includes geese and ducks. Swans are grouped with the closely related geese in the subfamily Anserinae where they form the tribe Cygnini. Sometimes, they are considered a distinct subfamily, Cygninae. Swans usually mate for life, though 'divorce' does sometimes occur, particularly following nesting failure. The number of eggs in each clutch ranges from three to eight.



That's all for this tutorial. Thanks, happy hacking..!

