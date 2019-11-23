---
id: hands-on-mnist
title: AquilaDB hands on with MNIST dataset
sidebar_label: AquilaDB hands on with MNIST dataset
---

_**Download Jupyter notebook [here](https://github.com/a-mma/AquilaDB-Examples/tree/master/MNIST_example)**_


To get started, you need docker installed already in your system and run [ammaorg/aquiladb](https://hub.docker.com/r/ammaorg/aquiladb) docker container. [Here](https://github.com/a-mma/AquilaDB#usage) are the setup instructions.

### setup GRPC helper functions

#### install grpc

Execute below commands in your terminal to make `GRPC` available in your system. We will be using `GRPC` for communication with the AquilaDB container.

!pip install grpcio

!pip install protobuf

#### Import grpc, proto etc

You need to copy `vecdb_pb2.py` and `vecdb_pb2_grpc.py` files from `<AquilaDb source directory>/src/test` to the working directory to make sure below imports will work.


```python
from __future__ import print_function

import grpc

import vecdb_pb2
import vecdb_pb2_grpc
```

#### create client connection

Now, let's create a `GRPC` channel between AquilaDB docker container and this notebook.


```python
channel = grpc.insecure_channel('localhost:50051')
stub = vecdb_pb2_grpc.VecdbServiceStub(channel)
```

#### create helper functions

Below are some helper functions that we will be using in this tutorial to interact with AquilaDB APIs. You don't need to worry about these now. We are planning to release client libraries for Node JS and Python ASAP, which will make your life easier.


```python
# API interface to add documents to AquilaDB
def addDocuments (documents_in):
    response = stub.addDocuments(vecdb_pb2.addDocRequest(documents=documents_in))
    return response
```


```python
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
```


```python
# API interface to get nearest documents from AquilaDB
def getNearest (matrix_in, k_in):
    response = stub.getNearest(vecdb_pb2.getNearestRequest(matrix=matrix_in, k=k_in))
    return response
```


```python
# helper function to convert native data to API friendly data
def convertMatrix(vector):
    return [{
            "e": vector
    }]
```

### Load MNIST data

Now let's load MNIST data.


```python
# Helper function to load MNIST data from disk
import numpy as np 
def load_data(path):
    with np.load(path) as f:
        x_train, y_train = f['x_train'], f['y_train']
        x_test, y_test = f['x_test'], f['y_test']
        return (x_train, y_train), (x_test, y_test)
```

##### Get MNIST dataset

Make sure that, you have downloaded Numpy friendly dump of MNIST dataset from [here at Kaggle](https://www.kaggle.com/vikramtiwari/mnist-numpy)


```python
# specify the location of downloaded dataset
# load data into x_train and y_train as image, label respectively.

(x_train, y_train), (x_test, y_test) = load_data('./mnist.npz')
```


```python
# Let's see if everything is properly loaded

print(x_train.shape)
print(y_train.shape)
print(x_train[0].flatten().shape)
```

    (60000, 28, 28)
    (60000,)
    (784,)



##### see contents

We use matplotlib to plot our images. Let's see if we can plot one from our dataset.


```python
from matplotlib import pyplot as plt

# we're going to plot 15th image from our dataset
index = 15

pixels = np.array(x_train[index], dtype='uint8')
label = y_train[index]
plt.title('Label is {label}'.format(label=label))
plt.imshow(pixels, cmap='gray')
plt.show()
```


![png](https://user-images.githubusercontent.com/19545678/60756239-e00b3180-a018-11e9-8962-a5a45deef216.png)


#### Send documents

Everything is setup. Now it's time to work with AquilaDB and see what it can do.

Let's send some documents to DB. Actually, 10201 documents, each containing a single image from MNIST dataset, along with some metadata information. We send image label as metadata information for now, you can send anything that you want to store.

Also note that - by default, you need to send documents not less than 10,000. It is the least number of documents you must keep inside the DB before doing anything else. If you don't have this much data and you don't want to work with this much data, we're sorry, this is not the right place for you..

You can configure the database to meet your requirements by balancing `speed vs accuracy vs data size`. We will discuss that in another tutorial.


```python
import time 

# to keep the generated documents to be sent to DB
docs_gen = []
# based on your system's processing power, you can choose a decent value for batch size.
# Our program will send `batch size` documents to DB and wait 1 second before sending another.
# this is not to blow up CPU and memory while you send large amount of data to DB
# you might ask, if this a `problem` that I need to wait until DB digest each chunk properly and is a waste of time?
# It is not. We will discuss where AquilaDB filts well and how P2P replication handles this smoothly.
# We will also discuss the best practices and see why this is not a big deal in production 
# in another tutorial or blog posts.
batch_len = 200

for i in range (0, 10201):
    vector = x_train[i].flatten().tolist()
    label = y_train[i].item()
    docs_gen.append(convertDocuments(vector, {"label":label}))
    
    # send batch_len batches
    if i%batch_len == 0:
        # add documents to AquilaDB
        response = addDocuments(docs_gen)
        print("index: "+str(i), "inserted: "+str(len(response._id)))
        docs_gen = []
        time.sleep(1)
```

    index: 0 inserted: 1
    index: 200 inserted: 200
    index: 400 inserted: 200
    .....
    .....
    index: 10000 inserted: 200
    index: 10200 inserted: 200


#### find nearest neighbour

Hopefully, you were able to insert images into DB. Now let's query the DB. We will give it an image and will expect the DB to return similar looking images back.

##### Search nearest neighbours


```python
# how many similar images we want in return 
k_in = 5

# what is our example image from the dataset?
index = 40000

# flatten the input image
vector_to_convert = x_train[index].flatten().tolist()
label_to_convert = y_train[index]

# display query image
pixels = x_train[index]
plt.title('Label is {label}'.format(label=label_to_convert))
plt.imshow(pixels, cmap='gray')
plt.show()

# search for nearest results
converted_vector = convertMatrix(vector_to_convert)
nearest_docs_result = getNearest(converted_vector, k_in)
nearest_docs_result = json.loads(nearest_docs_result.documents)
```


![png](https://user-images.githubusercontent.com/19545678/60756243-e4cfe580-a018-11e9-87a4-6eba716437c8.png)



```python
# See if the labels are the same
for doc in nearest_docs_result:
    # ids are printed to make sure that the resulting documents are different
    print(doc["id"], doc["doc"]["label"])
```

    47522d089f986f4571faaa2782ff59e7 7
    ef24c3659e1c6faa66b8a9b029327d48 7
    e574ccd989fc0d533e5c694e21916284 7
    d094352a6c22702ad681396bb0be33b9 7
    6e2d8145854453cee0d340ee237381dc 7



```python
# see if the images are correct
for doc in nearest_docs_result:
    image_ = doc["doc"]["vector"]
    label_ = doc["doc"]["label"]
    # draw image
    label = label_
    pixels = np.resize(np.array(image_),(28,28))
    plt.title('Label is {label}'.format(label=label))
    plt.imshow(pixels, cmap='gray')
    plt.show()
```


![png](https://user-images.githubusercontent.com/19545678/60756244-e699a900-a018-11e9-9f07-6abe22752845.png)



![png](https://user-images.githubusercontent.com/19545678/60756248-e7cad600-a018-11e9-86ee-4cd91bbe4f1a.png)



![png](https://user-images.githubusercontent.com/19545678/60756249-e8fc0300-a018-11e9-8930-0e944416bbcb.png)



![png](https://user-images.githubusercontent.com/19545678/60756250-ea2d3000-a018-11e9-8450-ea1543beeabb.png)



![png](https://user-images.githubusercontent.com/19545678/60756251-eac5c680-a018-11e9-90c3-4d0104cc47c2.png)


That's all for this tutorial. Thanks, happy hacking..!