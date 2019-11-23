---
id: api-reference
title: Client API reference 
sidebar_label: Client API reference
---

## gRPC
Available RPC services:

```
addDocuments (addDocRequest) returns (addDocResponse)
deleteDocuments (deleteDocRequest) returns (deleteDocResponse)
addNode (addNodeRequest) returns (addNodeResponse)
getNearest (getNearestRequest) returns (getNearestResponse)
```
Request - Response structures:

```
addDocRequest {
    repeated singleDocType documents
}

addDocResponse {
    bool status
    repeated string _id
}

deleteDocRequest {
    repeated singleDocType documents
}

deleteDocResponse {
    bool status
    repeated string _id
}

addNodeRequest {
    repeated string peers
}

addNodeResponse {
    bool status
    repeated string peers
}


getNearestRequest {
    repeated vectorType
    int32 k
}

getNearestResponse {
    bool status
    bytes dist_matrix
    bytes documents
}
```

Custom data types:

```
vectorType {
    repeated float e
}


singleDocType {
    string _id
    vectorType vector
    bytes b64data
}
```

[Here](https://github.com/a-mma/AquilaDB/blob/develop/src/proto/vecdb.proto) is the `.proto` file for detailed reference.

## Python

Here are the API methods exposed by [python client library](https://github.com/a-mma/AquilaDB-Python):

```
class AquilaClient (string root_url, string port)
# client constructor

method convertDocument(list vector, Object document)
# helper function to convert native data to API friendly data

method addDocuments (Object documents_in) 
# API interface to add documents to AquilaDB

method convertMatrix(list vector)
# helper function to convert native data to API friendly data

method getNearest (Object matrix_in, int k_in)
# API interface to get nearest documents from AquilaDB
```

## Node JS
[(Development in progres)](https://github.com/a-mma/AquilaDB-NodeJS)