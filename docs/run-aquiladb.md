---
id: run-aquiladb
title: Run- AquilaDB
sidebar_label: Run AquilaDB
---
To deploy and run AquilaDB, run in terminal:

```
docker run -d -i -p 50051:50051 -v "<SPECIFY DATA PERSIST LOCATION>:/data" -t ammaorg/aquiladb:<IMAGE LABEL>
```
