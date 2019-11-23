---
id: neural-information-retrieval
title: Neural Information retrieval
sidebar_label: Neural Information retrieval
---

# What is Neural Information Retrieval
Neural Information Retrieval is the application of shallow or deep neural networks to Information Retrieval tasks. Neural information retrieval highly relies on the dimensionality reduction capability of Neural Networks. 

Auto encoding techniques have proven this by letting the neural networks to learn the generic vector encodings of any kind of data through both supervised and unsupervised manner. Transfer learning inspired by a similar idea is now a foundational process in training deep neural networks.

# Why is Neural Information Retrieval
Traditional Information Retrieval systems required problem-specific feature engineering. The software architecture is highly reliant on the type of data the system is processing. This reduced the adaptability of the existing system to new kinds of data. This also demands both time and resources being spent on workarounds for adaptability. 

With pre-trained Deep Learning models, it is now an easy task. All you need to do is, encode each data you have with a pre-trained ML model into its latent vector and send it to AquilaDB along with the metadata description of that vector. Once you have done indexing enough data, you can send your query vector to AquilaDB to retrieve similar vectors along with their metadata. This way, you get a highly flexible system that exposes a unified API interface for all kinds of data.

![neural information retrieval process](https://user-images.githubusercontent.com/19545678/68528845-fa824e00-031d-11ea-8988-04299ec37d54.png)

# Learn more
We have started meeting developers and do small talks on AquilaDB. Here are the slides that we use on those occasions: http://bit.ly/AquilaDB-slides 

As of current AquilaDB release features, you can build **[Neural Information Retrieval](https://www.microsoft.com/en-us/research/uploads/prod/2017/06/INR-061-Mitra-neuralir-intro.pdf)** applications out of the box without any external dependencies. Here are some useful links to learn more about it and start building:

* These use case examples will give you an understanding of what is possible and what not: https://github.com/a-mma/AquilaDB/wiki
* Microsoft published a paper and youtube video on this to onboard anyone interested: 
  * paper: https://www.microsoft.com/en-us/research/uploads/prod/2017/06/INR-061-Mitra-neuralir-intro.pdf
  * video: https://www.youtube.com/watch?v=g1Pgo5yTIKg
* Embeddings for Everything: Search in the Neural Network Era: https://www.youtube.com/watch?v=JGHVJXP9NHw
* Autoencoders are one such deep learning algorithms that will help you to build semantic vectors - foundation for Neural Information retrieval. Here are some links to Autoencoders based IR:
  * go to chapter 15 in this link: https://www.cs.toronto.edu/~hinton/coursera_lectures.html
  * https://www.coursera.org/lecture/ml-foundations/examples-of-document-retrieval-in-action-CW25H
  * https://www.coursera.org/lecture/intro-to-deep-learning/autoencoders-101-QqBOa
* Note that, the idea of information retrieval applies not only to text data but for any data. All you need to do is, encode any source datatype to a dense vector with deep neural networks.
