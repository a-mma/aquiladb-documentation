---
id: reverse-image-search
title: Perform Direct and Reverse image search like Google Images
sidebar_label: Build a toy Google reverse image search
---
# Perform Direct and Reverse image search like Google Images

_**Download Jupyter notebook [here](https://github.com/a-mma/AquilaDB-Examples/tree/master/2way_image_search)**_


In this tutorial, we will be looking at how multi-model search can be done in AquilaDB. We will build a tool similar to Google Image search and we will be performing direct (text to image) and reverse (image to image) search with the help of two pretrained models - one is for text and the other one for image.

To make things faster and easier, will be using a `Fasttext` model for sentence embedding and a `MobileNet` model for image encoding.

This tutorial will be fast and will skim some unwanted details in code. If you find it hard to follow, please refer to previous tutorials where we take more time to discuss those details in the code.

So, Let's begin..


### Prerequisites

Install and import all required python libraries (we will be installing & importing AquilaDb library later).


```python
!pip install fasttext
!pip install Pillow
!pip install matplotlib
!pip install "tensorflow_hub==0.4.0"

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow_hub as hub

import pandas as pd
import fasttext as ft
```


### Load Flickr30k images dataset

You need to download [Flickr Image captioning dataset](https://www.kaggle.com/hsankesara/flickr-image-dataset) and extract it to a convenient location. We have extracted it into a directory `./flickr30k_images/` which is in the same directory as this notebook.

Load results.csv file as a Pandas dataframe - which contains all the captions along with file names of each image.


```python
# read image descriptions
image_descriptions = pd.read_csv('./flickr30k_images/results.csv', sep='\|\s', engine='python')
selected_columns = ['image_name', 'comment']
image_descriptions = image_descriptions[selected_columns]
```

### Train and load Fasttext Model

Now let's quickly build a Fasttext language model from the raw comments that we have.

To make things easy, we already have extracted all the comments from the CSV file to a text file - `results.txt`.
Let's train the Fasttext model on our data in skip-gram unsupervised mode.


```python
# create a language model quickly with fasttext
fasttext_model = ft.train_unsupervised(model='skipgram', input='flickr30k_images/results.txt')
# save model
fasttext_model.save_model("ftxt_model.bin")
```


```python
# load saved model
fasttext_model = ft.load_model("ftxt_model.bin")
```

​    


Verify that the model encodes the semantic information for different words properly.

Note that, fasttext is not good for encoding semantic information for sentences. We are using it here, because we expect the user to search images by giving importance to the words - resulting each object in the image rather than the overall context of the image.

In case you wanted semantic sentence based retrieval, feel free to use better language models (slower than Fasttext) like Universal Sentence Encoder. We have a tutorial on that [over here](https://github.com/a-mma/AquilaDB/wiki/Semantic-Text-Retrieval).


```python
# test the language model
! echo "girl" | fasttext nn ftxt_model.bin
! echo "===============" 
! echo "garden" | fasttext nn ftxt_model.bin
! echo "===============" 
! echo "glass" | fasttext nn ftxt_model.bin
! echo "===============" 
! echo "ball" | fasttext nn ftxt_model.bin
```

    Pre-computing word vectors... done.
    Query word? little 0.81607
    child 0.749877
    Girl 0.730085
    pink 0.729028
    Little 0.728659
    boy 0.721146
    young 0.70541
    Child 0.696403
    blond 0.69241
    pigtails 0.683836
    ...
    Query word? ===============
    Pre-computing word vectors... done.
    Query word? t-ball 0.850855
    T-ball 0.842449
    ballgame 0.822507
    A&M 0.745541
    Tennis 0.731484
    Rugby 0.726965
    rugby 0.719968
    33 0.719124
    defends 0.716951
    racquet 0.71034
    Query word? 

Just in case you wonder how we generate sentence embedding from Fasttext, here's a one-liner to do that.


```python
# convert string to embeddings
fasttext_model.get_sentence_vector('a cat is sitting on the carpet')
```




    array([ 3.55252177e-02,  4.62995056e-04, -5.44314571e-02, -3.67470682e-02,
            5.60869165e-02, -8.12834278e-02,  3.80968209e-03, -2.74911691e-02,
            ...
            5.96124977e-02, -1.29236341e-01,  5.84035628e-02,  1.21095881e-01,
            5.16762286e-02,  1.02854759e-01, -1.47027825e-03, -1.08863831e-01],
          dtype=float32)



### cleanup data (dataframe)

Before we proceed into the core of this tutorial, we need to cleanup the dataframe to keep only what we wanted. The code below is self explanatory, if you have a background knowledge using Pandas. We are skipping the explanation just because it is out of scope of this tutorial.


```python
def concater(x):
    try:
        return ' '.join(x)
    except Exception as e:
        return ''

# concatenate strings for same images
image_descriptions['comment'] = image_descriptions.groupby(['image_name'])['comment'].transform(concater)
image_descriptions = image_descriptions[['image_name','comment']].drop_duplicates()
image_descriptions.head(4)
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>image_name</th>
      <th>comment</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1000092795.jpg</td>
      <td>Two young guys with shaggy hair look at their ...</td>
    </tr>
    <tr>
      <th>5</th>
      <td>10002456.jpg</td>
      <td>Several men in hard hats are operating a giant...</td>
    </tr>
    <tr>
      <th>10</th>
      <td>1000268201.jpg</td>
      <td>A child in a pink dress is climbing up a set o...</td>
    </tr>
    <tr>
      <th>15</th>
      <td>1000344755.jpg</td>
      <td>Someone in a blue shirt and hat is standing on...</td>
    </tr>
  </tbody>
</table>



```python
# verify comments in each row
print(image_descriptions.iloc[0][0], image_descriptions.iloc[0][1])
print(image_descriptions.iloc[1][0], image_descriptions.iloc[1][1])
print(image_descriptions.iloc[500][0], image_descriptions.iloc[500][1])
```

    1000092795.jpg Two young guys with shaggy hair look at their hands while hanging out in the yard . Two young , White males are outside near many bushes . Two men in green shirts are standing in a yard . A man in a blue shirt standing in a garden . Two friends enjoy time spent together .
    
    10002456.jpg Several men in hard hats are operating a giant pulley system . Workers look down from up above on a piece of equipment . Two men working on a machine wearing hard hats . Four men on top of a tall structure . Three men on a large rig .
    
    1159425410.jpg A female washes her medium-sized dog outdoors in a plastic container while a friend secures it with a leash . A brown dog is in a blue tub , while one person holds his leash and another is soaping him . Two people give a dog a bath outdoors in a blue container . A small brown dog is being washed in a small blue bin . A dog calmly waits until his bath is over .


### Load pretrained MobileNet Model

Now we need to load pretrained MobileNet model from Tensorflow Hub. 


```python
# load mobilenet featurevector model as a Keras layer
module = tf.keras.Sequential([
    hub.KerasLayer("https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4", 
        output_shape=[1280],
        trainable=False)
])

# build the model
module.build([None, 224, 224, 3])

# This model will only accept images of size 224 x 224
# So, we need to make sure throughout the code, that we supply correcty resized images
im_height, im_width = 224, 224
```


### Helper functions

Here are some self explanatory helper functions that will help us during the embed/encode/predict stages.


```python
# Here is the helper function to load and resize image
def load_rsize_image(filename, w, h):
    # open the image file
    im = Image.open(filename)
    # resize the image
    im = im.resize(size=(w, h))
    return np.asarray(im)
```


```python
# Let's test loading an image
image_array = load_rsize_image('./flickr30k_images/flickr30k_images/301246.jpg', im_width, im_height)
plt.imshow(image_array)
```




![png](https://user-images.githubusercontent.com/19545678/62862203-095b6380-bd23-11e9-80b2-96ba622ec514.png)



```python
# helper function to retrieve fasttext word embeddings
def get_ftxt_embeddings(text):
    return fasttext_model.get_sentence_vector(text)

# helper function to encode images with mobilenet
def get_image_encodings(batch, module):
    message_embeddings = module.predict(batch)
    return message_embeddings
```


```python
# helper function to embed images and comments in a dataframe and return numpy matrices
# this function will iterate through a dataframe, which contains image file names in one column and
# comments in another column and will generate separate matrices for images and comments.
# row order of these matrices matters because same row index in both matrices represent related image and comments.
def embed_all(df, w, h):
    img_arr = []
    txt_arr = []
    # for each row, embed data
    for index, row in df.iterrows():
        # img_arr will contain all the image file data (will be passed to mobilenet later)
        img_arr.append(load_rsize_image('./flickr30k_images/flickr30k_images/' + row['image_name'], w, h))
        # txt_arr will contain all Fasttext sentance embedding for each comment 
        txt_arr.append(get_ftxt_embeddings(row['comment']))
    return img_arr, txt_arr
```


```python
img_emb, txt_emb = embed_all(image_descriptions, im_width, im_height)
# reset fasttext model
fasttext_model = None
```


```python
# verify that image is image loded correctly
plt.imshow(img_emb[2])
```




![png](https://user-images.githubusercontent.com/19545678/62862204-09f3fa00-bd23-11e9-88d6-d8ff52ffec20.png)


In above steps, we have embedded text data with Fasttext. Image data still need to be encoded. To keep the CPU and RAM away from exploding, we decided to do it in batches, before sending them to AquilaDB.

But just in case you wonder how an image can be encoded, here is a one-liner for that:


```python
# test image encodings generation
get_image_encodings(np.true_divide(np.array(img_emb[0:100]), 255), module).shape
```




    (100, 1280)



### Filter based indexing

This is the core idea we wanted to share with you through this tutorial.
In this tutorial, we are using multiple models that generate encodings. So we need to index both of them inside AquilaDB and need to somehow discriminate (filter) them during k-NN search. With AquilaDB we could do this efficiently.

Padding can be done in two ways:
1. Positional padding
2. Filter vector padding

#### Positional Padding

This is what we will be doing in this tutorial.
If you have a limited number of models ranging between 2 to 4, this will be the best method that you can use.

Suppose, we have two models `M1` and `M2`. And these models generate vectors `v1` and `v2`.
Then we will build two long vectors `vlong` as, `size(vlong) = size(v1) + size(v2)` for each models.

Then we will pad each of them with either preceding or following zeroes.

Example:

v1 = [1, 2, 3, 4, 5]

v2 = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

then; size(vlong) = 5 + 10 = 15

So, we will be sending two vectors to AquilaDB, each of them are:

v1long = [1, 2, 3, 4, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

v2long = [0, 0, 0, 0, 0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

#### Filter vector padding

If you have more than 4 models, we highly recommend you to use a better Machine Learning model that combine all of these and then use `Positional Padding`. But, of course there might be requirements apart from that, then use this method.

Consider designing filter vectors for each model. For example, we have two models M1 and M2. And these models generate vectors v1 and v2. Then, design two filter vectors f1 and f2 as,

f1 = [0, 0, 0, 0, 0, 0, ........ n items]

f1 = [1, 1, 1, 1, 1, 1, ........ n items]

value of `n` is a variable should be chosen to maximize the distance between two filters.

So, we will be sending two vectors to AquilaDB, each of them are:

v1long = append(f1, v1)

v2long = append(f2, v2)

### Send data to AquilaDB for indexing


```python
# install AquilaDb python client

! pip install aquiladb

# import AquilaDB client
from aquiladb import AquilaClient as acl
```



```python
# create DB instance.
# Please provide the IP address of the machine that have AquilaDB installed in.
db = acl('192.168.1.102', 50051)

# let's get our hands dirty for a moment..
# convert a sample dirty Document
sample = db.convertDocument([0.1,0.2,0.3,0.4], {"hello": "world"})
```


```python
# and print it
sample
```




    {'vector': {'e': [0.1, 0.2, 0.3, 0.4]}, 'b64data': b'{"hello":"world"}'}



As you can see above, this is what happens when a document along with a vector is serialized. This will then be sent to AquilaDB.

##### add documents to AquilaDB

In the code below we do a lot of things. So, please pay attention to the comments to see how it is done.


```python
# We are going to encode a small portion (6000) images/text that we have downloaded.
# You can add more if you have got enough interest, patience and a good machine.

# batch length - to be sent to mobilenet for encoding
blen = 500
# which index to start encoding - ofcause its 0
vstart = 0
# How much images/text we need to encode
vend = 6000

# convert text embeddings to numpy array
txt_emb = np.array(txt_emb)

# iterate over each batch of image/text data/embedding
for ndx in range(vstart, vend, blen):
    # encode each batch of images
    image_encoding = get_image_encodings(np.true_divide(np.array(img_emb[ndx:ndx+blen]), 255), module)
    
    # pad image and text vectors - this is discussed in section `filter based indexing`
    # select subset of data we're interested for text embeddings
    text_embedding = txt_emb[ndx:ndx+blen]
    # pad text encodings with trailing zeros
    text_embedding = np.pad(text_embedding, ((0, 0), (0, 1280)), 'constant')
    # pad image encodings with preceding zeros
    image_encoding = np.pad(image_encoding, ((0, 0), (100, 0)), 'constant')
    
    # finally, create and send each document
    for i in range(blen):
        # create document - text
        doc_txt = db.convertDocument(text_embedding[i], {"image_name": image_descriptions.iloc[ndx+i][0]})
        # create document - image
        doc_img = db.convertDocument(image_encoding[i], {"image_name": image_descriptions.iloc[ndx+i][0]})
        
        # send documents - text
        db.addDocuments([doc_txt])
        # send documents - image
        db.addDocuments([doc_img])
    
    # Wooh! done with nth batch   
    print('Done: ', ndx, ndx+blen)
```

    Done:  0 500
    Done:  500 1000
    ...
    Done:  5500 6000


### Show off final results

Yeah, we have indexed all our images and texts in AquilaDB. Now it's time to retrieve them either by text search or by image search.

#### search images by text


```python
import json 

# search by text
def search_by_text(text_in):
    # load saved model
    fasttext_model = ft.load_model("ftxt_model.bin")
    # generate embeddings
    text_embedding_ = fasttext_model.get_sentence_vector(text_in)
    # pad text embedding
    text_embedding_ = np.pad([text_embedding_], ((0, 0), (0, 1280)), 'constant')

    # convert query matrix
    q_matrix = db.convertMatrix(np.asarray(text_embedding_[0]))
    # do k-NN search
    k = 10
    result = db.getNearest(q_matrix, k)
    return json.loads(result.documents)

# render images
def render_images(doclist):
    for doc in doclist:
        filename = doc["doc"]["image_name"]
        im = Image.open('./flickr30k_images/flickr30k_images/' + filename)
        fig = plt.figure()
        plt.imshow(im)
```

#### text to image search 1


```python
render_images(search_by_text('people sitting on bench'))
```

​    



![png](https://user-images.githubusercontent.com/19545678/62862205-09f3fa00-bd23-11e9-9df8-cf77190a3170.png)



![png](https://user-images.githubusercontent.com/19545678/62862206-09f3fa00-bd23-11e9-8a00-8401aa2439c8.png)



![png](https://user-images.githubusercontent.com/19545678/62862207-0a8c9080-bd23-11e9-9597-9d2f9e114f7c.png)



![png](https://user-images.githubusercontent.com/19545678/62862209-0a8c9080-bd23-11e9-9bf1-82b4ff26abda.png)



![png](https://user-images.githubusercontent.com/19545678/62862210-0b252700-bd23-11e9-9cb8-a022130fd20b.png)



![png](https://user-images.githubusercontent.com/19545678/62862212-0b252700-bd23-11e9-81d6-4005dfde7f27.png)



![png](https://user-images.githubusercontent.com/19545678/62862213-0b252700-bd23-11e9-8c1b-ad99dee2c529.png)



![png](https://user-images.githubusercontent.com/19545678/62862215-0bbdbd80-bd23-11e9-8b3e-14dd742e222e.png)



![png](https://user-images.githubusercontent.com/19545678/62862216-0bbdbd80-bd23-11e9-9b02-0160d2430f76.png)



![png](https://user-images.githubusercontent.com/19545678/62862217-0bbdbd80-bd23-11e9-9248-49cc33ae2060.png)


#### text to image search 2


```python
render_images(search_by_text('kids playing in garden'))
```

​    



![png](https://user-images.githubusercontent.com/19545678/62862218-0c565400-bd23-11e9-87f6-b88bf9d2d5c8.png)



![png](https://user-images.githubusercontent.com/19545678/62862221-0c565400-bd23-11e9-9d20-9728f05e843c.png)



![png](https://user-images.githubusercontent.com/19545678/62862222-0ceeea80-bd23-11e9-8389-16cb88fbc345.png)



![png](https://user-images.githubusercontent.com/19545678/62862223-0ceeea80-bd23-11e9-884b-4eda05d2d1b3.png)



![png](https://user-images.githubusercontent.com/19545678/62862224-0ceeea80-bd23-11e9-920c-6d0da4df4f4d.png)



![png](https://user-images.githubusercontent.com/19545678/62862225-0d878100-bd23-11e9-9c44-0178b4695f38.png)



![png](https://user-images.githubusercontent.com/19545678/62862226-0d878100-bd23-11e9-9797-b62dc6d4bc5e.png)



![png](https://user-images.githubusercontent.com/19545678/62862227-0d878100-bd23-11e9-88d8-08bd8e52d83c.png)



![png](https://user-images.githubusercontent.com/19545678/62862228-0e201780-bd23-11e9-919a-5453dc435dda.png)



![png](https://user-images.githubusercontent.com/19545678/62862229-0e201780-bd23-11e9-926c-9cf3ef4748b0.png)


#### text to image search 3


```python
render_images(search_by_text('man riding a bike'))
```

​    



![png](https://user-images.githubusercontent.com/19545678/62862230-0e201780-bd23-11e9-8907-8a640771c409.png)



![png](https://user-images.githubusercontent.com/19545678/62862233-0eb8ae00-bd23-11e9-90b0-3fd1cdfe8f85.png)



![png](https://user-images.githubusercontent.com/19545678/62862234-0eb8ae00-bd23-11e9-8a65-2c9a13037375.png)



![png](https://user-images.githubusercontent.com/19545678/62862236-0eb8ae00-bd23-11e9-9d90-24b90c37afad.png)



![png](https://user-images.githubusercontent.com/19545678/62862237-0f514480-bd23-11e9-9383-ef404f98a3c3.png)



![png](https://user-images.githubusercontent.com/19545678/62862238-0f514480-bd23-11e9-83ee-9b0ab49b7b66.png)



![png](https://user-images.githubusercontent.com/19545678/62862241-0fe9db00-bd23-11e9-8536-853b69953d1a.png)



![png](https://user-images.githubusercontent.com/19545678/62862242-0fe9db00-bd23-11e9-8ac0-863d4c23f725.png)



![png](https://user-images.githubusercontent.com/19545678/62862243-0fe9db00-bd23-11e9-918b-bfa2adefd25a.png)



![png](https://user-images.githubusercontent.com/19545678/62862244-10827180-bd23-11e9-91c9-dc793e9e8a1e.png)


#### search images by image


```python
# search by image
def search_by_image(image_in, w, h, module):
    # load image
    q_image = load_rsize_image('./flickr30k_images/flickr30k_images/' + image_in, w, h)
    q_image = np.array([np.asarray(q_image)])
    # generate encodings
    image_encoding_ = get_image_encodings(np.true_divide(q_image, 255), module)
    # pad image encodings
    image_encoding_ = np.pad(image_encoding_, ((0, 0), (100, 0)), 'constant')

    # convert query matrix
    q_matrix = db.convertMatrix(np.asarray(image_encoding_[0]))
    # do k-NN search
    k = 10
    result = db.getNearest(q_matrix, k)
    return json.loads(result.documents)
```

#### image to image search 1


```python
q_im_file = '134206.jpg'

# show query image
render_images([{"doc":{"image_name": q_im_file}}])
```


![png](https://user-images.githubusercontent.com/19545678/62862245-10827180-bd23-11e9-8d3b-8d962dd31c35.png)



```python
# do search
render_images(search_by_image(q_im_file, im_width, im_height, module))
```


![png](https://user-images.githubusercontent.com/19545678/62862246-10827180-bd23-11e9-8f2c-5564cb5f30c2.png)



![png](https://user-images.githubusercontent.com/19545678/62862247-111b0800-bd23-11e9-8046-575126d22569.png)



![png](https://user-images.githubusercontent.com/19545678/62862249-111b0800-bd23-11e9-8455-aeb597c193ee.png)



![png](https://user-images.githubusercontent.com/19545678/62862250-111b0800-bd23-11e9-9fce-88a34a343e25.png)



![png](https://user-images.githubusercontent.com/19545678/62862251-11b39e80-bd23-11e9-93e1-c73ecd2ede95.png)



![png](https://user-images.githubusercontent.com/19545678/62862252-11b39e80-bd23-11e9-979e-efd85cfb7584.png)



![png](https://user-images.githubusercontent.com/19545678/62862253-124c3500-bd23-11e9-8cf9-acfd579cd190.png)



![png](https://user-images.githubusercontent.com/19545678/62862255-124c3500-bd23-11e9-8eb5-d6349f6c12f7.png)



![png](https://user-images.githubusercontent.com/19545678/62862258-124c3500-bd23-11e9-94f9-5a979cd0fd43.png)



![png](https://user-images.githubusercontent.com/19545678/62862259-12e4cb80-bd23-11e9-8d1c-9393b0703ec5.png)


#### image to image search 2


```python
q_im_file = '11808546.jpg'

# show query image
render_images([{"doc":{"image_name": q_im_file}}])
# do search
render_images(search_by_image(q_im_file, im_width, im_height, module))
```


![png](https://user-images.githubusercontent.com/19545678/62862260-12e4cb80-bd23-11e9-8f66-ea5ca921d5bb.png)



![png](https://user-images.githubusercontent.com/19545678/62862261-12e4cb80-bd23-11e9-841e-852ee76a9373.png)



![png](https://user-images.githubusercontent.com/19545678/62862263-137d6200-bd23-11e9-8748-bce914750868.png)



![png](https://user-images.githubusercontent.com/19545678/62862264-137d6200-bd23-11e9-8139-d7ae6e3b9193.png)



![png](https://user-images.githubusercontent.com/19545678/62862265-137d6200-bd23-11e9-91b5-728c7c4ae821.png)



![png](https://user-images.githubusercontent.com/19545678/62862266-1415f880-bd23-11e9-8972-eb976ef0126b.png)



![png](https://user-images.githubusercontent.com/19545678/62862268-1415f880-bd23-11e9-9f81-35aa1a5e796e.png)



![png](https://user-images.githubusercontent.com/19545678/62862270-14ae8f00-bd23-11e9-8f30-42751208e888.png)



![png](https://user-images.githubusercontent.com/19545678/62862271-14ae8f00-bd23-11e9-835f-2deed4f7abfd.png)



![png](https://user-images.githubusercontent.com/19545678/62862272-14ae8f00-bd23-11e9-86ad-c0d68344f96d.png)



![png](https://user-images.githubusercontent.com/19545678/62862273-15472580-bd23-11e9-9ddc-6a006b4871a4.png)


#### image to image search 3


```python
q_im_file = '14526359.jpg'

# show query image
render_images([{"doc":{"image_name": q_im_file}}])
# do search
render_images(search_by_image(q_im_file, im_width, im_height, module))
```


![png](https://user-images.githubusercontent.com/19545678/62862275-15472580-bd23-11e9-9e82-d13f7882d4ba.png)



![png](https://user-images.githubusercontent.com/19545678/62862276-15472580-bd23-11e9-9589-a00c56283334.png)



![png](https://user-images.githubusercontent.com/19545678/62862277-15dfbc00-bd23-11e9-9721-46d745c77464.png)



![png](https://user-images.githubusercontent.com/19545678/62862278-15dfbc00-bd23-11e9-9938-56bbcd2b9fc0.png)



![png](https://user-images.githubusercontent.com/19545678/62862280-15dfbc00-bd23-11e9-9041-8ab6787a2c87.png)



![png](https://user-images.githubusercontent.com/19545678/62862281-16785280-bd23-11e9-938d-8dcc81a02d72.png)



![png](https://user-images.githubusercontent.com/19545678/62862283-16785280-bd23-11e9-9afc-de85252102a1.png)



![png](https://user-images.githubusercontent.com/19545678/62862285-1710e900-bd23-11e9-831e-a159ad44e9e2.png)



![png](https://user-images.githubusercontent.com/19545678/62862286-1710e900-bd23-11e9-9fb8-c4c9c8a4a44d.png)



![png](https://user-images.githubusercontent.com/19545678/62862287-1710e900-bd23-11e9-9613-3234822685a8.png)



![png](https://user-images.githubusercontent.com/19545678/62862288-17a97f80-bd23-11e9-8e55-f8a6886ad77b.png)


#### image to image search 4


```python
q_im_file = '21164875.jpg'

# show query image
render_images([{"doc":{"image_name": q_im_file}}])
# do search
render_images(search_by_image(q_im_file, im_width, im_height, module))
```


![png](https://user-images.githubusercontent.com/19545678/62862289-17a97f80-bd23-11e9-8751-9fe27309d48d.png)



![png](https://user-images.githubusercontent.com/19545678/62862291-17a97f80-bd23-11e9-810d-97c001dfa980.png)



![png](https://user-images.githubusercontent.com/19545678/62862292-18421600-bd23-11e9-9201-8491cdeffd5e.png)



![png](https://user-images.githubusercontent.com/19545678/62862293-18421600-bd23-11e9-9cab-6e93122ccda9.png)



![png](https://user-images.githubusercontent.com/19545678/62862294-18421600-bd23-11e9-99b6-4df07b3d423f.png)



![png](https://user-images.githubusercontent.com/19545678/62862295-18daac80-bd23-11e9-987a-2212cd2dce59.png)



![png](https://user-images.githubusercontent.com/19545678/62862296-19734300-bd23-11e9-8b65-920890a646fb.png)



![png](https://user-images.githubusercontent.com/19545678/62862297-19734300-bd23-11e9-90e5-94afb0c0942f.png)



![png](https://user-images.githubusercontent.com/19545678/62862298-19734300-bd23-11e9-8463-de29c3df0edc.png)



![png](https://user-images.githubusercontent.com/19545678/62862300-1a0bd980-bd23-11e9-80df-c230c2109f34.png)



![png](https://user-images.githubusercontent.com/19545678/62862301-1a0bd980-bd23-11e9-8c4f-47beaae6117a.png)


#### image to image search 5


```python
q_im_file = '23008340.jpg'

# show query image
render_images([{"doc":{"image_name": q_im_file}}])
# do search
render_images(search_by_image(q_im_file, im_width, im_height, module))
```


![png](https://user-images.githubusercontent.com/19545678/62862302-1aa47000-bd23-11e9-9ee5-c90c6c4407ea.png)



![png](https://user-images.githubusercontent.com/19545678/62862303-1aa47000-bd23-11e9-9ddb-9b0c5dd329cf.png)



![png](https://user-images.githubusercontent.com/19545678/62862304-1b3d0680-bd23-11e9-80c5-cf0faf9a5ff8.png)



![png](https://user-images.githubusercontent.com/19545678/62862305-1b3d0680-bd23-11e9-88cb-467245055d75.png)



![png](https://user-images.githubusercontent.com/19545678/62862306-1b3d0680-bd23-11e9-9353-70d95aecfb4d.png)



![png](https://user-images.githubusercontent.com/19545678/62862307-1bd59d00-bd23-11e9-8219-4519b1205736.png)



![png](https://user-images.githubusercontent.com/19545678/62862308-1bd59d00-bd23-11e9-9063-cd7de96ed952.png)



![png](https://user-images.githubusercontent.com/19545678/62862309-1bd59d00-bd23-11e9-85ba-d8d2b090051e.png)



![png](https://user-images.githubusercontent.com/19545678/62862311-1c6e3380-bd23-11e9-9094-320a1de2beb8.png)



![png](https://user-images.githubusercontent.com/19545678/62862312-1c6e3380-bd23-11e9-83ed-87fe7b640c59.png)



![png](https://user-images.githubusercontent.com/19545678/62862313-1c6e3380-bd23-11e9-8d7f-3e4a4bbb9e8b.png)


That's all for this tutorial. Thanks, happy hacking..!