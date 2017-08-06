import json
import numpy as np
import os
from PIL import Image
from keras.layers import Embedding
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

EMBEDDING_DIM = 50
tokenizer = Tokenizer()
np.set_printoptions(threshold=5)


def load_data(path):
    ''' This gets the relation data from the json file for each image
        =    an ID code
           + true and false statements about each image
           + whether the statement is true or false (1 or 0)
        e.g., ['1304-0', 'There is a circle closely touching a corner of a box.', 1].
        There are 12410 items (sentences) in nlvr/train/train.json.
    '''
    f = open(path, 'r')
    data = []
    for l in f:
        jn = json.loads(l)
        s = jn['sentence']
        idn = jn['identifier']
        la = int(jn['label'] == 'true')
        data.append([idn, s, la])
    return data


def init_tokenizer(sdata):
    '''The keras tokenizer class vectorizes a text corpus by turning each
        sentence into either a sequence of integers (each integer being the index
        of a token (word) in a dictionary) or into a vector where the coefficient
        for each token could be binary, based on word count, based on tf-idf...
        {'colored': 206, 'bottom': 41, 'they': 115, 'only': 28, 'tower': 11, etc...'''
    texts = [t[1] for t in sdata]
    tokenizer.fit_on_texts(texts)


def tokenize_data(sdata, mxlen):
    '''Converts each word in data to a dictionary where the key is sentence ID (e.g., '1304-0'),
       and the value is a list of two items: 1) the n integer vector mxlen long

       ['1304-0', 'There is a circle closely touching a corner of a box.', 1]
       seqs (before padding):   [1, 3, 2, 30, 32, 15, 2, 58, 13, 2, 6]
      (these are simply the indices of each word in the list of all words in the dataset.
      This array is then padded with 0's, and then the id ('1304-1') is
       added along with the true/false (0 or 1) value at the end.
    example data item:
    ['11-2': [array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 1, 16, 25, 61, 4, 25, 8, 21], dtype=int32), 1]
    '''
    texts = [t[1] for t in sdata]
    ids = [t[0] for t in sdata]
    seqs = tokenizer.texts_to_sequences(texts)
    seqs = pad_sequences(seqs, mxlen)
    data = {}
    for k in range(len(sdata)):
        data[sdata[k][0]] = [seqs[k], sdata[k][2]]
    return data


def load_images(path, sdata, w, h, debug=False, split_images=False):
    data = {}
    cnt = 0
    N = 333
    for lists in os.listdir(path)[1:]:   # jameson: on Mac >DS_Store is the first "directory"
        p = os.path.join(path, lists)
        for f in os.listdir(p):
            cnt += 1
            if debug and cnt > N:
                break
            im_path = os.path.join(p, f)
            # im = Image.open(im_path)
            # im = im.convert('RGB')
            # im = im.resize((w,h))
            # im = im.rotate(90, expand=True)
            # im = np.array(im)
            # im = np.array(im).reshape((w,h,-1))  # im.resize() also transposes the array!
            # this reshape doesn't rotate the image properly because the color is mixed in
            # im = np.array(Image.open(im_path).convert('RGB').resize((w,h)).rotate(90, expand=True))
            im = np.array(Image.open(im_path).convert('RGB').resize((w,h)).transpose(Image.ROTATE_90))

            if split_images:
                # this is to separate out each of the tree sub-images
                w1 = int(1.5 * h)
                w2 = w1 + h
                w3 = w - h
                im0 = im[0:h]
                im1 = im[w1:w2]
                im2 = im[w3:]
                im = [im0,im1,im2]

            idf = f[f.find('-') + 1:f.rfind('-')]
            data[f] = [im] + sdata[idf]

    ims, ws, labels = [], [], []
    for key in data:
        ims.append(data[key][0])
        ws.append(data[key][1])
        labels.append(data[key][2])
    data.clear()
    idx = np.arange(0, len(ims), 1)
    np.random.shuffle(idx)
    ims = [ims[t] for t in idx]
    ws = [ws[t] for t in idx]
    labels = [labels[t] for t in idx]
    imgs = np.array(ims, dtype=np.float32)
    ws = np.array(ws, dtype=np.float32)
    labels = np.array(labels, dtype=np.float32)
    ''' imgs: image data (shape = (n_images, 50, 200, 3))  (3 for colors)
        ws:   image sentence expressed in word tokens (shape = (n_images, maxlen))
              (there is only one sentence per image; e.g., the red triangle of right over the blue square).
        labels: true or false (whether the sentence (ws) is a true of false statement about the image)
        The point of the key-value conditioning above is to get the words and labels. Which is 
        quite inefficient because translated back and forth.
        ims.shape = (4662, 50, 200, 3) = (no. of images_mini,  h, w, colors)
        ws.shape = (4662, 32) = (no. of images_mini,) = image sentence code (concat of
                                                   word index numbers, padded w/zeros)
        labels.shape = (4662,) = (no. of images_mini, )
        So we don't need the "labels" such as '11-2'.  
    '''
    return imgs, ws, labels


def get_embeddings_index(path):
    ''' this gets the word2vec vectors for a big file'''
    embeddings_index = {}
    f = open(path, 'r', errors='ignore')
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    return embeddings_index


def get_embedding_matrix(word_index, embeddings_index):
    embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector
    return embedding_matrix


def embedding_layer(word2vec_path, word_index, sequence_len):
    embedding_index = get_embeddings_index(word2vec_path)
    ''' prepare.get_embeddings_index() creates a dictionary with word strings 
        as keys and corresponding (50 dimensional) vectors as the values 
        (from glove.6B.50d.txt;  this has way more words than needed for nlvr)'''
    embedding_matrix = get_embedding_matrix(word_index, embedding_index)
    ''' This has shape (n_words, EMBEDDING_DIM),
        where n_words = number of words used for the nlvr task  
    '''
    return Embedding(len(word_index) + 1,
                     EMBEDDING_DIM,
                     weights=[embedding_matrix],
                     input_length=sequence_len,
                     trainable=False)
'''
For Embedding() see:  
    https://blog.keras.io/using-pre-trained-word-embeddings-in-a-keras-model.html

and: https://keras.io/layers/embeddings/
    
From keras code:
    Embedding(input_dim, output_dim, embeddings_initializer='uniform', 
              embeddings_regularizer=None, activity_regularizer=None, 
              embeddings_constraint=None, mask_zero=False, input_length=None)
    
    input_dim: int > 0. Size of the vocabulary, i.e. maximum integer index + 1.
    
    output_dim: int >= 0. Dimension of the dense embedding.
    
    weights: the input to this layer (input2 in train.py) is the vector of integers 
             representing the words indices for the input sentence, e.g., 
             input2 = (0, 0, 0, ..., 0, 12, 221, 4, 6).  Thus 
             K.dot(input2, embedding_matrix) = matrix = list of (word2vec) vectors for 
             corresponding to words in the sentence. The zeros are padding for
             sentences shorter than 32 words.  
        
    input_length: Length of input sequences, when it is constant. This argument is 
                  required if you are going to connect Flatten then Dense layers 
                  upstream (without it, the shape of 
                  the dense outputs cannot be computed).
    
    trainable: False because the embedding_matrix is constant - it was already learned 
               (by skip-gram e.g.) 
'''
