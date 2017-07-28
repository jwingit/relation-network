
import numpy as np
from keras.models import Model
from keras.layers import Dense, Dropout, Activation, Input, \
    LSTM, Bidirectional, Lambda, Add

from keras.layers.merge import concatenate

from keras.layers.convolutional import Conv2D, MaxPooling2D, AveragePooling2D
from keras.layers.normalization import BatchNormalization, regularizers
from keras.optimizers import Adam  # , RMSprop
import prepare
import pickle

# image size (scaled from the original data set):
w = 200
h = 50

debugg = 0  # sets it up for tiny databases so can debug easily
            # (need to create mini version of glove.6B.50d.txt)

np.random.seed(1)
mxlen = 32
embedding_dim = 50
lstm_unit = 64
MLP_unit = 128

epochs = 200

batch_size = 256
l2_norm = 1e-4
n_feat = 16


data_path = '/Users/jwjameson/Dropbox/Data/'
path = data_path + 'nlvr/'
train_json = path + 'train/train.json'
# train_img_folder = path + 'train/images_mini'
train_img_folder = path + 'train/images'
test_json = path + 'test/test.json'
test_img_folder = path + 'test/images'
word2vec_path = data_path + 'text_embedding/glove.6B.50d.txt'

if debugg:
    word2vec_path = data_path + 'text_embedding/glove.6B.50d.tiny.txt'

# for following word embedding method see:
# https://blog.keras.io/using-pre-trained-word-embeddings-in-a-keras-model.html

data = prepare.load_data(train_json)
''' This gets the relation data from the json file for each image 
    =    an ID code 
       + true and false statements about each image
       + whether the statement is true or false (1 or 0)
    e.g., ['1304-0', 'There is a circle closely touching a corner of a box.', 1].
    There are 12410 items (sentences) in nlvr/train/train.json.
'''

prepare.init_tokenizer(data)
'''This class allows to vectorize a text corpus, by turning each
    sentence into either a sequence of integers (each integer being the index
    of a token (word) in a dictionary) or into a vector where the coefficient
    for each token could be binary, based on word count, based on tf-idf...

    {'colored': 206, 'bottom': 41, 'they': 115, 'only': 28, 'tower': 11, etc...}
    '''
data = prepare.tokenize_data(data, mxlen)
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

imgs, ws, labels = prepare.load_images(train_img_folder, data, w, h, debug=debugg)
''' imgs: image data (shape = (n_images, 50, 200, 3))  (3 for colors)
    ws:   image sentence expressed in word tokens (shape = (n_images, maxlen))
          (there is only one sentence per image; e.g., the red triangle of right over the blue square).
    labels: true or false (whether the sentence (ws) is a true of false statement about the image)

Hmmmm, the whole point of the key-value conditioning above was
    to get the words and labels. Which is quite inefficient because translated back and forth.
    ims.shape = (4662, 50, 200, 3) = (no. of images_mini,  h, w, colors)
    ws.shape = (4662, 32) = (no. of images_mini,) = image sentence code (concat of
                                               word index numbers, padded w/zeros)
    labels.shape = (4662,) = (no. of images_mini, )
    So turns out you don't need the "labels" such as '11-2'.  Goofy way of doing it.
'''

data.clear()

# test_data = prepare.load_data(test_json)
# test_data = prepare.tokenize_data(test_data, mxlen)
# test_imgs, test_ws, test_labels = prepare.load_images(test_img_folder, test_data)
# test_data.clear()

print("finished reading training images...")

imgs_mean = np.mean(imgs)
imgs_std = np.std(imgs - imgs_mean)
imgs = (imgs - imgs_mean) / imgs_std
n_images = imgs.shape[0]
print("no. of images =", n_images)


# test_imgs = (test_imgs - imgs_mean) / imgs_std


def bn_layer(x, conv_unit):
    def f(inputs):
        md = Conv2D(x, conv_unit, padding='same', kernel_initializer='he_normal')(inputs)
        md = BatchNormalization()(md)
        return Activation('relu')(md)

    return f


def conv_net(inputs):
    model = bn_layer(16, 3)(inputs)
    model = MaxPooling2D((4, 4), 4)(model)
    model = bn_layer(16, 3)(model)
    model = MaxPooling2D((3, 3), 3)(model)
    model = bn_layer(32, 3)(model)
    model = MaxPooling2D((2, 2), 2)(model)
    model = bn_layer(32, 3)(model)
    return model


def Get_location_features(w, h):
    wh = 1.0 * np.ones((n_images, w, h, 2))
    for w_ in range(w):
        wh[:, w_, :, 0] *= w_ / w
    for h_ in range(h):
        wh[:, :, h_:, 1] *= h_ / h
    return wh

input1 = Input((w, h, 3))

input2 = Input((mxlen,))

cnn_features = conv_net(input1)
shapes = cnn_features.shape
wc, hc = int(shapes[1]), int(shapes[2])  # this only works with tf backend (theano gives symbolic tensors)
print("wc, hc =", wc, hc)

whc = Get_location_features(wc, hc)
whcI = Input(shape=(wc, hc, 2), name='wh_val')  # '2' for the w and h dimensions
cnn_features = concatenate([cnn_features, whcI], axis=3)  # (?, x,y,feat) = (0,1,2,3)

word_index = prepare.tokenizer.word_index
''' word_index is the tokenized list of words (e.g., 'over' = 25) in the corpus for the current problem
('25' = "token" = "word index").   '''

embedding_layer = prepare.embedding_layer(word2vec_path, word_index, mxlen)
''' The embedding matrix is simply the set of word vectors for a sentence. '''

embedding = embedding_layer(input2)
''' input2 is the english sentence (which is true or false) made of 32 maxlen integers (mostly 0)
    and turns it into an embedding matrix where each vec is the word2vec for the corresponding word
'''

bi_lstm = Bidirectional(LSTM(lstm_unit, implementation=2, return_sequences=False,
                             recurrent_regularizer=regularizers.l2(l2_norm),
                             recurrent_dropout=0.25))
lstm_encode = bi_lstm(embedding)

features = []  # these are layers
nfn = n_feat * 2 + 2

for w_ in range(wc):
    for h_ in range(hc):
        lay = Lambda(lambda x: cnn_features[:, w_, h_, :], output_shape=(nfn,))(cnn_features)
        # Note: do not include the batch dimension in the output shape
        # note also that output_shape is not needed for tensorflow
        features.append(lay)

relations = []
for i, feature1 in enumerate(features):
    for j, feature2 in enumerate(features):
        if j <= i:
            continue
        relations.append(concatenate([feature1, feature2, lstm_encode], axis=1))
        # note also that axis spec. is not needed for tensorflow
print("Contatenated the features with the lstm_encode")


def stack_layer(layers):
    def f(x):
        for k in range(len(layers)):
            x = layers[k](x)
        return x

    return f


def get_MLP(n):
    r = []
    for k in range(n):
        s = stack_layer([
            Dense(MLP_unit),
            BatchNormalization(),
            Activation('relu')
        ])
        r.append(s)
    return stack_layer(r)


def bn_dense(x):
    y = Dense(MLP_unit)(x)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)
    y = Dropout(0.5)(y)
    return y


g_MLP = get_MLP(3)

mid_relations = []
for r in relations:
    mid_relations.append(g_MLP(r))
combined_relation = Add()(mid_relations)

rn = bn_dense(combined_relation)
# rn = bn_dense(rn)
pred = Dense(1, activation='sigmoid')(rn)

model = Model(inputs=[input1, input2, whcI], outputs=pred)
print("Defined the model successfully!")

# print(model.summary())   # this is a huge summary!

optimizer = Adam(lr=3e-4)
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
print("Compiled the model successfully!")

# model.load_weights('weights_single_image')

for i in range(epochs):
    model.fit([imgs, ws, whc], labels, epochs=1, batch_size=batch_size)
    # model.save_weights('weights_single_image')

# tokenizer_file = open('tokenizer', 'wb')
# pickle.dump(prepare.tokenizer, tokenizer_file)
# tokenizer_file.close()
# gc.collect()
# subprocess.Popen("rundll32.exe powrprof.dll,SetSuspendState 0,1,0")
