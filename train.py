import numpy as np
from keras.models import Model
from keras.layers import Dense, Dropout, Activation, Input, \
    LSTM, Bidirectional, Lambda, Add

from keras.layers.merge import concatenate

from keras.layers.convolutional import Conv2D, MaxPooling2D, AveragePooling2D
from keras.layers.normalization import BatchNormalization, regularizers
from keras.optimizers import Adam  # , RMSprop
import prepare
from os.path import expanduser
home = expanduser("~/Dropbox/Data")

# image size (scaled from the original data set):
w = 200
h = 50

debugg = 0  # sets it up for tiny databases so can debug easily
            # (need to create mini version of glove.6B.50d.txt)

validate = 1

np.random.seed(1)
mxlen = 32
embedding_dim = 50
lstm_unit = 64
MLP_unit = 128

epochs = 200

batch_size = 256
l2_norm = 1e-4
n_feat = 16

nlvr_path = home + '/nlvr'
w2v_path  = home + '/text_embedding'
word2vec_path    = w2v_path + '/glove.6B.50d.txt'
if debugg:
    word2vec_path = w2v_path + 'text_embedding/glove.6B.50d.tiny.txt'
    validate = False

# for following word embedding method see:
# https://blog.keras.io/using-pre-trained-word-embeddings-in-a-keras-model.html

train_json       = nlvr_path + '/train/train.json'
train_img_folder = nlvr_path + '/train/images'
data = prepare.load_data(train_json)
prepare.init_tokenizer(data)
data = prepare.tokenize_data(data, mxlen)
imgs, ws, labels = prepare.load_images(train_img_folder, data, w, h, debug=debugg)
data.clear()
n_images = imgs.shape[0]

if validate:
    test_json = nlvr_path + '/test/test.json'
    test_img_folder = nlvr_path + '/test/images'
    test_data = prepare.load_data(test_json)
    prepare.init_tokenizer(test_data)
    test_data = prepare.tokenize_data(test_data, mxlen)
    test_imgs, test_ws, test_labels = prepare.load_images(test_img_folder, test_data, w, h)
    n_images_validation = test_imgs.shape[0]
    test_data.clear()

    draw_test_data_from_training_data = 1
    if draw_test_data_from_training_data:
        ''' This code is working poorly with the validation data this section 
            ensures that it is not any trouible with the validation data itself '''
        n_images_validation  = int(0.08*n_images)
        test_imgs = imgs[-n_images_validation:]
        test_ws = ws[-n_images_validation :]
        test_labels = labels[-n_images_validation :]

        n_images = n_images - n_images_validation
        imgs = imgs[:n_images_validation]
        ws = ws[:n_images_validation]
        labels = labels[:n_images_validation]


print("finished reading training images...")

imgs_mean = np.mean(imgs)
imgs_std = np.std(imgs - imgs_mean)
imgs = (imgs - imgs_mean) / imgs_std
print("no. of images =", n_images)

if validate:
    test_imgs = (test_imgs - imgs_mean) / imgs_std


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
    wh = 1.0*np.ones((n_images, w, h, 2))
    for w_ in range(w):
        wh[:, w_, :, 0] = w_/w
        for h_ in range(h):
            wh[:,:, h_:, 1] = h_/h
    return wh

input1 = Input((w, h, 3))

input2 = Input((mxlen,))

cnn_features = conv_net(input1)
shapes = cnn_features.shape
wf, hf = int(shapes[1]), int(shapes[2])  # this only works with tf backend (theano gives symbolic tensors)
print("wf, hf =", wf, hf)

whf = Get_location_features(wf, hf)
if validate:
    if n_images < n_images_validation:
        raise NameError("Too many test images specified.")
    whf_validation = whf[:n_images_validation]

whfI = Input(shape=(wf, hf, 2), name='whfI')  # '2' for the w and h dimensions
cnn_features = concatenate([cnn_features, whfI], axis=3)  # (?, x,y,feat) = (0,1,2,3)

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

for w_ in range(wf):
    for h_ in range(hf):
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
print("Concatenated the features with the lstm_encode")


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

pred = Dense(1, activation='sigmoid')(rn)

model = Model(inputs=[input1, input2, whfI], outputs=pred)

# print(model.summary())   # this is a huge summary!

optimizer = Adam(lr=3e-4)
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
print("Compiled the model successfully!")

model.load_weights('weights')
ep = 0
ep_prt = 10
super_epochs = int(epochs/ep_prt)

for i in range(super_epochs):
    if validate:
        model.fit([imgs, ws, whf], labels, 
                  validation_data=[[test_imgs, test_ws, whf_validation], test_labels],
                  epochs=ep_prt, batch_size=batch_size)
    else:
        model.fit([imgs, ws, whf], labels, epochs=ep_prt, batch_size=batch_size)
    ep += ep_prt
    print("Epoch total = ", ep)
    model.save_weights('weights')


# tokenizer_file = open('tokenizer', 'wb')
# pickle.dump(prepare.tokenizer, tokenizer_file)
# tokenizer_file.close()
# gc.collect()
# subprocess.Popen("rundll32.exe powrprof.dll,SetSuspendState 0,1,0")
