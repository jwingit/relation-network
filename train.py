import numpy as np
import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten, Input, Embedding, \
    LSTM, Bidirectional, Lambda, Concatenate, Add
from keras.layers.convolutional import Conv2D, MaxPooling2D, AveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam, RMSprop
import gc
import prepare
import subprocess
import pickle

mxlen = 32
embedding_dim = 50
lstm_unit = 128
MLP_unit = 128
epochs = 50
batch_size = 128

train_json = 'nlvr\\train\\train.json'
train_img_folder = 'nlvr\\train\\images'
test_json = 'nlvr\\test\\test.json'
test_img_folder = 'nlvr\\test\\images'
data = prepare.load_data(train_json)
prepare.init_tokenizer(data)
data = prepare.tokenize_data(data, mxlen)
imgs, ws, labels = prepare.load_images(train_img_folder, data)
data.clear()

test_data = prepare.load_data(test_json)
test_data = prepare.tokenize_data(test_data, mxlen)
test_imgs, test_ws, test_labels = prepare.load_images(test_img_folder, test_data)
test_data.clear()

imgs_mean = np.mean(imgs)
imgs_std = np.std(imgs - imgs_mean)
imgs = (imgs - imgs_mean) / imgs_std

test_imgs = (test_imgs - imgs_mean) / imgs_std


def bn_layer(x, conv_unit):
    def f(inputs):
        md = Conv2D(x, (conv_unit, conv_unit), padding='same', kernel_initializer='he_normal')(inputs)
        md = BatchNormalization()(md)
        return Activation('relu')(md)

    return f


def conv_net(inputs):
    model = bn_layer(32, 3)(inputs)
    model = MaxPooling2D((3, 3), 3)(model)
    model = bn_layer(32, 3)(model)
    model = MaxPooling2D((2, 2), 2)(model)
    model = bn_layer(32, 3)(model)
    model = MaxPooling2D((2, 2), 2)(model)
    model = bn_layer(32, 3)(model)
    model = MaxPooling2D((2, 2), 2)(model)
    model = bn_layer(64, 3)(model)
    return model


input1 = Input((50, 200, 3))
input2 = Input((mxlen,))
cnn_features = conv_net(input1)
embedding_layer = prepare.embedding_layer(prepare.tokenizer.word_index, prepare.get_embeddings_index(), mxlen)
embedding = embedding_layer(input2)
# embedding = Embedding(mxlen, embedding_dim)(input2)
bi_lstm = Bidirectional(LSTM(lstm_unit, implementation=2, return_sequences=False))
lstm_encode = bi_lstm(embedding)
shapes = cnn_features.shape
w, h = shapes[1], shapes[2]


def slice_1(t):
    return t[:, 0, :, :]


def slice_2(t):
    return t[:, 1:, :, :]


def slice_3(t):
    return t[:, 0, :]


def slice_4(t):
    return t[:, 1:, :]


slice_layer1 = Lambda(slice_1)
slice_layer2 = Lambda(slice_2)
slice_layer3 = Lambda(slice_3)
slice_layer4 = Lambda(slice_4)

features = []
for k1 in range(w):
    features1 = slice_layer1(cnn_features)
    cnn_features = slice_layer2(cnn_features)
    for k2 in range(h):
        features2 = slice_layer3(features1)
        features1 = slice_layer4(features1)
        features.append(features2)

relations = []
concat = Concatenate()
for feature1 in features:
    for feature2 in features:
        relations.append(concat([feature1, feature2, lstm_encode]))


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
    print(len(mid_relations))
combined_relation = Add()(mid_relations)

rn = bn_dense(combined_relation)
rn = bn_dense(rn)
pred = Dense(1, activation='sigmoid')(rn)

model = Model(inputs=[input1, input2], outputs=pred)
optimizer = Adam(lr=3e-4)
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
for epoch in range(epochs):
    model.fit([imgs, ws], labels, epochs=1, batch_size=batch_size)
    p = model.predict([test_imgs, test_ws], batch_size=batch_size)
    p = np.array([t[0] for t in p])
    acc = np.sum((p >= 0.5) == (test_labels >= 0.5)) / len(p)
    avg = np.sum(p) / len(p)
    print('epoch: ', epoch, ", acc: ", acc, ", avg = ", avg)
    for k in range(100):
        print(p[k], test_labels[k])
model.save('model')
tokenizer_file = open('tokenizer', 'wb')
pickle.dump(prepare.tokenizer, tokenizer_file)
tokenizer_file.close()
gc.collect()
# subprocess.Popen("rundll32.exe powrprof.dll,SetSuspendState 0,1,0")
