import prepare
import keras
import pickle
import numpy as np

mxlen = 32

token_file = open('tokenizer', 'rb')
prepare.tokenizer = pickle.load(token_file)

train_json = 'nlvr\\train\\train.json'
train_img_folder = 'nlvr\\train\\images'
test_json = 'nlvr\\test\\test.json'
test_img_folder = 'nlvr\\test\\images'

data = prepare.load_data(train_json)
data = prepare.tokenize_data(data, mxlen)
imgs, ws, labels = prepare.load_images(train_img_folder, data)
data.clear()

model = keras.models.load_model('model')


test_data = prepare.load_data(test_json)
test_data = prepare.tokenize_data(test_data, mxlen)
test_imgs, test_ws, test_labels = prepare.load_images(test_img_folder, test_data)
test_data.clear()

imgs_mean = np.mean(imgs)
imgs_std = np.std(imgs - imgs_mean)

test_imgs = (test_imgs - imgs_mean) / imgs_std

print(model.evaluate([test_imgs, test_ws], test_labels, batch_size=128))