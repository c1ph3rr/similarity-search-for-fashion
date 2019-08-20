import tensorflow as tf
import numpy as np
from tqdm import tqdm
from glob import glob

data = sorted(glob('datset/train/*/*'))
print('Total train data:', len(train))

model = tf.keras.models.load_model('xception.h5')
model.load_weights('model_weights.h5')
base = Model(inputs=model.input, outputs=model.layers[-2].output)
base.save('base.h5')

embeddings = np.zeros((len(data), 2048))

bad = []
for i in tqdm(range(len(data))):
    try:
        img = tf.keras.preprocessing.image.load_img(
            data[i], target_size=(331, 331, 3))
        img = tf.keras.preprocessing.image.img_to_array(img)
        img = tf.keras.applications.xception.preprocess_input(img)
        embeddings[i] = base.predict(img[None, ...])[0]
    except:
        bad.append(i)

np.save('embedding.npy', embeddings)
