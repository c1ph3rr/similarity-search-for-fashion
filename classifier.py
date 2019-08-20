import tensorflow as tf
import numpy as np
import math
import os
from glob import glob
from PIL import ImageFile

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

ImageFile.LOAD_TRUNCATED_IMAGES = True

print('Tensorflow:', tf.__version__)

train_path = 'dataset/train/'
test_path = 'dataset/val/'

lr = 1e-3
batch_size = 32
img_height, img_width = 331, 331
classes = 6


def create_dict(path):
    p = sorted(glob(path + '*'))
    label_dict = {}

    for i, v in enumerate(p):
        label_dict[i] = len(glob(v + '/*'))
    return label_dict


def create_class_weight(labels_dict, mu):
    total = np.sum([labels_dict[i] for i in labels_dict.keys()])
    keys = labels_dict.keys()
    class_weight = {}

    for key in keys:
        score = math.log(mu * total / float(labels_dict[key]))
        class_weight[key] = score if score > 1.0 else 1.0
    return class_weight


class_weights = create_class_weight(create_dict(train_path), .40)


train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    preprocessing_function=preprocess_input,
    width_shift_range=0.3,
    height_shift_range=0.3,
    rotation_range=30,
    shear_range=0.5,
    zoom_range=.7,
    channel_shift_range=0.3,
    cval=0.5,
    vertical_flip=True,
    fill_mode='nearest')

test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    preprocessing_function=preprocess_input)

train_generator = train_datagen.flow_from_directory(
    train_path,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical')

test_generator = test_datagen.flow_from_directory(
    test_path,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical')


model_path = 'weights/'
final_weights_path = os.path.join(
    os.path.abspath(model_path), 'model_weights.h5')

callbacks_list = [tf.keras.callbacks.ModelCheckpoint(final_weights_path, monitor='val_acc', verbose=1, save_best_only=True),
                  tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, verbose=1)]


xception = tf.keras.applications.Xception(include_top=False, weights='imagenet',
                                          input_shape=(img_height, img_width, 3))
x = xception.output
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dense(classes, activation='softmax')(x)
model = tf.keras.models.Model(inputs=xception.input, outputs=x)


unfreeze = 85
for layer in model.layers[:-unfreeze]:
    layer.trainable = False

for layer in model.layers[-unfreeze:]:
    layer.trainable = True


try:
    model.load_weights(model_path + 'model_weights.h5')
except:
    print('Training from scratch :(')

model.compile(optimizer=tf.keras.optimizers.Nadam(
    lr), loss='categorical_crossentropy', metrics=['accuracy'])


history = model.fit_generator(train_generator,
                              steps_per_epoch=train_generator.__len__(),
                              epochs=50,
                              validation_data=test_generator,
                              validation_steps=test_generator.__len__(),
                              class_weight=class_weights,
                              use_multiprocessing=False,
                              callbacks=callbacks_list)

model.save('xception.h5')
