# some_file.py
import os
from keras.callbacks import LambdaCallback, EarlyStopping
import numpy as np
import pandas as pd
import cv2
import tensorflow as tf
import sys
from keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping
from keras.callbacks import ReduceLROnPlateau
from keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split

CNN_Model = os.getcwd() + "\\" + "CNN"
sys.path.append(CNN_Model)


import neuralNetwork



# LambdaCallback is constructed with anonymous functions that will be called at the appropriate time.
# Note that the callbacks expects positional arguments, as: on_epoch_begin/on_epoch_end,on_batch_begin/on_batch_end, on_train_begin/on_train_end
# Early stopping is used to stop  training when a monitored quantity has stopped improving.

# Import model file

data_generator = ImageDataGenerator(
                        featurewise_center=False,
                        featurewise_std_normalization=False,
                        rotation_range=10,
                        width_shift_range=0.1,
                        height_shift_range=0.1,
                        zoom_range=.1,
                        horizontal_flip=True)

image_size=(48,48)
dataset_path = 'modelTraining/fer2013.csv'

def load_fer2013():
        data = pd.read_csv(dataset_path)
        pixels = data['pixels'].tolist()
        width, height = 48, 48
        faces = []
        for pixel_sequence in pixels:
            face = [int(pixel) for pixel in pixel_sequence.split(' ')]
            face = np.asarray(face).reshape(width, height)
            face = cv2.resize(face.astype('uint8'),image_size)
            faces.append(face.astype('float32'))
        faces = np.asarray(faces)
        faces = np.expand_dims(faces, -1)
        emotions = pd.get_dummies(data['emotion']).values
        return faces, emotions

def preprocess_input(x, v2=True):
    x = x.astype('float32')
    x = x / 255.0
    if v2:
        x = x - 0.5
        x = x * 2.0
    return x

batch_size = 32
num_epochs = 10000
input_shape = (48, 48,1)
validation_split = .2
verbose = 1
num_classes = 7
patience = 50
base_path = 'model Training/'

model = neuralNetwork.mini_XCEPTION(input_shape, num_classes)
model.compile(optimizer=Adam(learning_rate=0.001),
            loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()
tf.keras.utils.plot_model(model, 'graphs/modelArchitecture.png', show_shapes=True)


# callbacks
log_file_path = base_path + '_emotion_training.log'
csv_logger = CSVLogger(log_file_path, append=False)
early_stop = EarlyStopping('val_loss', patience=patience)
reduce_lr = ReduceLROnPlateau('val_loss', factor=0.1,
                            patience=int(patience/4), verbose=1)
trained_models_path = base_path + '_mini_XCEPTION'
model_names = trained_models_path + '.{epoch:02d}-{val_accuracy:.2f}.hdf5'
model_checkpoint = ModelCheckpoint(model_names, 'val_loss', verbose=1,
                                save_best_only=True)
callbacks = [model_checkpoint, csv_logger, early_stop, reduce_lr]

history = model.fit(train_generator,
                        validation_data=test_generator,
                        validation_steps=validation_steps_per_epoch,
                        steps_per_epoch = train_steps_per_epoch,
                        epochs = 80,
                        verbose=2,
                        callbacks=[checkpoint, es])

#Graphs 
fig, (ax1, ax2) = plt.subplots(2,1, figsize=(15,13), sharex=True)

val_acc = history.history['val_accuracy']
val_loss = history.history['val_loss']

test_acc = history.history['accuracy']
test_loss = history.history['loss']


ax1.plot(range(80), test_loss, label='loss')
ax1.plot(range(80), val_loss, label='val loss')
ax1.set_title('Training Loss & Validation Loss')
ax1.set_ylabel('Loss')
ax1.set_xlabel('Epoch')
ax1.set_ylim(bottom=0.5, top=3.)
ax1.legend(['Train', 'Validation'], loc='upper right');


ax2.plot(range(80), test_acc, label='accuracy')
ax2.plot(range(80), val_acc, label='val accuracy')
ax2.set_title('Training Accuracy & Validation Accuracy')
ax2.set_ylabel('Accuracy')
ax2.set_ylim(bottom=0.2, top=0.9)
ax2.legend(['Train', 'Validation'], loc='upper right');

fig.savefig('graphs.png')
 
#Confusin Matrix
cm = confusion_matrix(y_true = y_true, y_pred = y_hat)
plot_confusion_matrix(cm, classes, cmap = 'Reds')
plt.savefig('confusionMatrix.png')

# loading dataset
faces, emotions = load_fer2013()
faces = preprocess_input(faces)
num_samples, num_classes = emotions.shape
xtrain, xtest,ytrain,ytest = train_test_split(faces, emotions,test_size=0.2,shuffle=True)
model.fit_generator(data_generator.flow(xtrain, ytrain,
                                            batch_size),
                        steps_per_epoch=len(xtrain) / batch_size,
                        epochs=num_epochs, verbose=1, callbacks=callbacks,
                        validation_data=(xtest,ytest))

