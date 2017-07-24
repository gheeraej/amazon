import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import numpy as np

from sklearn.metrics import fbeta_score
from sklearn.model_selection import train_test_split

import keras as k
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import Callback, EarlyStopping, ModelCheckpoint
from keras import backend
"""
import tensorflow.contrib.keras.api.keras as k
from tensorflow.contrib.keras.api.keras.models import Sequential
from tensorflow.contrib.keras.api.keras.layers import Dense, Dropout, Flatten
from tensorflow.contrib.keras.api.keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.contrib.keras.api.keras.optimizers import Adam
from tensorflow.contrib.keras.api.keras.callbacks import Callback, EarlyStopping
from tensorflow.contrib.keras import backend
"""

class LossHistory(Callback):
    def __init__(self):
        super(LossHistory, self).__init__()
        self.train_losses = []
        self.val_losses = []

    def on_epoch_end(self, epoch, logs={}):
        self.train_losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))


class AmazonKerasClassifier:
    def __init__(self):
        self.losses = []
        self.classifier = Sequential()

    def add_conv_layer(self, img_size=(32, 32), img_channels=3):
        self.classifier.add(BatchNormalization(input_shape=(img_size[0], img_size[1], img_channels)))

        '''
        self.classifier.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
        self.classifier.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
        self.classifier.add(MaxPooling2D(pool_size=2))
        self.classifier.add(Dropout(0.25))

        self.classifier.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
        self.classifier.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
        self.classifier.add(MaxPooling2D(pool_size=2))
        self.classifier.add(Dropout(0.25))

        self.classifier.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
        self.classifier.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
        self.classifier.add(MaxPooling2D(pool_size=2))
        self.classifier.add(Dropout(0.25))

        self.classifier.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
        self.classifier.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
        self.classifier.add(MaxPooling2D(pool_size=2))
        self.classifier.add(Dropout(0.25))
        '''
        
        self.classifier.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
        self.classifier.add(BatchNormalization(input_shape=(img_size[0], img_size[1], 32)))
        self.classifier.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
        self.classifier.add(MaxPooling2D(pool_size=2))
        self.classifier.add(BatchNormalization(input_shape=(img_size[0]/2, img_size[1]/2, 32)))
        #self.classifier.add(Dropout(0.25))

        self.classifier.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
        self.classifier.add(BatchNormalization(input_shape=(img_size[0]/2, img_size[1]/2, 64)))
        self.classifier.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
        self.classifier.add(MaxPooling2D(pool_size=2))
        self.classifier.add(BatchNormalization(input_shape=(img_size[0]/4, img_size[1]/4, 64)))
        #self.classifier.add(Dropout(0.25))
        
        self.classifier.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
        self.classifier.add(BatchNormalization(input_shape=(img_size[0]/4, img_size[1]/4, 128)))
        self.classifier.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
        self.classifier.add(MaxPooling2D(pool_size=2))
        self.classifier.add(BatchNormalization(input_shape=(img_size[0]/8, img_size[1]/8, 128)))
        #self.classifier.add(Dropout(0.25))

        self.classifier.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
        self.classifier.add(BatchNormalization(input_shape=(img_size[0]/8, img_size[1]/8, 256)))
        self.classifier.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
        self.classifier.add(MaxPooling2D(pool_size=2))
        self.classifier.add(BatchNormalization(input_shape=(img_size[0]/16, img_size[1]/16, 256)))
        #self.classifier.add(Dropout(0.25))
        
        '''
        self.classifier.add(Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1'))
        self.classifier.add(BatchNormalization(input_shape=(img_size[0], img_size[1], 64)))
        self.classifier.add(Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2'))
        self.classifier.add(MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool'))
        self.classifier.add(BatchNormalization(input_shape=(img_size[0]/2, img_size[1]/2, 64)))
        
        self.classifier.add(Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1'))
        self.classifier.add(BatchNormalization(input_shape=(img_size[0]/2, img_size[1]/2, 128)))
        self.classifier.add(Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2'))
        self.classifier.add(MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool'))
        self.classifier.add(BatchNormalization(input_shape=(img_size[0]/4, img_size[1]/4, 128)))
        
        self.classifier.add(Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1'))
        self.classifier.add(BatchNormalization(input_shape=(img_size[0]/4, img_size[1]/4, 256)))
        self.classifier.add(Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2'))
        self.classifier.add(BatchNormalization(input_shape=(img_size[0]/4, img_size[1]/4, 256)))
        self.classifier.add(Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3'))
        self.classifier.add(MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool'))
        self.classifier.add(BatchNormalization(input_shape=(img_size[0]/8, img_size[1]/8, 256)))
        
        self.classifier.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1'))
        self.classifier.add(BatchNormalization(input_shape=(img_size[0]/8, img_size[1]/8, 512)))
        self.classifier.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2'))
        self.classifier.add(BatchNormalization(input_shape=(img_size[0]/8, img_size[1]/8, 512)))
        self.classifier.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3'))
        self.classifier.add(MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool'))
        self.classifier.add(BatchNormalization(input_shape=(img_size[0]/16, img_size[1]/16, 512)))
        
        self.classifier.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1'))
        self.classifier.add(BatchNormalization(input_shape=(img_size[0]/16, img_size[1]/16, 512)))
        self.classifier.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2'))
        self.classifier.add(BatchNormalization(input_shape=(img_size[0]/16, img_size[1]/16, 512)))
        self.classifier.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3'))
        self.classifier.add(MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool'))
        self.classifier.add(BatchNormalization(input_shape=(img_size[0]/32, img_size[1]/32, 512)))
        '''
    
    def add_flatten_layer(self):
        self.classifier.add(Flatten())


    def add_ann_layer(self, output_size):
        self.classifier.add(Dense(512, activation='relu'))
        self.classifier.add(BatchNormalization())
        self.classifier.add(Dropout(0.5))
        self.classifier.add(Dense(output_size, activation='sigmoid'))

    def _get_fbeta_score(self, classifier, X_valid, y_valid):
        p_valid = classifier.predict(X_valid)
        return fbeta_score(y_valid, np.array(p_valid) > 0.2, beta=2, average='samples')

    def train_model(self, x_train, y_train, learn_rate=0.001, epoch=5, batch_size=128, validation_split_size=0.2, train_callbacks=()):
        history = LossHistory()

        X_train, X_valid, y_train, y_valid = train_test_split(x_train, y_train,
                                                              test_size=validation_split_size, random_state=42)

        opt = Adam(lr=learn_rate)

        self.classifier.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])


        # early stopping will auto-stop training process if model stops learning after 3 epochs
        earlyStopping = EarlyStopping(monitor='val_loss', patience=3, verbose=0, mode='auto')
        checkpoint = ModelCheckpoint(train_callbacks[0].filepath, train_callbacks[0].monitor, train_callbacks[0].verbose, train_callbacks[0].save_best_only)
        
        self.classifier.fit(X_train, y_train,
                            batch_size=batch_size,
                            epochs=epoch,
                            verbose=2,
                            validation_data=(X_valid, y_valid),
                            callbacks=[history, checkpoint, earlyStopping])
        fbeta_score = self._get_fbeta_score(self.classifier, X_valid, y_valid)
        return [history.train_losses, history.val_losses, fbeta_score]

    def save_weights(self, weight_file_path):
        self.classifier.save_weights(weight_file_path)

    def load_weights(self, weight_file_path):
        self.classifier.load_weights(weight_file_path)

    def predict(self, x_test):
        predictions = self.classifier.predict(x_test)
        return predictions

    def map_predictions(self, predictions, labels_map, thresholds):
        """
        Return the predictions mapped to their labels
        :param predictions: the predictions from the predict() method
        :param labels_map: the map
        :param thresholds: The threshold of each class to be considered as existing or not existing
        :return: the predictions list mapped to their labels
        """
        predictions_labels = []
        for prediction in predictions:
            labels = [labels_map[i] for i, value in enumerate(prediction) if value > thresholds[i]]
            predictions_labels.append(labels)

        return predictions_labels

    def close(self):
        backend.clear_session()
