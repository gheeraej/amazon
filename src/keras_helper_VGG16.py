import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import gc
import numpy as np

from sklearn.metrics import fbeta_score
from sklearn.model_selection import train_test_split

import keras as k
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.callbacks import Callback, EarlyStopping, ModelCheckpoint
from keras import backend

from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input, decode_predictions

class LossHistory(Callback):
    def __init__(self):
        super(LossHistory, self).__init__()
        self.train_losses = []
        self.val_losses = []

    def on_epoch_end(self, epoch, logs={}):
        self.train_losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))


class AmazonKerasClassifier_VGG16:
    def __init__(self, img_size, output_size):
        self.losses = []
        if img_size[0]<200 or img_size[1]<200:
            print("Minimum size is 200x200 for ResNet50. Errors may occur")
        base_model = VGG16(weights='imagenet', input_shape=(img_size[0], img_size[1], 3), include_top=False)
        x = base_model.output
        x = Flatten()(x)
        x = Dropout(0.5)(x)
        predictions = Dense(output_size, activation='sigmoid')(x)
        self.classifier = Model(inputs=base_model.input, outputs=predictions)

    def _get_fbeta_score(self, classifier, X_valid, y_valid):
        p_valid = classifier.predict(X_valid)
        return fbeta_score(y_valid, np.array(p_valid) > 0.2, beta=2, average='samples')

    def train_model(self, X_train, X_valid, y_train, y_valid, learn_rate=0.001, epoch=5, batch_size=128, validation_split_size=0.2, train_callbacks=()):
        history = LossHistory()

        #X_train, X_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=validation_split_size, random_state=42)

        opt = Adam(lr=learn_rate)

        self.classifier.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

        datagen = ImageDataGenerator(
            #featurewise_center=True,
            #featurewise_std_normalization=True,
            rotation_range=90,
            width_shift_range=0.2,
            height_shift_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            vertical_flip=True,
            fill_mode="nearest"
        )
        
        #datagen.fit(X_train)
        
        # early stopping will auto-stop training process if model stops learning after 3 epochs
        earlyStopping = EarlyStopping(monitor='val_loss', patience=5, verbose=0, mode='auto')
        checkpoint = ModelCheckpoint(train_callbacks[0].filepath, train_callbacks[0].monitor, train_callbacks[0].verbose, train_callbacks[0].save_best_only)
        
        self.classifier.fit_generator(datagen.flow(X_train, y_train, batch_size=batch_size),
                                      steps_per_epoch=len(X_train) / batch_size,
                                      epochs=epoch,
                                      verbose=2,
                                      validation_data=(X_valid, y_valid),
                                      callbacks=[history, checkpoint, earlyStopping])
        """
        self.classifier.fit(X_train, y_train,
                            batch_size=batch_size,
                            epochs=epoch,
                            verbose=2,
                            validation_data=(X_valid, y_valid),
                            callbacks=[history, checkpoint, earlyStopping])
        """
        
        fbeta_score = self._get_fbeta_score(self.classifier, X_valid, y_valid)
        
        del opt, datagen, earlyStopping, checkpoint
        gc.collect()
        
        return [history.train_losses, history.val_losses, fbeta_score]

    def save_weights(self, weight_file_path):
        self.classifier.save_weights(weight_file_path)

    def load_weights(self, weight_file_path):
        self.classifier.load_weights(weight_file_path)

    def predict(self, x_test):
        predictions = self.classifier.predict(x_test)
        return predictions
    
    def transform_input(self, X):
        """ 
        transform rotates and flips images and labels
        X (numpy.array): image
        Returns: transformed image
        """
        list_out = []
        for i in range(4):
            list_out.append(np.expand_dims(np.rot90(X, k=i, axes=(0,1)), axis=0))
            list_out.append(np.expand_dims(np.rot90(np.flip(X, axis=0), k=i, axes=(0,1)), axis=0))
            list_out.append(np.expand_dims(np.rot90(np.flip(X, axis=1), k=i, axes=(0,1)), axis=0))
            list_out.append(np.expand_dims(np.rot90(np.flip(np.flip(X, axis=-1), axis=0), k=i, axes=(0,1)), axis=0))

        return np.array(list_out).squeeze()

    def predict_TTA(self, x_test):
        list_out = []
        for i in range(x_test.shape[0]):
            array_tta = self.transform_input(x_test[i])
            list_out.append(self.classifier.predict(array_tta))
        return list_out
    
    def flattend_predictions(self, pred_input):
        out_flatten = []
        for list_elt in pred_input:
            for elt in list_elt:
                out_flatten.append(elt)
        return out_flatten
    
    def map_predictions_TTA(self, predictions, labels_map, thresholds):
        """
        Return the predictions mapped to their labels
        :param predictions: the predictions from the predict() method
        :param labels_map: the map
        :param thresholds: The threshold of each class to be considered as existing or not existing
        :return: the predictions list mapped to their labels
        """
        predictions_labels = []
        for image_pred in predictions:
            list_labels = self.map_predictions(image_pred, labels_map, thresholds)
            list_flatten = self.flattend_predictions(list_labels)
            labels = []
            for nb, label in labels_map.iteritems():
                if list_flatten.count(label)/float(len(labels_map)) > 0.5:
                    labels.append(label)
            predictions_labels.append(labels)    
        return predictions_labels
    
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
