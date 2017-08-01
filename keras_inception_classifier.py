import os
import pickle

import numpy as np
from keras.applications import InceptionV3
from keras.applications import VGG16
from keras.layers import GlobalAveragePooling2D, Dense
from keras.optimizers import SGD
from keras.models import Model as KerasModel
from keras.utils import to_categorical
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc, roc_auc_score

from nideep.datasets.amfed.amfed import AMFED
import pandas as pd

class KerasInceptionClassifier(object):
    def __init__(self, sess, gan, cache_dir='/mnt/raid/data/ni/dnn/rparra/cache/', input_height=64, input_width=64,
                 y_dim=1, c_dim=3, global_step=-1):
        self.sess = sess
        self.input_height = input_height
        self.input_width = input_width
        self.y_dim = y_dim
        self.c_dim = c_dim
        self.cache_dir = cache_dir
        self.gan = gan
        self.global_step = global_step

    def get_dataset(self):
        dataset = AMFED(dir_prefix='/mnt/raid/data/ni/dnn/AMFED/', video_type=AMFED.VIDEO_TYPE_AVI,
                        cache_dir=self.cache_dir)

        (X_train, y_train, videos_train, _, X_test, y_test, _, _) = dataset.as_numpy_array(train_proportion=0.8)
        X_train = np.flip((X_train.astype(np.float32) - 127.5) / 127.5, axis=-1)
        X_test = np.flip((X_test.astype(np.float32) - 127.5) / 127.5, axis=-1)
        y_train = np.squeeze(y_train)
        y_test = np.squeeze(y_test)

        return X_train, X_test, y_train, y_test, len(set(videos_train))

    def get_classifier(self):
        # param_grid = {'C': [2 ** x for x in range(-3, 5)],
        #               'gamma': [2 ** x for x in range(-3, 5)]}
        # return GridSearchCV(SVC(), param_grid)
        return Model()

    def evaluate(self, config):
        if config.dataset == 'amfed':
            X_train, X_test, y_train, y_test, video_number = self.get_dataset()
            results = []

            clf = self.get_classifier()
            print 'Evaluating unbalanced dataset'
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test, probability=False)
            y_score = clf.predict(X_test)
            auc_result = self.save_roc(y_test, y_score, 'unbalanced_roc.p')
            print("Accuracy: {0:0.1f}%".format(accuracy_score(y_test, y_pred) * 100))
            print(confusion_matrix(y_test, y_pred))
            results.append(self._build_result(auc_result, 'imbalanced'))

            clf = self.get_classifier()
            print 'Evaluating oversampled dataset'
            X_train_oversampled, y_train_oversampled = self.oversample(X_train, y_train)
            clf.fit(X_train_oversampled, y_train_oversampled)
            y_pred = clf.predict(X_test, probability=False)
            y_score = clf.predict(X_test)
            auc_result = self.save_roc(y_test, y_score, 'oversampled_roc.p')
            print("Accuracy: {0:0.1f}%".format(accuracy_score(y_test, y_pred) * 100))
            print(confusion_matrix(y_test, y_pred))
            results.append(self._build_result(auc_result, 'oversampled'))


            clf = self.get_classifier()
            print 'Evaluating augmented dataset'
            X_train_augmented, y_train_augmented = self.get_augmented_dataset(X_train, y_train, config, video_number)
            clf.fit(X_train_augmented, y_train_augmented)
            y_pred = clf.predict(X_test, probability=False)
            y_score = clf.predict(X_test)
            auc_result = self.save_roc(y_test, y_score, 'augmented_roc.p')
            print("Accuracy: {0:0.1f}%".format(accuracy_score(y_test, y_pred) * 100))
            print(confusion_matrix(y_test, y_pred))
            results.append(self._build_result(auc_result, 'augmented'))
            return pd.DataFrame(results)

    def _build_result(self, auc, setting):
        return {'classifier': 'vgg_pretrained', 'dataset': setting, 'auc': auc, 'global_step': self.global_step}

    def save_roc(self, y_test, y_score, name):
        fpr, tpr, thresholds = roc_curve(y_test, y_score)
        roc_auc = auc(fpr, tpr)
        result = {'fpr': fpr, 'tpr': tpr, 'thresholds': thresholds, 'auc': roc_auc}
        pickle.dump(result, open(name, "wb"))
        auc_result = roc_auc_score(y_test, y_score)
        print 'AUC: %s' % str(auc_result)
        return auc_result

    def get_augmented_dataset(self, X_train, y_train, config, video_number):
        X_gen, y_gen = self.augment(y_train, config, video_number)
        return np.concatenate((X_train, X_gen)), np.concatenate((y_train, y_gen))

    def augment(self, y_train, config, video_number):
        selected_indices = np.where(y_train == 1)
        sample_size = (y_train.shape[0] - 2 * selected_indices[0].shape[0]) / config.batch_size
        y_one_hot = np.ones((config.batch_size, 1))
        y_video_label = np.random.choice(video_number, (config.batch_size, 1)) / float(video_number)
        y_sample = np.concatenate([y_one_hot, y_video_label], axis=1)
        result = []
        for i in range(sample_size):
            z_sample = np.random.uniform(-1, 1, size=[int(config.batch_size), self.gan.z_dim])
            samples = self.sess.run(self.gan.sampler, feed_dict={self.gan.z: z_sample, self.gan.y: y_sample})
            result.append(samples)

        X_augmented = np.concatenate(result)
        y_augmented = np.ones((config.batch_size * sample_size,))

        return X_augmented, y_augmented

    def oversample(self, X_train, y_train):
        selected_indices = np.where(y_train == 1)
        sample_size = y_train.shape[0] - 2 * selected_indices[0].shape[0]
        oversampled_indices = np.random.choice(selected_indices[0], sample_size)
        return np.concatenate((X_train, X_train[oversampled_indices])), \
               np.concatenate((y_train, y_train[oversampled_indices]))


class Model(object):
    def __init__(self, num_classes=2, batch_size=64, input_height=64, input_width=64):
        # create the base pre-trained model
        self.base_model = VGG16(weights='imagenet', include_top=False,
                                      input_shape=(input_height, input_width, 3))

        # add a global spatial average pooling layer
        x = self.base_model.output
        x = GlobalAveragePooling2D()(x)
        # let's add a fully-connected layer
        x = Dense(1024, activation='relu')(x)
        # and a logistic layer -- let's say we have 200 classes
        predictions = Dense(num_classes, activation='softmax')(x)

        # this is the model we will train
        self.model = KerasModel(inputs=self.base_model.input, outputs=predictions)
        self.batch_size = batch_size

    def fit(self, X_train, y_train, epochs=3):
        # X_train = X_train.repeat(3, axis=1).repeat(3, axis=2)
        y_binary = to_categorical(y_train)
        # first: train only the top layers (which were randomly initialized)
        # i.e. freeze all convolutional InceptionV3 layers
        for layer in self.base_model.layers:
            layer.trainable = False

        # compile the model (should be done *after* setting layers to non-trainable)
        self.model.compile(optimizer='rmsprop', loss='categorical_crossentropy')

        # train the model on the new data for a few epochs
        self.model.fit(X_train, y_binary, epochs=epochs, batch_size=64)

        # at this point, the top layers are well trained and we can start fine-tuning
        # convolutional layers from inception V3. We will freeze the bottom N layers
        # and train the remaining top layers.

        # let's visualize layer names and layer indices to see how many layers
        # we should freeze:
        for i, layer in enumerate(self.base_model.layers):
            print(i, layer.name)

        # we chose to train the top 2 inception blocks, i.e. we will freeze
        # the first 249 layers and unfreeze the rest:
        for layer in self.model.layers[:249]:
            layer.trainable = False
        for layer in self.model.layers[249:]:
            layer.trainable = True

        self.model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy')

        # we train our model again (this time fine-tuning the top 2 inception blocks
        # alongside the top Dense layers
        self.model.fit(X_train, y_binary, epochs=epochs, batch_size=64)

    def predict(self, X_test, probability=True):
        # X_test = X_test.repeat(3, axis=1).repeat(3, axis=2)
        probs = self.model.predict(X_test)
        if probability:
            return probs[:, 1]
        else:
            return np.argmax(probs, axis=1)
