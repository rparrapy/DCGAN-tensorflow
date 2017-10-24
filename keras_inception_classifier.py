import pickle
import time

import numpy as np
import pandas as pd
from keras.applications import VGG16
from keras.layers import GlobalAveragePooling2D, Dense
from keras.models import Model as KerasModel
from keras.optimizers import SGD
from keras.utils import to_categorical
from sklearn.metrics import confusion_matrix, roc_curve, auc, roc_auc_score

from base_classifier import BaseClassifier
from dataset import DataSet
from nideep.datasets.amfed.amfed import AMFED
from nideep.datasets.celeba.celeba import CelebA


class KerasInceptionClassifier(BaseClassifier):
    def __init__(self, sess, gan, cache_dir='/mnt/raid/data/ni/dnn/rparra/cache/', global_step=-1):
        self.sess = sess
        self.cache_dir = cache_dir
        self.gan = gan
        self.global_step = global_step

    def get_dataset(self, config, imbalance_proportion=None, train_proportion=0.8, cache=True):
        if config.dataset == 'amfed':
            dataset = AMFED(dir_prefix='/mnt/raid/data/ni/dnn/AMFED/', video_type=AMFED.VIDEO_TYPE_AVI,
                            cache_dir=self.cache_dir)

            (X_train, y_train, videos_train, _, X_test, y_test, _, _) = dataset.as_numpy_array(
                train_proportion=train_proportion)
            X_train = np.flip((X_train.astype(np.float32) - 127.5) / 127.5, axis=-1)
            X_test = np.flip((X_test.astype(np.float32) - 127.5) / 127.5, axis=-1)
            y_train = np.squeeze(y_train)
            y_test = np.squeeze(y_test)
        else:
            dataset = CelebA(dir_prefix='/mnt/antares_raid/home/rparra/workspace/DCGAN-tensorflow/data/celebA',
                             cache_dir='/mnt/raid/data/ni/dnn/rparra/cache/')
            y_train, X_train, y_test, X_test = dataset.as_numpy_array(train_proportion=train_proportion,
                                                                      imbalance_proportion=imbalance_proportion,
                                                                      cache=cache)
            y_train = np.squeeze(y_train)
            y_test = np.squeeze(y_test)
            videos_train = [1]

        return X_train, X_test, y_train, y_test, len(set(videos_train))

    def get_classifier(self, config):
        # param_grid = {'C': [2 ** x for x in range(-3, 5)],
        #               'gamma': [2 ** x for x in range(-3, 5)]}
        # return GridSearchCV(SVC(), param_grid)
        return Model(config)

    def evaluate(self, config):
        ooc = config.dataset == 'celeba'
        X_train, X_test, y_train, y_test, video_number = self.get_dataset(config)
        X_train_balanced, X_test_balanced, y_train_balanced, y_test_balanced, _ = self.get_dataset(config,
                                                                                                   imbalance_proportion=None,
                                                                                                   train_proportion=0.8,
                                                                                                   cache=False)

        start = time.time()
        X_train_oversampled, y_train_oversampled = self.oversample(X_train, y_train)
        end = time.time()
        print('Oversampling done in ' + str(end - start) + 's')
        start = time.time()
        X_train_augmented, y_train_augmented = self.get_augmented_dataset(X_train, y_train, config, video_number,
                                                                          ooc=ooc)
        end = time.time()
        print('GAN sampling done in ' + str(end - start) + 's')
        start = time.time()
        X_test_generated, y_test_generated = self.generate_dataset(X_test.shape[0], config, ooc=ooc)
        end = time.time()
        print('GAN sampling of test dataset done in ' + str(end - start) + 's')

        results = []

        clf = self.get_classifier(config)
        print 'Evaluating unbalanced dataset'
        clf.fit(X_train, y_train)
        y_score = clf.predict(X_test)
        # auc_result = self.save_roc(y_test, y_score, 'imbalanced_roc.p')
        acc, auc, f1, bacc, avgp, pacc, nacc = self.get_metrics(y_test, y_score)
        results.append(self.build_result(acc, auc, f1, bacc, avgp, pacc, nacc, 'imbalanced'))

        clf = self.get_classifier(config)
        print 'Evaluating oversampled dataset'
        clf.fit(X_train_oversampled, y_train_oversampled)
        y_score = clf.predict(X_test)
        # auc_result = self.save_roc(y_test, y_score, 'oversampled_roc.p')
        acc, auc, f1, bacc, avgp, pacc, nacc = self.get_metrics(y_test, y_score)
        results.append(self.build_result(acc, auc, f1, bacc, avgp, pacc, nacc, 'oversampled'))

        clf = self.get_classifier(config)
        print 'Evaluating augmented dataset'
        clf.fit(X_train_augmented, y_train_augmented)
        y_score = clf.predict(X_test)
        # auc_result = self.save_roc(y_test, y_score, 'augmented_roc.p')
        acc, auc, f1, bacc, avgp, pacc, nacc = self.get_metrics(y_test, y_score)
        results.append(self.build_result(acc, auc, f1, bacc, avgp, pacc, nacc, 'augmented'))

        clf = self.get_classifier(config)
        print 'Evaluating synthesized dataset'
        clf.fit(X_train_balanced, y_train_balanced)
        y_score = clf.predict(X_test_generated)
        # auc_result = self.save_roc(y_test, y_score, 'imbalanced_roc.p')
        acc, auc, f1, bacc, avgp, pacc, nacc = self.get_metrics(y_test_generated, y_score)
        results.append(self.build_result(acc, auc, f1, bacc, avgp, pacc, nacc, 'synthesized'))

        y_score = clf.predict(X_test_balanced)
        # auc_result = self.save_roc(y_test, y_score, 'imbalanced_roc.p')
        acc, auc, f1, bacc, avgp, pacc, nacc = self.get_metrics(y_test_balanced, y_score)
        results.append(self.build_result(acc, auc, f1, bacc, avgp, pacc, nacc, 'balanced'))

        return pd.DataFrame(results)

    def save_roc(self, y_test, y_score, name):
        fpr, tpr, thresholds = roc_curve(y_test, y_score)
        roc_auc = auc(fpr, tpr)
        result = {'fpr': fpr, 'tpr': tpr, 'thresholds': thresholds, 'auc': roc_auc}
        pickle.dump(result, open(name, "wb"))
        auc_result = roc_auc_score(y_test, y_score)
        print 'AUC: %s' % str(auc_result)
        return auc_result


class Model(object):
    def __init__(self, config):
        # create the base pre-trained model
        self.base_model = VGG16(weights='imagenet', include_top=False,
                                input_shape=(config.output_height, config.output_width, 3))
        self.num_classes = 2
        self.ooc = config.dataset == 'celeba'
        self.config = config
        # add a global spatial average pooling layer
        x = self.base_model.output
        x = GlobalAveragePooling2D()(x)
        # let's add a fully-connected layer
        x = Dense(1024, activation='relu')(x)
        # and a logistic layer -- let's say we have 200 classes
        predictions = Dense(self.num_classes, activation='softmax')(x)

        # this is the model we will train
        self.model = KerasModel(inputs=self.base_model.input, outputs=predictions)
        self.batch_size = config.batch_size

    def fit(self, X_train, y_train, epochs=5):
        # X_train = X_train.repeat(3, axis=1).repeat(3, axis=2)
        y_binary = to_categorical(y_train)
        # first: train only the top layers (which were randomly initialized)
        # i.e. freeze all convolutional InceptionV3 layers
        for layer in self.base_model.layers:
            layer.trainable = False

        # compile the model (should be done *after* setting layers to non-trainable)
        self.model.compile(optimizer='rmsprop', loss='categorical_crossentropy')

        # train the model on the new data for a few epochs
        if self.ooc:
            data = DataSet(X_train, y_binary, self.config)

            def train_generator():
                while True:
                    yield data.next_batch(self.config.batch_size)

            self.model.fit_generator(train_generator(), data.num_examples // self.config.batch_size, epochs)
        else:
            self.model.fit(X_train, y_binary, epochs=epochs, batch_size=self.config.batch_size)

        # at this point, the top layers are well trained and we can start fine-tuning
        # convolutional layers from inception V3. We will freeze the bottom N layers
        # and train the remaining top layers.

        # let's visualize layer names and layer indices to see how many layers
        # we should freeze:
        for i, layer in enumerate(self.base_model.layers):
            print(i, layer.name)

        # we chose to train the top 2 inception blocks, i.e. we will freeze
        # the first 249 layers and unfreeze the rest:
        for layer in self.model.layers[:15]:
            layer.trainable = False
        for layer in self.model.layers[15:]:
            layer.trainable = True

        self.model.compile(optimizer=SGD(lr=0.001, momentum=0.9), loss='categorical_crossentropy')

        # we train our model again (this time fine-tuning the top 2 inception blocks
        # alongside the top Dense layers
        if self.ooc:
            data = DataSet(X_train, y_binary, self.config)

            def train_generator():
                while True:
                    yield data.next_batch(self.config.batch_size)

            self.model.fit_generator(train_generator(), data.num_examples // self.config.batch_size, epochs)
        else:
            self.model.fit(X_train, y_binary, epochs=epochs, batch_size=self.config.batch_size)

    def predict(self, X_test, probability=True):
        # X_test = X_test.repeat(3, axis=1).repeat(3, axis=2)
        if self.ooc:
            test_data = DataSet(X_test, np.zeros([X_test.shape[0]]), config=self.config)
            preds = []
            while test_data.epochs_completed == 0:
                x_test_batch, y_test_batch = test_data.next_batch(self.config.batch_size)
                preds.append(self.model.predict_on_batch(x_test_batch))
            probs = np.concatenate(preds)[:X_test.shape[0]]
        else:
            probs = self.model.predict(X_test)
        return probs[:, 1] if probability else np.argmax(probs, axis=1)
