from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

from nideep.datasets.amfed.amfed import AMFED
import numpy as np
import os
import sys
import re
from six.moves import urllib
import tarfile
import tensorflow as tf
from tensorflow.python.platform import gfile
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc, roc_auc_score
import pickle

DATA_URL = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'


class InceptionClassifier(object):
    PREPROCESSED_TRAIN = 'xtrain_preprocessed.dat'
    PREPROCESSED_AUGEMENTED = 'xaugment_preprocessed.dat'
    PREPROCESSED_TEST = 'xtest_preprocessed.dat'
    NB_FEATURES = 2048

    def __init__(self, sess, gan, cache_dir='/mnt/raid/data/ni/dnn/rparra/cache/', input_height=64, input_width=64,
                 y_dim=1, c_dim=3):
        self.sess = sess
        self.input_height = input_height
        self.input_width = input_width
        self.y_dim = y_dim
        self.c_dim = c_dim
        self.cache_dir = cache_dir
        self.gan = gan
        self.__maybe_download_and_extract()

    def get_dataset(self):
        tmp_train = os.path.join(self.cache_dir, self.PREPROCESSED_TRAIN)
        tmp_test = os.path.join(self.cache_dir, self.PREPROCESSED_TEST)
        dataset = AMFED(dir_prefix='/mnt/raid/data/ni/dnn/AMFED/', video_type=AMFED.VIDEO_TYPE_AVI,
                        cache_dir=self.cache_dir)

        (X_train, y_train, _, _, X_test, y_test, _, _) = dataset.as_numpy_array(train_proportion=0.8)

        if os.path.exists(tmp_train) and os.path.exists(tmp_test):
            X_train_memmap = np.memmap(tmp_train, dtype='float32').reshape((-1, self.NB_FEATURES))
            X_test_memmap = np.memmap(tmp_test, dtype='float32').reshape((-1, self.NB_FEATURES))
        else:
            print 'got dataset'
            X_train = np.flip((X_train.astype(np.float32) - 127.5) / 127.5, axis=-1)
            X_test = np.flip((X_test.astype(np.float32) - 127.5) / 127.5, axis=-1)
            y_train = y_train.ravel()
            y_test = y_test.ravel()
            print "train dataset shape: " + str(X_train.shape)
            print "test dataset shape: " + str(X_test.shape)

            X_train_preprocessed = self.extract_features(X_train)
            X_test_preprocessed = self.extract_features(X_test)

            X_train_memmap = np.memmap(tmp_train, shape=X_train_preprocessed.shape, mode='w+', dtype='float32')
            X_test_memmap = np.memmap(tmp_test, shape=X_test_preprocessed.shape, mode='w+', dtype='float32')
            X_train_memmap[:] = X_train_preprocessed[:]
            X_test_memmap[:] = X_test_preprocessed[:]

        return X_train_memmap, X_test_memmap, y_train, y_test

    def get_classifier(self):
        # param_grid = {'C': [2 ** x for x in range(-3, 5)],
        #               'gamma': [2 ** x for x in range(-3, 5)]}
        # return GridSearchCV(SVC(), param_grid)
        return LogisticRegression(C=1000.0)

    def evaluate(self, config):
        if config.dataset == 'amfed':
            X_train, X_test, y_train, y_test = self.get_dataset()
            y_train = np.squeeze(y_train)
            y_test = np.squeeze(y_test)

            clf = self.get_classifier()
            print 'Evaluating unbalanced dataset'
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            y_score = clf.decision_function(X_test)
            self.save_roc(y_test, y_score,  'unbalanced_roc.p')
            print("Accuracy: {0:0.1f}%".format(accuracy_score(y_test, y_pred) * 100))
            print(confusion_matrix(y_test, y_pred))

            clf = self.get_classifier()
            print 'Evaluating oversampled dataset'
            X_train_oversampled, y_train_oversampled = self.oversample(X_train, y_train)
            clf.fit(X_train_oversampled, y_train_oversampled)
            y_pred = clf.predict(X_test)
            y_score = clf.decision_function(X_test)
            self.save_roc(y_test, y_score, 'oversampled_roc.p')
            print("Accuracy: {0:0.1f}%".format(accuracy_score(y_test, y_pred) * 100))
            print(confusion_matrix(y_test, y_pred))

            clf = self.get_classifier()
            print 'Evaluating augmented dataset'
            X_train_augmented, y_train_augmented = self.get_augmented_dataset(X_train, y_train, config)
            clf.fit(X_train_augmented, y_train_augmented)
            y_pred = clf.predict(X_test)
            y_score = clf.decision_function(X_test)
            self.save_roc(y_test, y_score, 'augmented_roc.p')
            print("Accuracy: {0:0.1f}%".format(accuracy_score(y_test, y_pred) * 100))
            print(confusion_matrix(y_test, y_pred))

    def save_roc(self, y_test, y_score, name):
        fpr, tpr, thresholds = roc_curve(y_test, y_score)
        roc_auc = auc(fpr, tpr)
        result = {'fpr': fpr, 'tpr': tpr, 'thresholds': thresholds, 'auc': roc_auc}
        pickle.dump(result, open(name, "wb"))
        print 'AUC: %s' % str(roc_auc_score(y_test, y_score))

    def get_augmented_dataset(self, X_train, y_train, config):
        tmp = os.path.join(self.cache_dir, self.PREPROCESSED_AUGEMENTED)
        if os.path.exists(tmp):
            X_gen = np.memmap(tmp, dtype='float32').reshape((-1, self.NB_FEATURES))
            y_gen = np.ones((X_gen.shape[0], ))
        else:
            X_gen, y_gen = self.augment(y_train, config, tmp)
        return np.concatenate((X_train, X_gen)), np.concatenate((y_train, y_gen))

    def augment(self, y_train, config, tmp_path):
        selected_indices = np.where(y_train == 1)
        sample_size = (y_train.shape[0] - 2 * selected_indices[0].shape[0]) / config.batch_size
        y_one_hot = np.ones((config.batch_size, 1))
        result = []
        for i in range(sample_size):
            z_sample = np.random.uniform(-1, 1, size=[int(config.batch_size), self.gan.z_dim])
            samples = self.sess.run(self.gan.sampler, feed_dict={self.gan.z: z_sample, self.gan.y: y_one_hot})
            result.append(samples)

        X_augmented = np.concatenate(result)
        X_augmented_preprocessed = self.extract_features(X_augmented)
        y_augmented = np.ones((config.batch_size * sample_size, ))

        X_augmented_memmap = np.memmap(tmp_path, shape=X_augmented_preprocessed.shape, dtype='float32', mode='w+')
        X_augmented_memmap[:] = X_augmented_preprocessed[:]

        return X_augmented_memmap, y_augmented

    def oversample(self, X_train, y_train):
        selected_indices = np.where(y_train == 1)
        sample_size = y_train.shape[0] - 2 * selected_indices[0].shape[0]
        oversampled_indices = np.random.choice(selected_indices[0], sample_size)
        return np.concatenate((X_train, X_train[oversampled_indices])),\
            np.concatenate((y_train, y_train[oversampled_indices]))

    def __maybe_download_and_extract(self):
        """Download and extract model tar file."""
        dest_directory = self.cache_dir
        if not os.path.exists(dest_directory):
            os.makedirs(dest_directory)
        filename = DATA_URL.split('/')[-1]
        filepath = os.path.join(dest_directory, filename)
        if not os.path.exists(filepath):
            def _progress(count, block_size, total_size):
                sys.stdout.write('\r>> Downloading %s %.1f%%' % (
                    filename, float(count * block_size) / float(total_size) * 100.0))
                sys.stdout.flush()

            filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath, _progress)
            print()
            statinfo = os.stat(filepath)
            print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
        tarfile.open(filepath, 'r:gz').extractall(dest_directory)

    def create_graph(self):
        with gfile.FastGFile(os.path.join(self.cache_dir, 'classify_image_graph_def.pb'), 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            _ = tf.import_graph_def(graph_def, name='')

    def extract_features(self, images):
        features = np.empty((images.shape[0], self.NB_FEATURES))
        print 'before create graph'
        self.create_graph()
        print 'after create graph'
        next_to_last_tensor = self.sess.graph.get_tensor_by_name('pool_3:0')

        for ind, image in enumerate(images):
            predictions = self.sess.run(next_to_last_tensor, {'DecodeJpeg:0': image})
            if (ind % 100 == 0):
                print('Finished processing %s/%s images' % (ind, images.shape[0]))

            features[ind, :] = np.squeeze(predictions)

        return features
