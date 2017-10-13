import os
import pickle

import numpy as np
import scipy
from keras.applications import VGG16
from keras.layers import GlobalAveragePooling2D, Dense
from keras.optimizers import SGD
from keras.models import Model as KerasModel
from keras.utils import to_categorical
from sklearn.metrics import confusion_matrix, roc_curve, auc, roc_auc_score, average_precision_score, accuracy_score
from sklearn.metrics import f1_score

from scipy import misc
from scipy import ndimage
from dataset import DataSet
from nideep.datasets.amfed.amfed import AMFED
import pandas as pd
import time

from nideep.datasets.celeba.celeba import CelebA


class KerasInceptionClassifier(object):
    def __init__(self, sess, gan, cache_dir='/mnt/raid/data/ni/dnn/rparra/cache/', global_step=-1):
        self.sess = sess
        self.cache_dir = cache_dir
        self.gan = gan
        self.global_step = global_step

    def get_dataset(self, config, imbalance_proportion=0.1, train_proportion=0.8):
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
                                                                      imbalance_proportion=imbalance_proportion)
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
        X_train_balanced, X_test_balanced, y_train_balanced, y_test_balanced, _ = self.get_dataset(config, imbalance_proportion=1.0,
                                                                             train_proportion=0.8)

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
        y_pred = clf.predict(X_test, probability=False)
        y_score = clf.predict(X_test)
        # auc_result = self.save_roc(y_test, y_score, 'imbalanced_roc.p')
        conf_mat = confusion_matrix(y_test, y_pred)
        acc, auc, f1, bacc, avgp = self.get_metrics(y_test, y_pred, y_score, conf_mat)
        results.append(self._build_result(acc, auc, f1, bacc, avgp, 'imbalanced'))

        clf = self.get_classifier(config)
        print 'Evaluating oversampled dataset'
        clf.fit(X_train_oversampled, y_train_oversampled)
        y_pred = clf.predict(X_test, probability=False)
        y_score = clf.predict(X_test)
        # auc_result = self.save_roc(y_test, y_score, 'oversampled_roc.p')
        conf_mat = confusion_matrix(y_test, y_pred)
        acc, auc, f1, bacc, avgp = self.get_metrics(y_test, y_pred, y_score, conf_mat)
        results.append(self._build_result(acc, auc, f1, bacc, avgp, 'oversampled'))

        clf = self.get_classifier(config)
        print 'Evaluating augmented dataset'
        clf.fit(X_train_augmented, y_train_augmented)
        y_pred = clf.predict(X_test, probability=False)
        y_score = clf.predict(X_test)
        # auc_result = self.save_roc(y_test, y_score, 'augmented_roc.p')
        conf_mat = confusion_matrix(y_test, y_pred)
        acc, auc, f1, bacc, avgp = self.get_metrics(y_test, y_pred, y_score, conf_mat)
        results.append(self._build_result(acc, auc, f1, bacc, avgp, 'augmented'))

        clf = self.get_classifier(config)
        print 'Evaluating synthesized dataset'
        clf.fit(X_train_balanced, y_train_balanced)
        y_pred = clf.predict(X_test_generated, probability=False)
        y_score = clf.predict(X_test_generated)
        # auc_result = self.save_roc(y_test, y_score, 'imbalanced_roc.p')
        conf_mat = confusion_matrix(y_test_generated, y_pred)
        acc, auc, f1, bacc, avgp = self.get_metrics(y_test_generated, y_pred, y_score, conf_mat)
        results.append(self._build_result(acc, auc, f1, bacc, avgp, 'synthesized'))

        y_pred = clf.predict(X_test_balanced, probability=False)
        y_score = clf.predict(X_test_balanced)
        # auc_result = self.save_roc(y_test, y_score, 'imbalanced_roc.p')
        conf_mat = confusion_matrix(y_test_balanced, y_pred)
        acc, auc, f1, bacc, avgp = self.get_metrics(y_test_balanced, y_pred, y_score, conf_mat)
        results.append(self._build_result(acc, auc, f1, bacc, avgp, 'balanced'))


        return pd.DataFrame(results)

    def _build_result(self, acc, auc, f1, bacc, avgp, setting):
        return {'classifier': 'vgg_pretrained', 'dataset': setting, 'acc': acc, 'auc': auc, 'f1': f1, 'bacc': bacc, 'avgp': avgp,
                'global_step': self.global_step}

    def save_roc(self, y_test, y_score, name):
        fpr, tpr, thresholds = roc_curve(y_test, y_score)
        roc_auc = auc(fpr, tpr)
        result = {'fpr': fpr, 'tpr': tpr, 'thresholds': thresholds, 'auc': roc_auc}
        pickle.dump(result, open(name, "wb"))
        auc_result = roc_auc_score(y_test, y_score)
        print 'AUC: %s' % str(auc_result)
        return auc_result

    def generate_dataset(self, sample_size, config, ooc, train_proportion=0.8):
        y_one_hot = np.zeros((config.batch_size, 1))
        y_one_hot[1::2] += 1
        result = []
        labels = []
        for i in range(sample_size / config.batch_size):
            z_sample = np.random.uniform(-1, 1, size=[int(config.batch_size), self.gan.z_dim])
            samples = self.sess.run(self.gan.sampler, feed_dict={self.gan.z: z_sample, self.gan.y: y_one_hot})
            labels.append(y_one_hot)
            if ooc:
                for j, sample in enumerate(samples):
                    suffix = "xgenerated_%s_%s.png" % (i, j)
                    path = os.path.join(self.cache_dir, suffix)
                    scipy.misc.imsave(path, sample)
                    result.append(path)
            else:
                result.append(samples)

        X_gen = np.array(result) if ooc else np.concatenate(result)
        y_gen = np.concatenate(labels)
        return X_gen, np.squeeze(y_gen)

    def get_augmented_dataset(self, X_train, y_train, config, video_number, ooc=False):
        X_gen, y_gen = self.augment(y_train, config, video_number, ooc)
        X_augmented = np.concatenate((X_train, X_gen))
        y_augmented = np.concatenate((y_train, y_gen))
        p = np.random.permutation(y_augmented.shape[0])
        return X_augmented[p], y_augmented[p]

    def augment(self, y_train, config, video_number, ooc=False):
        selected_indices = np.where(y_train == 1)
        sample_size = (y_train.shape[0] - 2 * selected_indices[0].shape[0]) / config.batch_size
        y_one_hot = np.ones((config.batch_size, 1))
        y_video_label = np.random.choice(video_number, (config.batch_size, 1)) / float(video_number)
        y_sample = y_one_hot if ooc else np.concatenate([y_one_hot, y_video_label], axis=1)

        result = []
        for i in range(sample_size):
            z_sample = np.random.uniform(-1, 1, size=[int(config.batch_size), self.gan.z_dim])
            samples = self.sess.run(self.gan.sampler, feed_dict={self.gan.z: z_sample, self.gan.y: y_sample})
            if ooc:
                for j, sample in enumerate(samples):
                    suffix = "xaugment_%s_%s.png" % (i, j)
                    path = os.path.join(self.cache_dir, suffix)
                    scipy.misc.imsave(path, sample)
                    result.append(path)
            else:
                result.append(samples)

        X_augmented = np.array(result) if ooc else np.concatenate(result)
        y_augmented = np.ones((config.batch_size * sample_size,))

        return X_augmented, y_augmented

    def oversample(self, X_train, y_train, noisy=True):
        selected_indices = np.where(y_train == 1)
        sample_size = y_train.shape[0] - 2 * selected_indices[0].shape[0]
        oversampled_indices = np.random.choice(selected_indices[0], sample_size)
        if noisy:
            X_oversampled = np.concatenate((X_train, self.noisify(X_train[oversampled_indices])))
        else:
            X_oversampled = np.concatenate((X_train, X_train[oversampled_indices]))
        y_oversampled = np.concatenate((y_train, y_train[oversampled_indices]))
        p = np.random.permutation(y_oversampled.shape[0])
        return X_oversampled[p], y_oversampled[p]

    def noisify(self, X_train):
        result = []
        for i, f in enumerate(X_train):
            x = misc.imread(f)
            flipped = np.fliplr(x)
            noisy = ndimage.gaussian_filter1d(flipped, sigma=3, axis=0)
            noisy = ndimage.gaussian_filter1d(noisy, sigma=3, axis=1)
            suffix = "xnoisy_%s.png" % (i,)
            path = os.path.join(self.cache_dir, suffix)
            scipy.misc.imsave(path, noisy)
            result.append(path)
        return np.array(result)

    def get_metrics(self, y_true, y_pred, y_score, conf_mat):
        auc = roc_auc_score(y_true, y_score)
        f1 = f1_score(y_true, y_pred)
        acc = accuracy_score(y_true, y_pred)
        bacc = self.balanced_accuracy(conf_mat)
        avgp = average_precision_score(y_true, y_score)
        print conf_mat
        print 'ACC:' + str(acc)
        print 'AUC: ' + str(auc)
        print 'F1:' + str(f1)
        print 'BACC:' + str(bacc)
        print 'AVGP:' + str(avgp)
        return acc, auc, f1, bacc, avgp

    def balanced_accuracy(self, conf_mat):
        return (float(conf_mat[0][0]) / sum(conf_mat[0]) + float(conf_mat[1][1]) / sum(conf_mat[1])) / 2.0


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

    def fit(self, X_train, y_train, epochs=10):
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

        self.model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy')

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
