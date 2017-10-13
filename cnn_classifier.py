import os

import numpy as np
import pandas as pd
import scipy
import tensorflow as tf
from scipy import misc
from scipy import ndimage
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.utils import shuffle

from dataset import DataSet
from nideep.datasets.amfed.amfed import AMFED
from nideep.datasets.celeba.celeba import CelebA


class CNNClassifier(object):
    def __init__(self, sess, gan, cache_dir='/mnt/raid/data/ni/dnn/rparra/cache/', global_step=-1):
        self.sess = sess
        self.cache_dir = cache_dir
        self.gan = gan
        self.num_classes = 2
        self.AUGEMENTED = 'xaugment.dat'
        self.global_step = global_step

    def get_dataset(self, config):
        if config.dataset == 'amfed':
            dataset = AMFED(dir_prefix='/mnt/raid/data/ni/dnn/AMFED/', video_type=AMFED.VIDEO_TYPE_AVI,
                            cache_dir=self.cache_dir)

            (X_train, y_train, videos_train, _, X_test, y_test, _, _) = dataset.as_numpy_array(train_proportion=0.8)
        else:
            dataset = CelebA(dir_prefix='/mnt/antares_raid/home/rparra/workspace/DCGAN-tensorflow/data/celebA',
                             cache_dir='/mnt/raid/data/ni/dnn/rparra/cache/')
            y_train, X_train, y_test, X_test = dataset.as_numpy_array(imbalance_proportion=0.1)
            videos_train = [1]

        return X_train, X_test, y_train, y_test, len(set(videos_train))

    def evaluate(self, config, teardown=False):
        ooc = config.dataset == 'celeba'
        X_train, X_test, y_train, y_test, video_number = self.get_dataset(config)
        y_train = np.squeeze(y_train)
        y_test = np.squeeze(y_test)

        if not ooc:
            X_train = (X_train.astype(np.float32) - 127.5) / 127.5
            X_test = (X_test.astype(np.float32) - 127.5) / 127.5

        results = []
        X_train_oversampled, y_train_oversampled = self.oversample(X_train, y_train)
        X_train_augmented, y_train_augmented = self.get_augmented_dataset(X_train, y_train, config, video_number, ooc)

        model = Model(X_train, y_train, self.sess, config, 'cnn_clf_imbalanced')
        model.optimize(num_iterations=10000)
        auc, f1, bacc = model.evaluate(X_test, y_test, ooc=ooc)
        results.append(self._build_result(auc, f1, bacc, 'imbalanced'))

        model = Model(X_train_oversampled, y_train_oversampled, self.sess, config, 'cnn_clf_oversampled')
        model.optimize(num_iterations=10000)
        auc, f1, bacc = model.evaluate(X_test, y_test, ooc=ooc)
        results.append(self._build_result(auc, f1, bacc, 'oversampled'))

        model = Model(X_train_augmented, y_train_augmented, self.sess, config, 'cnn_clf_augmented')
        model.optimize(num_iterations=10000)
        auc, f1, bacc = model.evaluate(X_test, y_test, ooc=ooc)
        results.append(self._build_result(auc, f1, bacc, 'augmented'))

        if teardown and not ooc:
            tmp = os.path.join(self.cache_dir, self.AUGEMENTED)
            os.remove(tmp)
        return pd.DataFrame(results)

    def _build_result(self, auc, f1, bacc, setting):
        return {'classifier': 'custom_cnn', 'dataset': setting, 'auc': auc, 'f1': f1, 'bacc': bacc,
                'global_step': self.global_step}

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
            suffix = "xnoisy_%s.png" % (i, )
            path = os.path.join(self.cache_dir, suffix)
            scipy.misc.imsave(path, noisy)
            result.append(path)
        return np.array(result)

    def get_augmented_dataset(self, X_train, y_train, config, video_number, ooc=False):
        tmp = os.path.join(self.cache_dir, self.AUGEMENTED)
        if os.path.exists(tmp):
            X_gen = np.memmap(tmp, dtype='float32').reshape((-1, self.input_height, self.input_width, self.c_dim))
            y_gen = np.ones((X_gen.shape[0],))
        else:
            X_gen, y_gen = self.augment(y_train, config, tmp, video_number, ooc)
        X_augmented = np.concatenate((X_train, X_gen))
        y_augmented = np.concatenate((y_train, y_gen))
        p = np.random.permutation(y_augmented.shape[0])
        return X_augmented[p], y_augmented[p]

        return np.concatenate((X_train, X_gen)), np.concatenate((y_train, y_gen))

    def augment(self, y_train, config, tmp_path, video_number, ooc):
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
                    suffix = "_%s_%s.png" % (i, j)
                    path = tmp_path[:-4] + suffix
                    scipy.misc.imsave(path, sample)
                    result.append(path)
            else:
                result.append(samples)

        X_augmented = result if ooc else np.concatenate(result)
        y_augmented = np.ones((config.batch_size * sample_size,))

        if ooc:
            X_augmented_memmap = X_augmented
        else:
            X_augmented_memmap = np.memmap(tmp_path, shape=X_augmented.shape, dtype='float32', mode='w+')
            X_augmented_memmap[:] = X_augmented[:]

        return X_augmented_memmap, y_augmented


class Model(object):
    def __init__(self, X_train, y_train, sess, config, name):
        self.sess = sess
        self.config = config
        self.input_height = config.output_height
        self.input_width = config.output_width

        self.y_dim = 1
        self.c_dim = config.c_dim
        # Convolutional Layer 1.
        self.filter_size1 = 3
        self.num_filters1 = 32

        # Convolutional Layer 2.
        self.filter_size2 = 3
        self.num_filters2 = 32

        # Convolutional Layer 3.
        self.filter_size3 = 3
        self.num_filters3 = 64

        # Fully-connected layer.
        self.fc_size = 128  # Number of neurons in fully-connected layer.

        # Number of color channels for the images: 1 channel for gray-scale.
        self.num_channels = 3

        # Size of image when flattened to a single dimension
        self.img_size_flat = self.input_width * self.input_height * self.c_dim

        # Tuple with height and width of images used to reshape arrays.
        self.img_shape = (self.input_width, self.input_height)

        # class info
        self.num_classes = 2

        # batch size
        self.batch_size = config.batch_size

        # validation split
        self.validation_size = .2

        # how long to wait after validation loss stops improving before terminating training
        self.early_stopping = None  # use None if you don't want to implement early stoping
        self.data = self.read_train_sets(X_train, y_train, config, validation_size=0.2)

        self.x = tf.placeholder(tf.float32, shape=[None, self.img_size_flat], name='x')
        x_image = tf.reshape(self.x, [-1, self.input_height, self.input_width, self.c_dim])

        # self.y_true = tf.placeholder(tf.float32, shape=[None, self.num_classes], name='y_true')
        self.y_true = tf.placeholder(tf.int64, shape=[None], name='y_true')
        self.y_true_cls = self.y_true
        # self.y_true_cls = tf.argmax(self.y_true, dimension=1)

        with tf.variable_scope(name):
            layer_conv1, weights_conv1 = \
                self.new_conv_layer(input=x_image,
                                    num_input_channels=self.c_dim,
                                    filter_size=self.filter_size1,
                                    num_filters=self.num_filters1,
                                    name='conv1',
                                    use_pooling=True)
            # print("now layer2 input")
            # print(layer_conv1.get_shape())
            layer_conv2, weights_conv2 = \
                self.new_conv_layer(input=layer_conv1,
                                    num_input_channels=self.num_filters1,
                                    filter_size=self.filter_size2,
                                    num_filters=self.num_filters2,
                                    name='conv2',
                                    use_pooling=True)
            # print("now layer3 input")
            # print(layer_conv2.get_shape())

            layer_conv3, weights_conv3 = \
                self.new_conv_layer(input=layer_conv2,
                                    num_input_channels=self.num_filters2,
                                    filter_size=self.filter_size3,
                                    num_filters=self.num_filters3,
                                    name='conv3',
                                    use_pooling=True)
            # print("now layer flatten input")
            # print(layer_conv3.get_shape())

            layer_flat, num_features = self.flatten_layer(layer_conv3)

            layer_fc1 = self.new_fc_layer(input=layer_flat,
                                          num_inputs=num_features,
                                          num_outputs=self.fc_size,
                                          name='fc1',
                                          use_relu=True)

            layer_fc2 = self.new_fc_layer(input=layer_fc1,
                                          num_inputs=self.fc_size,
                                          num_outputs=self.num_classes,
                                          name='fc2',
                                          use_relu=False)

        self.y_pred = tf.nn.softmax(layer_fc2, name='y_pred')

        self.y_pred_cls = tf.argmax(self.y_pred, dimension=1)
        # cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=layer_fc2,
        #                                                        labels=self.y_true)
        self.cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=layer_fc2, labels=self.y_true)
        self.cost = tf.reduce_mean(self.cross_entropy)

        temp = set(tf.all_variables())
        self.optimizer = tf.train.AdamOptimizer(learning_rate=1e-5).minimize(self.cost)
        correct_prediction = tf.equal(self.y_pred_cls, self.y_true_cls)
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        self.confusion_matrix = tf.confusion_matrix(self.y_true_cls, self.y_pred_cls, num_classes=self.num_classes)
        self.auc = tf.metrics.auc(self.y_true_cls, self.y_pred_cls)
        new_vars = set(tf.all_variables()) - temp

        # As hacky as it gets
        self.sess.run(tf.initialize_variables(new_vars))
        self.sess.run(tf.local_variables_initializer())
        with tf.variable_scope(name, reuse=True):
            uninitialized_variables = [tf.get_variable(name.split('/', 1)[-1]) for name in
                                       self.sess.run(tf.report_uninitialized_variables())]
            self.sess.run(tf.initialize_variables(uninitialized_variables))

        # self.sess.run(tf.global_variables_initializer())  # for newer versions
        # self.sess.run(tf.local_variables_initializer())  # for newer versions
        # self.sess.run(tf.initialize_all_variables())  # for older versions
        self.train_batch_size = self.batch_size

    def new_weights(self, shape, name):
        with tf.variable_scope(name):
            return tf.get_variable(name='w', initializer=tf.truncated_normal(shape, stddev=0.05))

    def new_biases(self, length, name):
        with tf.variable_scope(name):
            return tf.get_variable(name='b', initializer=tf.truncated_normal([length], stddev=0.05))

    def new_conv_layer(self, input,  # The previous layer.
                       num_input_channels,  # Num. channels in prev. layer.
                       filter_size,  # Width and height of each filter.
                       num_filters,  # Number of filters.
                       name,  # name for variable scope
                       use_pooling=True):  # Use 2x2 max-pooling.

        # Shape of the filter-weights for the convolution.
        # This format is determined by the TensorFlow API.
        shape = [filter_size, filter_size, num_input_channels, num_filters]

        # Create new weights aka. filters with the given shape.
        weights = self.new_weights(shape=shape, name=name)

        # Create new biases, one for each filter.
        biases = self.new_biases(length=num_filters, name=name)

        # Create the TensorFlow operation for convolution.
        # Note the strides are set to 1 in all dimensions.
        # The first and last stride must always be 1,
        # because the first is for the image-number and
        # the last is for the input-channel.
        # But e.g. strides=[1, 2, 2, 1] would mean that the filter
        # is moved 2 pixels across the x- and y-axis of the image.
        # The padding is set to 'SAME' which means the input image
        # is padded with zeroes so the size of the output is the same.
        layer = tf.nn.conv2d(input=input,
                             filter=weights,
                             strides=[1, 1, 1, 1],
                             padding='SAME')

        # Add the biases to the results of the convolution.
        # A bias-value is added to each filter-channel.
        layer += biases

        # Use pooling to down-sample the image resolution?
        if use_pooling:
            # This is 2x2 max-pooling, which means that we
            # consider 2x2 windows and select the largest value
            # in each window. Then we move 2 pixels to the next window.
            layer = tf.nn.max_pool(value=layer,
                                   ksize=[1, 2, 2, 1],
                                   strides=[1, 2, 2, 1],
                                   padding='SAME')

        # Rectified Linear Unit (ReLU).
        # It calculates max(x, 0) for each input pixel x.
        # This adds some non-linearity to the formula and allows us
        # to learn more complicated functions.
        layer = tf.nn.relu(layer)

        # Note that ReLU is normally executed before the pooling,
        # but since relu(max_pool(x)) == max_pool(relu(x)) we can
        # save 75% of the relu-operations by max-pooling first.

        # We return both the resulting layer and the filter-weights
        # because we will plot the weights later.
        return layer, weights

    def flatten_layer(self, layer):
        # Get the shape of the input layer.
        layer_shape = layer.get_shape()

        # The shape of the input layer is assumed to be:
        # layer_shape == [num_images, img_height, img_width, num_channels]

        # The number of features is: img_height * img_width * num_channels
        # We can use a function from TensorFlow to calculate this.
        num_features = layer_shape[1:4].num_elements()

        # Reshape the layer to [num_images, num_features].
        # Note that we just set the size of the second dimension
        # to num_features and the size of the first dimension to -1
        # which means the size in that dimension is calculated
        # so the total size of the tensor is unchanged from the reshaping.
        layer_flat = tf.reshape(layer, [-1, num_features])

        # The shape of the flattened layer is now:
        # [num_images, img_height * img_width * num_channels]

        # Return both the flattened layer and the number of features.
        return layer_flat, num_features

    def new_fc_layer(self, input,  # The previous layer.
                     num_inputs,  # Num. inputs from prev. layer.
                     num_outputs,  # Num. outputs.
                     name,  # name for variable scope
                     use_relu=True):  # Use Rectified Linear Unit (ReLU)?

        # Create new weights and biases.
        weights = self.new_weights(shape=[num_inputs, num_outputs], name=name)
        biases = self.new_biases(length=num_outputs, name=name)

        # Calculate the layer as the matrix multiplication of
        # the input and weights, and then add the bias-values.
        layer = tf.matmul(input, weights) + biases

        # Use ReLU?
        if use_relu:
            layer = tf.nn.relu(layer)

        return layer

    def optimize(self, num_iterations):
        best_val_loss = float("inf")

        for i in range(num_iterations):
            # Get a batch of training examples.
            # x_batch now holds a batch of images and
            # y_true_batch are the true labels for those images.
            x_batch, y_true_batch = self.data.train.next_batch(self.train_batch_size)
            x_valid_batch, y_valid_batch = self.data.valid.next_batch(self.train_batch_size)

            # Convert shape from [num examples, rows, columns, depth]
            # to [num examples, flattened image shape]
            x_batch = x_batch.reshape(self.train_batch_size, self.img_size_flat)
            x_valid_batch = x_valid_batch.reshape(self.train_batch_size, self.img_size_flat)
            # Put the batch into a dict with the proper names
            # for placeholder variables in the TensorFlow graph.
            feed_dict_train = {self.x: x_batch, self.y_true: y_true_batch}

            feed_dict_validate = {self.x: x_valid_batch, self.y_true: y_valid_batch}

            # Run the optimizer using this batch of training data.
            # TensorFlow assigns the variables in feed_dict_train
            # to the placeholder variables and then runs the optimizer.
            self.sess.run(self.optimizer, feed_dict=feed_dict_train)
            # saver = tf.train.Saver()
            # saver.save(self.sess, 'my_test_model')

            # Print status at end of each epoch (defined as full pass through training dataset).
            if i % int(self.data.train.num_examples / self.batch_size) == 0:
                val_loss = self.sess.run(self.cost, feed_dict=feed_dict_validate)
                epoch = int(i / int(self.data.train.num_examples / self.batch_size))

                self.print_progress(epoch, feed_dict_train, feed_dict_validate, val_loss)

                # if i % 100 == 0:
                #     val_loss = self.sess.run(self.cost, feed_dict=feed_dict_validate)
                #     epoch = int(i/100)
                #     self.print_progress(epoch, feed_dict_train, feed_dict_validate, val_loss)

    def print_progress(self, epoch, feed_dict_train, feed_dict_validate, val_loss):
        # Calculate the accuracy on the training-set.
        acc = self.sess.run(self.accuracy, feed_dict=feed_dict_train)
        val_acc = self.sess.run(self.accuracy, feed_dict=feed_dict_validate)
        msg = "Epoch {0} --- Training Accuracy: {1:>6.1%}, Validation Accuracy: {2:>6.1%}, Validation Loss: {3:.3f}"
        print(msg.format(epoch + 1, acc, val_acc, val_loss))

    def read_train_sets(self, X, y, config, validation_size=0):
        class DataSets(object):
            pass

        data_sets = DataSets()

        images, labels = shuffle(X, y)  # shuffle the data

        if isinstance(validation_size, float):
            validation_size = int(validation_size * images.shape[0])

        validation_images = images[:validation_size]
        validation_labels = labels[:validation_size]

        train_images = images[validation_size:]
        train_labels = labels[validation_size:]

        data_sets.train = DataSet(train_images, train_labels, config)
        data_sets.valid = DataSet(validation_images, validation_labels, config)

        return data_sets

    def evaluate(self, X_test, y_test, ooc):
        test_data = DataSet(X_test, y_test, config=self.config)
        conf_mat = np.zeros((self.num_classes, self.num_classes))
        labels = []
        pred_scores = []
        pred_labels = []
        while test_data.epochs_completed == 0:
            x_test_batch, y_test_batch = test_data.next_batch(self.train_batch_size)
            x_test_batch = x_test_batch.reshape(self.train_batch_size, self.img_size_flat)
            feed_dict_test = {self.x: x_test_batch, self.y_true: y_test_batch}
            cm, y_pred = self.sess.run([self.confusion_matrix, self.y_pred], feed_dict=feed_dict_test)
            conf_mat += cm
            pred_scores.append(y_pred[:, 1])
            pred_labels.append(np.argmax(y_pred, axis=1))
            labels.append(y_test_batch)
        return self.get_metrics(np.concatenate(labels), np.concatenate(pred_labels), np.concatenate(pred_scores),
                                conf_mat)

    def get_metrics(self, y_true, y_pred, y_score, conf_mat):
        auc = roc_auc_score(y_true, y_score)
        f1 = f1_score(y_true, y_pred)
        bacc = self.balanced_accuracy(conf_mat)
        print conf_mat
        print 'AUC: ' + str(auc)
        print 'F1:' + str(f1)
        print 'BACC:' + str(bacc)
        return auc, f1, bacc

    def balanced_accuracy(self, conf_mat):
        return (float(conf_mat[0][0]) / sum(conf_mat[0]) + float(conf_mat[1][1]) / sum(conf_mat[1])) / 2.0
