from nideep.datasets.celeba.celeba import CelebA

from utils import *
import os


class DiscriminatorEvaluator(object):
    def __init__(self, sess, model, config, cache_dir='/mnt/raid/data/ni/dnn/rparra/cache/'):
        self.sess = sess
        self.model = model
        self.config = config
        self.cache_dir = cache_dir
        self.dataset = CelebA(dir_prefix='/mnt/antares_raid/home/rparra/workspace/DCGAN-tensorflow/data/celebA',
                              cache_dir='/mnt/raid/data/ni/dnn/rparra/cache/')
        self.is_grayscale = (self.config.c_dim == 1)
        self.disc_label = tf.placeholder(tf.float32, [self.config.batch_size, 1], name='disc_y')
        self.prediction_tmp = tf.argmax(self.model.D_logits, 1)
        self.prediction = tf.squeeze(tf.round(self.model.D))

    @staticmethod
    def _keep_by_label(X, y, keep_label):
        return np.where(y == keep_label)[0]

    def evaluate(self):
        y_real_positive, X_real_positive, _, _ = self.dataset.as_numpy_array(
            filter_by=lambda x, y: DiscriminatorEvaluator._keep_by_label(x, y, 1))
        y_real_negative, X_real_negative, _, _ = self.dataset.as_numpy_array(
            filter_by=lambda x, y: DiscriminatorEvaluator._keep_by_label(x, y, 0))
        X_fake_positive = self._get_generated_data(X_real_positive.shape[0], 1)
        X_fake_negative = self._get_generated_data(X_real_negative.shape[0], 1)

        self.evaluate_dataset(X_real_positive, y_real_positive, 'real_positive', True)
        self.evaluate_dataset(X_real_negative, y_real_negative, 'real_negative', True)
        self.evaluate_dataset(X_fake_positive, y_real_positive, 'fake_positive', False)
        self.evaluate_dataset(X_fake_negative, y_real_negative, 'fake_negative', False)

    def evaluate_dataset(self, data, data_y, scope, is_real):
        op, acc, conf = self._get_streaming_metrics(scope, self.prediction, self.disc_label, 2)
        batch_idxs = len(data) // self.config.batch_size

        stream_vars = [i for i in tf.local_variables() if i.name.split('/')[0] == scope]
        self.sess.run(tf.initialize_variables([conf]))
        self.sess.run(tf.initialize_variables(stream_vars))

        batch_disc_labels = np.ones([self.config.batch_size, 1]) if is_real else np.zeros(
            [self.config.batch_size, 1])

        for idx in xrange(0, batch_idxs):
            batch_labels = data_y[idx * self.config.batch_size:(idx + 1) * self.config.batch_size]
            batch_files = data[idx * self.config.batch_size:(idx + 1) * self.config.batch_size]
            batch = [
                get_image(batch_file,
                          input_height=self.config.input_height,
                          input_width=self.config.input_width,
                          resize_height=self.config.output_height,
                          resize_width=self.config.output_width,
                          is_crop=self.config.is_crop,
                          is_grayscale=self.is_grayscale) for batch_file in batch_files]
            batch_images = np.array(batch).astype(np.float32)

            a, b, c = self.sess.run([op, self.prediction, self.prediction_tmp], feed_dict={self.model.inputs: batch_images, self.model.y: batch_labels,
                                         self.disc_label: batch_disc_labels})
            t = 2

        print 'Confusion matrix for: ' + scope
        print conf.eval()

    def _get_generated_data(self, sample_size, label):
        num_batches = sample_size // self.config.batch_size
        result = []
        for idx in xrange(num_batches):
            z_sample = np.random.uniform(-1, 1, size=[int(self.config.batch_size), self.model.z_dim])
            y_one_hot = np.zeros((self.config.batch_size, 1))
            y_one_hot[::] = label
            samples = self.sess.run(self.model.sampler, feed_dict={self.model.z: z_sample, self.model.y: y_one_hot})
            for j, sample in enumerate(samples):
                suffix = "xaugment_%s_%s.png" % (idx, j)
                path = os.path.join(self.cache_dir, suffix)
                scipy.misc.imsave(path, sample)
                result.append(path)

        return np.array(result)

    def _get_streaming_metrics(self, scope, prediction, label, num_classes):
        with tf.variable_scope(scope):
            # the streaming accuracy (lookup and update tensors)
            accuracy, accuracy_update = tf.metrics.accuracy(label, prediction,
                                                            name='accuracy')
            # Compute a per-batch confusion
            batch_confusion = tf.confusion_matrix(label, prediction,
                                                  num_classes=num_classes,
                                                  name='batch_confusion')
            # Create an accumulator variable to hold the counts
            confusion = tf.get_variable(name='confusion', initializer=tf.zeros([num_classes, num_classes],
                                                                               dtype=tf.int32))
            # Create the update op for doing a "+=" accumulation on the batch
            confusion_update = confusion.assign(confusion + batch_confusion)
            # Cast counts to float so tf.summary.image renormalizes to [0,255]
            confusion_image = tf.reshape(tf.cast(confusion, tf.float32),
                                         [1, num_classes, num_classes, 1])
            # Combine streaming accuracy and confusion matrix updates in one op
            test_op = tf.group(accuracy_update, confusion_update)

            tf.summary.image('confusion', confusion_image)
            tf.summary.scalar('accuracy', accuracy)

        return test_op, accuracy, confusion
