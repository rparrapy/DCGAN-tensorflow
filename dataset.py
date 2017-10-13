from utils import get_image
import numpy as np


class DataSet(object):
    def __init__(self, images, labels, config):
        '''
        Create a dataset for batch-based training

        :param images: if ooc an array of images, otherwise an array of image paths.
        :param labels: array of labels
        :param ooc: out-of-core flag, wether images are in memory or not.
        '''
        self.ooc = config.dataset == 'celeba'
        self.is_grayscale = (config.c_dim == 1)
        self._num_examples = len(images) if self.ooc else images.shape[0]
        self.config = config
        #images = (images.astype(np.float32) - 127.5) / 127.5
        #images = np.multiply(images, 1.0 / 255.0)

        if self.ooc:
            self._images = None
            self._labels = labels
            self._image_paths = images
        else:
            self._images = images
            self._labels = labels
            self._image_paths = None

        self._epochs_completed = 0
        self._index_in_epoch = 0

    @property
    def images(self):
        return self._images

    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size):
        """Return the next `batch_size` examples from this data set."""
        start = self._index_in_epoch
        self._index_in_epoch += batch_size

        if self._index_in_epoch > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1

            # # Shuffle the data (maybe)
            # perm = np.arange(self._num_examples)
            # np.random.shuffle(perm)
            # self._images = self._images[perm]
            # self._labels = self._labels[perm]
            # Start next epoch

            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
        end = self._index_in_epoch

        if self.ooc:
            batch_files = self._image_paths[start:end]

            images_batch = np.array([
                get_image(batch_file,
                          input_height=self.config.input_height,
                          input_width=self.config.input_width,
                          resize_height=self.config.output_height,
                          resize_width=self.config.output_width,
                          is_crop=self.config.is_crop,
                          is_grayscale=self.is_grayscale) for batch_file in batch_files])
        else:
            images_batch = self._images[start:end]

        labels_batch = self._labels[start:end]

        return images_batch, labels_batch
