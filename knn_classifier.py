from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

from nideep.datasets.amfed.amfed import AMFED
from sklearn import preprocessing
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

class KNNClassifier(object):
    def __init__(self, sess, gan, cache_dir='/mnt/raid/data/ni/dnn/rparra/cache/', input_height=64, input_width=64,
                 y_dim=1, c_dim=3, batch_size=64):
        self.sess = sess
        self.input_height = input_height
        self.input_width = input_width
        self.y_dim = 1
        self.c_dim = c_dim
        self.cache_dir = cache_dir
        self.gan = gan
        self.num_classes = 2
        self.batch_size = batch_size
        self.AUGEMENTED = 'xaugment.dat'

    def get_dataset(self):
        dataset = AMFED(dir_prefix='/mnt/raid/data/ni/dnn/AMFED/', video_type=AMFED.VIDEO_TYPE_AVI,
                        cache_dir=self.cache_dir)

        (X, _, videos_train, _, _, _, _, _) = dataset.as_numpy_array(train_proportion=0.8)

        le = preprocessing.LabelEncoder()
        le.fit(videos_train)
        X_train, X_test, y_train, y_test = train_test_split(X, le.transform(videos_train), test_size = 0.2)
        X_train = (X_train.astype(np.float32) - 127.5) / 127.5
        X_test = (X_test.astype(np.float32) - 127.5) / 127.5
        X_train = X_train.reshape(-1, self.input_width * self.input_height * self.c_dim)
        X_test = X_test.reshape(-1, self.input_width * self.input_height * self.c_dim)

        return X_train, X_test, y_train, y_test, le

    def evaluate(self, config, teardown=False):
        X_train, X_test, y_train, y_test, label_encoder = self.get_dataset()
        clf = self.get_classifier()
        clf.fit(X_train, y_train)
        explained_variance = clf.named_steps['pca'].explained_variance_ratio_
        acc = clf.score(X_test, y_test)
        y_pred = clf.predict(X_test)
        print 'Sanity check for our frame to vid classifier'
        print 'Variance explained: %s' % str(np.sum(explained_variance))
        print 'Mean accuracy: %s' % str(acc)

        # We retrain the classifier on the entire dataset
        clf = self.get_classifier()
        y = np.concatenate([y_train, y_test])
        clf.fit(np.concatenate([X_train, X_test]), y)
        X_gen = self.get_generated_samples(config, video_number=label_encoder.classes_.shape[0])
        y_gen_pred = clf.predict(X_gen)

        np.save('videos_train.npy', label_encoder.inverse_transform(y))
        np.save('videos_test.npy', label_encoder.inverse_transform(y_test))
        np.save('videos_gen.npy', label_encoder.inverse_transform(y_gen_pred))
        np.save('frames_test.npy', X_test)
        np.save('frames_gen.npy', X_gen)

    def get_classifier(self):
        knn = KNeighborsClassifier(n_neighbors=5)
        pca = PCA(n_components=100)
        clf = Pipeline(steps=[('pca', pca), ('knn', knn)])
        return clf

    def get_generated_samples(self, config, video_number, sample_size=1833):
        y_one_hot = np.ones((config.batch_size, 1))
        y_video_label = np.random.choice(video_number, (config.batch_size, 1)) / float(video_number)
        y_sample = np.concatenate([y_one_hot, y_video_label], axis=1)
        result = []
        for i in range(sample_size):
            z_sample = np.random.uniform(-1, 1, size=[int(config.batch_size), self.gan.z_dim])
            samples = self.sess.run(self.gan.sampler, feed_dict={self.gan.z: z_sample, self.gan.y: y_sample})
            result.append(samples)

        return np.concatenate(result).reshape(-1, self.input_width * self.input_height * self.c_dim)
