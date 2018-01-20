import os

import numpy as np
import scipy
from scipy import misc, ndimage
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix, accuracy_score, f1_score


class BaseClassifier(object):
    """
    Common methods for classifiers to benchmark GAN sample performance for imabalanced datasets
    """

    def build_result(self, acc, auc, f1, bacc, avgp, setting):
        return {'classifier': 'vgg_pretrained', 'dataset': setting, 'acc': acc, 'auc': auc, 'f1': f1, 'bacc': bacc,
                'avgp': avgp, 'global_step': self.global_step}

    def generate_dataset(self, sample_size, config, ooc):
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
                    path = os.path.join(config.cache_dir, suffix)
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
                    path = os.path.join(config.cache_dir, suffix)
                    scipy.misc.imsave(path, sample)
                    result.append(path)
            else:
                result.append(samples)

        X_augmented = np.array(result) if ooc else np.concatenate(result)
        y_augmented = np.ones((config.batch_size * sample_size,))

        return X_augmented, y_augmented

    def oversample(self, X_train, y_train, config, noisy=True):
        selected_indices = np.where(y_train == 1)
        sample_size = y_train.shape[0] - 2 * selected_indices[0].shape[0]
        oversampled_indices = np.random.choice(selected_indices[0], sample_size)
        if noisy:
            X_oversampled = np.concatenate((X_train, self.noisify(X_train[oversampled_indices], config)))
        else:
            X_oversampled = np.concatenate((X_train, X_train[oversampled_indices]))
        y_oversampled = np.concatenate((y_train, y_train[oversampled_indices]))
        p = np.random.permutation(y_oversampled.shape[0])
        return X_oversampled[p], y_oversampled[p]

    def noisify(self, X_train, config):
        result = []
        for i, f in enumerate(X_train):
            x = misc.imread(f)
            flipped = np.fliplr(x)
            noisy = ndimage.gaussian_filter1d(flipped, sigma=3, axis=0)
            noisy = ndimage.gaussian_filter1d(noisy, sigma=3, axis=1)
            suffix = "xnoisy_%s.png" % (i,)
            path = os.path.join(config.cache_dir, suffix)
            scipy.misc.imsave(path, noisy)
            result.append(path)
        return np.array(result)

    def get_metrics(self, y_true, y_score, y_valid, y_valid_score):
        auc = roc_auc_score(y_true, y_score)
        avgp = average_precision_score(y_true, y_score)
        threshold_acc, threshold_bacc, threshold_f1 = self.estimate_best_thresholds(y_valid, y_valid_score)

        y_pred = self.preds_from_score(y_score, threshold_acc)
        acc = accuracy_score(y_true, y_pred)

        y_pred = self.preds_from_score(y_score, threshold_bacc)
        bacc = self.balanced_accuracy(y_true, y_pred)

        y_pred = self.preds_from_score(y_score, threshold_f1)
        f1 = f1_score(y_true, y_pred)

        print 'ACC:' + str(acc)
        print 'AUC: ' + str(auc)
        print 'F1:' + str(f1)
        print 'BACC:' + str(bacc)
        print 'AVGP:' + str(avgp)
        return acc, auc, f1, bacc, avgp

    def get_all_thresholds(self, y_true, y_score):
        """
        https://stackoverflow.com/questions/31488517/getting-the-maximum-accuracy-for-a-binary-probabilistic-classifier-in-scikit-lea
        """
        y_true = np.asarray(y_true, dtype=np.bool_)
        y_score = np.asarray(y_score, dtype=np.float_)
        assert (y_score.size == y_true.size)

        order = np.argsort(y_score)  # Just ordering stuffs
        y_true = y_true[order]
        # The thresholds to consider are just the values of score, and 0 (accept everything)
        thresholds = np.insert(y_score[order], 0, 0)
        TP = [sum(
            y_true)]  # Number of True Positives (For Threshold = 0 => We accept everything => TP[0] = # of postive in true y)
        FP = [sum(
            ~y_true)]  # Number of True Positives (For Threshold = 0 => We accept everything => TP[0] = # of postive in true y)
        TN = [0]  # Number of True Negatives (For Threshold = 0 => We accept everything => we don't have negatives !)
        FN = [0]  # Number of True Negatives (For Threshold = 0 => We accept everything => we don't have negatives !)

        for i in range(1, thresholds.size):  # "-1" because the last threshold
            # At this step, we stop predicting y_score[i-1] as True, but as False.... what y_true value say about it ?
            # if y_true was True, that step was a mistake !
            TP.append(TP[-1] - int(y_true[i - 1]))
            FN.append(FN[-1] + int(y_true[i - 1]))
            # if y_true was False, that step was good !
            FP.append(FP[-1] - int(~y_true[i - 1]))
            TN.append(TN[-1] + int(~y_true[i - 1]))

        TP = np.asarray(np.squeeze(TP), dtype=np.float32)
        FP = np.asarray(np.squeeze(FP), dtype=np.float32)
        TN = np.asarray(TN, dtype=np.float32)
        FN = np.asarray(FN, dtype=np.float32)

        return thresholds, TP, FP, TN, FN

    def get_all_metrics(self, y_true, y_score):
        thresholds, TP, FP, TN, FN = self.get_all_thresholds(y_true, y_score)
        accuracy = (TP + TN) / (TP + FP + TN + FN)
        balanced_accuracy = (TP/(TP + FN) + TN/(TN + FP)) / 2
        f1_score = 2 * TP / (2 * TP + FP + FN)

        return thresholds, accuracy, balanced_accuracy, f1_score

    def get_best_metrics(self, y_true, y_score):
        thresholds, accuracy, balanced_accuracy, f1_score = self.get_all_metrics(y_true, y_score)
        return max(accuracy), max(balanced_accuracy), max(f1_score)

    def estimate_best_thresholds(self, y_true, y_score):
        thresholds, accuracy, balanced_accuracy, f1_score = self.get_all_metrics(y_true, y_score)

        max_acc_idx = np.argmax(accuracy)
        max_bacc_idx = np.argmax(balanced_accuracy)
        max_f1_idx = np.argmax(f1_score)

        return thresholds[max_acc_idx], thresholds[max_bacc_idx], thresholds[max_f1_idx]

    def preds_from_score(self, y_score, threshold):
        y_pred = np.copy(y_score)
        y_pred[y_pred >= threshold] = 1
        y_pred[y_pred < threshold] = 0
        return y_pred


    def balanced_accuracy(self, y_true, y_pred):
        conf_mat = confusion_matrix(y_true, y_pred)
        return (float(conf_mat[0][0]) / sum(conf_mat[0]) + float(conf_mat[1][1]) / sum(conf_mat[1])) / 2.0
