# suppose a sample x = (x0,x1,...,xn)
# there is m class c0,c1,...,cm
# for unknown sample x_k, we compute p(c|x_k), where c=(c0,c1,...,cm)
# and c_pred = argmax_c{p(c|x_k}
# the bayes thm.
#     p(c_j|x_k)*p(x_k) = p(c_j, x_k)
#  =  p(x_k|c_j)*p(c_j)
# for 0<=j<=m, p(x_k) remains the same and thus can be omitted;
# p(c_j) is the prior and can be computed on the training data
# As to p(x_k|c_j):
#   according to the 'naive' nature, i.e., attributes are independent on each other
#     p(x_k|c_j) = p(x0_k,x1_k,...,xn_k|c_j)
#  =  p(x0_k|c_j)*p(x1_k|c_j)*...*p(xn_k|c_j)
#   for each attributes' conditional proba,
#   (1) attributes are discrete values, the proba is calculated as
#       count of the value divided by total count of all possible values
#   (2) attributes are continuous values, obtain a model for the
#       attribute's distribution estimated as a gaussian.
#   all of the above computation is done by the training data
import numpy as np
import logging


class NaiveBayes(object):
    def __init__(self, continuous_attr, logger='logger.txt'):
        logging.basicConfig(filename=logger, filemode='w',
                            level=logging.DEBUG, format='%(message)s')  # debug < info < warning < error < critical
        self.continuous = continuous_attr
        self.samples = None
        self.labels = None
        self.cls = []
        self.n_samples = None
        self.n_attribute = None
        self.n_classes = None
        self.class_priors = None
        self.c_x_priors = None  # n x m, each element is a dict, denoting for each attribute, the distribution of its values

    def _read_data(self, file, train=True):
        # cls = []
        self.samples = []
        self.labels = []
        with open(file) as f:
            for line in f.readlines():
                sample = [x.strip().strip('.') for x in line.split(',')]
                sample, label = sample[:-1], sample[-1]

                if '?' in sample or len(sample) < 2:
                    continue
                # only convert label to numbers during training
                if train:
                    if label in self.cls:
                        label = self.cls.index(label)
                    else:
                        tmp = label
                        label = len(self.cls)
                        self.cls.append(tmp)

                self.samples.append(sample)
                self.labels.append(label)

        self.samples = np.array(self.samples)
        self.labels = np.array(self.labels)
        self.n_classes = len(self.cls)
        self.n_samples = len(self.samples)
        self.n_attribute = len(self.samples[0])


    def _read_excel_data(self, file, train=True):
        samples_f, labels_f = file

        from pandas import read_excel
        # process samples
        samples = read_excel(samples_f, header=None)
        # process labels
        labels = read_excel(labels_f, header=None) if labels_f is not None else None


        self.samples = samples.as_matrix()
        # (len,) 1d array
        tmp = np.squeeze(labels.as_matrix(), axis=-1) if labels is not None else None # 1,2,3
        if tmp is not None:
            self.cls = []
            for l in tmp:
                if not l in self.cls:
                    self.cls.append(l)
                else:
                    continue

            self.labels = np.array([self.cls.index(x) for x in tmp])
        else:
            self.labels = None

        if train: # only update during training
            self.n_classes = len(self.cls) # 0,1,2
            self.n_samples = len(self.samples)
            self.n_attribute = len(self.samples[0])

    def _compute_class_prior(self):
        self.class_priors = np.array([0] * self.n_classes, dtype=np.float32)
        for sample, label in zip(self.samples, self.labels):
            self.class_priors[label] += 1
        self.class_priors /= self.n_samples

    def _compute_c_x_prior(self):

        if self.continuous == 'all':
            self.continuous = []
            for i in range(self.n_attribute):
                all_cls_stds = [np.std(self.samples[c == self.labels, i].astype(np.float32)) for c in range(len(self.cls))]
                if not 0. in all_cls_stds:
                    self.continuous.append(i)


        tmp = [None] * (self.n_attribute * self.n_classes)
        self.c_x_priors = np.reshape(np.array(tmp), newshape=(self.n_classes, self.n_attribute))
        # m x n x a
        # sample : n x 1 可用花式索引 [label, list(range(n_attribute)), sample]
        # self.x_c_priors = np.array([0]*(self.n_classes*self.n_attribute*self.attr_max))
        for sample, label in zip(self.samples, self.labels):
            for i in range(self.n_attribute):

                if not i in self.continuous:
                    entry = self.c_x_priors[label, i]
                    if entry is None:
                        entry = dict()
                    if entry.get(sample[i], None) is None:
                        entry[sample[i]] = 1
                    else:
                        entry[sample[i]] += 1
                    self.c_x_priors[label, i] = entry

        for i in self.continuous:  # should make sure it's numerical value
            # avg = np.mean(self.samples[,i])
            # stddev = np.std(self.samples[:,i])
            self.c_x_priors[:, i] = [(np.mean(self.samples[c == self.labels, i].astype(np.float32)),
                                      np.std(self.samples[c == self.labels, i].astype(np.float32)))
                                     for c in range(len(self.cls))]
        # for attributes in self.c_x_priors:
        #     # attributes is a vector
        #     denoms = [np.sum(list(attr.values())) for attr in attributes] # attr is dict

    def _gaussian(self, x, c, i):
        epsilon = 1e-4
        yita, sigma = self.c_x_priors[c, i]
        sigma += epsilon
        # tmp = (np.sqrt(2 * np.pi) * sigma)
        # if tmp == 0:
        #     print(sigma)
        fraction = 1. / (np.sqrt(2 * np.pi) * sigma)
        exp = np.exp(-(x.astype(np.float32) - yita) ** 2 / (2 * sigma ** 2))
        return fraction * exp

    def _predict(self, sample):
        # p(x_k|c_j)
        proba = [1] * self.n_classes

        for i, p in enumerate(proba):
            # for class i, compute chaining proba for each attr
            for idx, attr in enumerate(sample):
                if not idx in self.continuous:  # discrete values
                    entry = self.c_x_priors[i, idx]  # dict
                    total = self.n_samples * self.class_priors[i]
                    if entry.get(attr) is None:
                        p *= 1. / (total + len(list(entry.values())) + 1)
                        # entry[attr] = 1
                        # self.c_x_priors[i, idx] = entry
                    else:
                        p *= np.float32(entry[attr]) / (total)
                else:
                    p *= self._gaussian(attr, i, idx)  # the gaussian density for class i attr idx

            proba[i] = p

        pred = np.argmax(proba)
        pred = self.cls[pred]

        return proba, pred

    def train(self, file='adult.data'):
        if isinstance(file, str):
            self._read_data(file=file)
        else:
            self._read_excel_data(file=file)
        self._compute_class_prior()
        self._compute_c_x_prior()
        logging.info('priors have been extracted from {} training samples'.format(self.n_samples))

    def test(self, file='adult.test'):
        if isinstance(file, str):
            self._read_data(file=file, train=False)
        else:
            self._read_excel_data(file=file, train=False)

        # the variables are shared with testing
        accuracy = 0

        tmp =  [self._predict(sample) for sample in self.samples]
        probas, predicts = zip(*tmp) # list
        logging.info('there are {} testing samples'.format(self.n_samples))

        if self.labels is not None:
            # to avoid predicts converting to multidimensional array
            # the return value of each predict is a tuple
            correctness = self.labels == np.array(predicts)
            accuracy = np.sum(correctness)
            accuracy /= self.n_samples
            for idx, is_correct in enumerate(correctness):
                logging.info('-------------{}-------------'.format(idx))
                if is_correct:
                    logging.info('sample={} is predicted {}'.format(self.samples[idx], predicts[idx]))
                else:
                    logging.warning('\n=====================================\n'
                                    + 'sample={} is predicted {}\nwrong answer, should be {}. the estimated proba is {}\n'.format(
                        self.samples[idx], predicts[idx], self.labels[idx], probas[idx])
                                    + '=====================================\n')
            logging.info('\nthe overall accuracy is {}'.format(accuracy))
        else:
            for idx, predicted in enumerate(predicts):
                logging.info('-------------{}-------------'.format(idx))
                logging.info('sample={} is predicted {} with estimated proba {}'.format(self.samples[idx], predicted, probas[idx]))

def main():
    logger = 'logger.txt'
    newdataset = True
    train_files = (r'../uspst_uni_train.xls', r'../uspst_gnd_train.xls')#(r'../balance_uni_train.xls', r'../balance_gnd_train.xls')
    test_files = (r'../uspst_uni_test.xls', None)

    cont = 'all' if newdataset else [0,2,4,10,11,12]
    trainf = train_files if newdataset else 'adult.data'
    testf = test_files if newdataset else 'adult.test'

    nb = NaiveBayes(continuous_attr=cont, logger=logger)
    nb.train(file=trainf)
    nb.test(file=testf)


if __name__ == '__main__':
    main()
