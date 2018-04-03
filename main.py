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
from scipy.stats import multivariate_normal as multi_gauss
import logging
import abc


class Bayes(object):
    def __init__(self, continuous_attr, logger='logger.txt', dataset='unknown'):
        np.seterr(divide='raise')
        logging.basicConfig(filename=logger, filemode='w',
                            level=logging.DEBUG, format='%(message)s')  # debug < info < warning < error < critical
        self.dataset_name = dataset
        self.continuous = continuous_attr
        self.samples = []
        self.raw_labels = []
        self.cls = []
        self.n_samples = None
        self.n_attribute = None
        self.n_classes = None
        self.class_priors = None
        self.class_pd_priors = None  # n x m, each element is a dict, denoting for each attribute, the distribution of its values

    @abc.abstractmethod
    def _read_data(self, file, train=True):
        """
        labels should be processed into raw indices
        """
        pass

    def _compute_class_prior(self):
        self.class_priors = np.array([0] * self.n_classes, dtype=np.float32)
        for sample, label in zip(self.samples, self.raw_labels):
            self.class_priors[label] += 1
        self.class_priors /= self.n_samples

    @abc.abstractmethod
    def _compute_class_pd_prior(self):  # proba density
        """
        freq for discrete
        stats for continuous
        :return:
        """
        pass

    @abc.abstractmethod
    def _compute_class_probas(self, sample):
        """

        :param sample:
        :return: a list of scalars
        """
        pass

    def _predict(self, sample):
        # p(x_k|c_j)

        proba = self._compute_class_probas(sample)
        pred = np.argmax(proba)
        # pred = self.cls[pred]

        return proba, pred

    def train(self, file):
        self._read_data(file)
        self._compute_class_prior()
        self._compute_class_pd_prior()
        logging.info('priors have been extracted from {} training samples'.format(self.n_samples))

    def _prepare_labels(self):
        pass

    def test(self, file):
        self._read_data(file=file, train=False)

        tmp = [self._predict(sample) for sample in self.samples]
        probas, raw_predicts = zip(*tmp)  # list
        mapped_predicts = [self.cls[p] for p in raw_predicts]
        logging.info('there are {} testing samples'.format(len(self.samples)))

        self._prepare_labels()
        # mappings = lambda x:1 if x==0 else (2 if x<0 else 3)
        # self.labels = np.array([mappings(a[0]*a[1]-a[2]*a[3])for a in self.samples])

        if self.raw_labels is not None:
            # to avoid predicts converting to multidimensional array
            # the return value of each predict is a tuple
            correctness = self.raw_labels == np.array(raw_predicts)
            accuracy = np.sum(correctness)
            accuracy /= len(self.samples)
            for idx, is_correct in enumerate(correctness):
                logging.info('-------------{}-------------'.format(idx))
                if is_correct:
                    logging.info('sample={} is predicted {}'.format(self.samples[idx], mapped_predicts[idx]))
                else:
                    logging.warning('\n=====================================\n'
                                    + 'sample={} is predicted {}\nwrong answer, should be {}. the estimated proba is {}\n'.format(
                        self.samples[idx], mapped_predicts[idx], self.cls[self.raw_labels[idx]], probas[idx])
                                    + '=====================================\n')
            logging.info('\nthe overall accuracy is {}'.format(accuracy))
        else:
            for idx, predicted in enumerate(mapped_predicts):
                logging.info('-------------{}-------------'.format(idx))
                logging.info('sample={} is predicted {} with estimated proba {}'.format(self.samples[idx], predicted,
                                                                                        probas[idx]))

        return mapped_predicts


class NaiveBayes(Bayes): # cannot make sure all float32

    def _read_data(self, file, train=True):
        # cls = []
        self.samples = []
        self.raw_labels = []
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
                    self.raw_labels.append(label)
                else:# test
                    self.raw_labels.append(self.cls.index(label))

                self.samples.append(sample)


        self.samples = np.array(self.samples)
        self.raw_labels = np.array(self.raw_labels, dtype=np.int)
        if train:
            self.n_classes = len(self.cls)
            self.n_samples = len(self.samples)
            self.n_attribute = len(self.samples[0])

    def _compute_class_pd_prior(self):

        if self.continuous == 'all':
            self.continuous = []
            for i in range(self.n_attribute):
                all_cls_stds = [np.std(self.samples[c == self.raw_labels, i].astype(np.float32)) for c in
                                range(len(self.cls))]
                if not 0. in all_cls_stds:
                    self.continuous.append(i)

        tmp = [None] * (self.n_attribute * self.n_classes)
        self.class_pd_priors = np.reshape(np.array(tmp), newshape=(self.n_classes, self.n_attribute))
        # m x n x a
        # sample : n x 1 可用花式索引 [label, list(range(n_attribute)), sample]
        # self.x_c_priors = np.array([0]*(self.n_classes*self.n_attribute*self.attr_max))
        for sample, label in zip(self.samples, self.raw_labels):
            for i in range(self.n_attribute):

                if not i in self.continuous:
                    entry = self.class_pd_priors[label, i]
                    if entry is None:
                        entry = dict()
                    if entry.get(sample[i], None) is None:
                        entry[sample[i]] = 1
                    else:
                        entry[sample[i]] += 1
                    self.class_pd_priors[label, i] = entry

        for i in self.continuous:  # should make sure it's numerical value
            self.class_pd_priors[:, i] = [(np.mean(self.samples[c == self.raw_labels, i].astype(np.float32)),
                                           np.std(self.samples[c == self.raw_labels, i].astype(np.float32)))
                                          for c in range(len(self.cls))]

    def _gaussian_log_density(self, x, c, i):
        epsilon = 0 # to avoid log0
        yita, sigma = self.class_pd_priors[c, i] # yita is the average, sigma is the std-dev
        sigma += epsilon

        fraction = -(0.5*np.log(2 * np.pi)+np.log(sigma))
        exp = -(x.astype(np.float32) - yita) ** 2 / (2 * sigma ** 2)

        return fraction + exp


    def _compute_class_probas(self, sample):
        probas = []

        for i in range(self.n_classes):
            p = 0
            # for class i, compute chaining proba for each attr
            for idx, attr in enumerate(sample):
                if not idx in self.continuous:  # discrete values
                    entry = self.class_pd_priors[i, idx]  # dict
                    total = self.n_samples * self.class_priors[i]
                    if entry.get(attr) is None:
                        p += -np.log(total + len(list(entry.values())) + 1)
                        # entry[attr] = 1
                        # self.c_x_priors[i, idx] = entry
                    else:
                        p += np.log(entry[attr]) - np.log(total)
                else:
                    p += self._gaussian_log_density(attr, i, idx)  # the gaussian density for class i attr idx

            probas.append(p)

        return probas

class Test_Bayes(Bayes):
    def _read_data(self, file, train=True):
        samples_f, labels_f = file

        from pandas import read_excel
        # process samples
        samples = read_excel(samples_f, header=None, dtype=np.float64)
        # process labels
        labels = read_excel(labels_f, header=None) if labels_f is not None else None

        self.samples = samples.as_matrix()
        # (len,) 1d array
        tmp = np.squeeze(labels.as_matrix(), axis=-1) if labels is not None else None  # 1,2,3
        if tmp is not None:
            # self.cls = []
            for l in tmp:
                if not l in self.cls:
                    self.cls.append(l)
                else:
                    continue

            self.raw_labels = np.array([self.cls.index(x) for x in tmp], dtype=np.int)
        else:
            self.raw_labels = None

        if train:  # only update during training
            self.n_classes = len(self.cls)  # 0,1,2
            self.n_samples = len(self.samples)
            self.n_attribute = len(self.samples[0])

    def test(self, file):
        mapped_predicts = super().test(file)
        mapped_predicts = np.expand_dims(np.array(mapped_predicts), axis=-1) # len x 1

        from pandas import DataFrame
        df = DataFrame(mapped_predicts)
        df.to_excel('result_{}.xls'.format(self.dataset_name), index=False, header=False)


class Test_NaiveBayes(Test_Bayes, NaiveBayes):
    pass

class multinomial_Bayes(Bayes):
    def _compute_class_pd_prior(self):
        self.class_pd_priors = []
        for c in range(self.n_classes):
            c_samples = self.samples[c == self.raw_labels, :].astype(np.float32)
            avg_vec = np.mean(c_samples, axis=0)
            c_samples = np.transpose(c_samples)
            cov_mat = np.cov(c_samples)
            self.class_pd_priors.append((avg_vec, cov_mat))

    def _multi_gaussian_log_density(self, x_vec, c):
        avg_vec, cov_mat = self.class_pd_priors[c]
        result = multi_gauss.logpdf(x_vec, avg_vec, cov_mat, allow_singular=True)
        if result>0:
            print('wrong')

        return result

    def _compute_class_probas(self, sample):

        # avg_vec, cov_mat = self.class_pd_priors[c]
        return [self._multi_gaussian_log_density(sample, c) for c in range(self.n_classes)]

class Test_multi_Bayes(multinomial_Bayes, Test_Bayes):
    # pass
    def _prepare_labels(self):
        mappings = lambda x:0 if x==0 else (1 if x<0 else 2)
        self.raw_labels = np.array([mappings(a[0]*a[1]-a[2]*a[3])for a in self.samples])

class ups_multi_Bayes(NaiveBayes, Test_Bayes):
    def _read_data(self, file, train=True):
        # init for self.samples and self.raw_inputs
        self.samples = []
        self.raw_labels = []
        # self.cls = []
        with open(file) as f:
            for line in f.readlines():
                fields = line.split(' ')
                # obtaining label
                label = fields[0]
                if train:
                    if label in self.cls:
                        label = self.cls.index(label)
                    else:
                        tmp = label
                        label = len(self.cls)
                        self.cls.append(tmp)
                    self.raw_labels.append(label)
                else:# test
                    self.raw_labels.append(self.cls.index(label))

                # obtaining attrs
                attrs = [eval(f.split(':')[-1]) for f in fields[1:] if f!='\n']
                self.samples.append(attrs)

        self.samples = np.array(self.samples, dtype=np.float64)
        self.raw_labels = np.array(self.raw_labels, dtype=np.int)
        if train:
            self.n_samples = len(self.samples)
            self.n_attribute = len(self.samples[0])
            self.n_classes = len(self.cls)

def main():
    logger = 'logger_{}.txt'
    dataset = 'uspst'
    train_files = {'balance':(r'../balance_uni_train.xls', r'../balance_gnd_train.xls'),
                   'adult':'adult.data',
                   'ups':'../usps',
                   'uspst':(r'../uspst_uni_train.xls', r'../uspst_gnd_train.xls')}
    test_files = {'balance':(r'../balance_uni_test.xls', None),
                  'adult':'adult.test',
                  'ups':'../usps.t',
                  'uspst':(r'../uspst_uni_test.xls', None)}
    cls = {'balance':Test_multi_Bayes,
           'adult':NaiveBayes,
           'ups':ups_multi_Bayes,
           'uspst':Test_NaiveBayes}

    cont_values = {'balance':None,
                   'adult': [0, 2, 4, 10, 11, 12],
                   'ups':'all',
                   'uspst':'all'}

    trainf = train_files[dataset]
    testf = test_files[dataset]

    Method = cls[dataset]
    nb = Method(continuous_attr=cont_values[dataset], logger=logger.format(dataset), dataset=dataset)
    nb.train(file=trainf)
    nb.test(file=testf)


if __name__ == '__main__':
    main()
