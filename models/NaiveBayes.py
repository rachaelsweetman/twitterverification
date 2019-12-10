# NaiveBayes.py
# --------------
# Licensing Information:  You are free to use or extend this project for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to The Ohio State University, including a link to http://aritter.github.io/courses/5525_spring19.html
#
# Attribution Information: This assignment was developed at The Ohio State University
# by Alan Ritter (ritter.1492@osu.edu).

import sys
import math
import numpy as np
from Eval import Eval
from imdb import IMDBdata


class NaiveBayes:
    def __init__(self, X, Y, ALPHA=1.0):
        self.ALPHA = ALPHA
        # initialize parameters
        self.num_samples = Y.shape[0]
        self.pos_prior = 0
        self.neg_prior = 0
        self.pos_posteriors = [self.ALPHA] * X.shape[1]
        self.neg_posteriors = [self.ALPHA] * X.shape[1]
        self.Train(X, Y)

    def Train(self, X, Y):
        total_pos = len([i for i in Y if i == 1])
        total_neg = len([i for i in Y if i != 1])

        # calculate prior probabilities
        self.pos_prior = total_pos / self.num_samples
        self.neg_prior = total_neg / self.num_samples

        # calculate posterior probabilities
        # for every feature, we need the probability of that feature occurring, given that the sample is pos/neg
        for sample_index in range(X.shape[0]):    # for every sample
            # obtain the indices of all features that are positive for that sample
            positive_indices = X.getrow(sample_index).nonzero()[1]
            for index in positive_indices:      # for every positive feature in the sample
                if Y[sample_index] == 1:
                    self.pos_posteriors[index] += 1
                else:
                    self.neg_posteriors[index] += 1

        self.pos_posteriors = [i / total_pos for i in self.pos_posteriors]
        self.neg_posteriors = [i / total_neg for i in self.neg_posteriors]

        return

    def Predict(self, X):
        Y_pred = []

        # for each sample
        for sample_index in range(X.shape[0]):
            sample = X.getrow(sample_index)
            pos_feature_indices = sample.nonzero()[1]

            # calculate summation term
            summation_term_positive = 0
            summation_term_negative = 0
            for index in pos_feature_indices:
                summation_term_positive += math.log(self.pos_posteriors[index])
                summation_term_negative += math.log(self.neg_posteriors[index])

            # calculate probability the sample is positive and negative, given its features
            pos_prob = math.log(self.pos_prior) + summation_term_positive
            neg_prob = math.log(self.neg_prior) + summation_term_negative

            print('pos {} neg {}'.format(pos_prob, neg_prob))

            # take the argmax of positive and negative probabilities
            if pos_prob > neg_prob:
                Y_pred.append(1)
            else:
                Y_pred.append(-1)

        return Y_pred

    def Eval(self, X_test, Y_test):
        Y_pred = self.Predict(X_test)
        ev = Eval(Y_pred, Y_test)
        return ev.Accuracy()


if __name__ == "__main__":
    train = IMDBdata("%s/train" % sys.argv[1])
    test = IMDBdata("%s/test" % sys.argv[1], vocab=train.vocab)

    print('Training model...')
    nb = NaiveBayes(train.X, train.Y, float(sys.argv[2]))
    print('Trained! Testing...')
    print(nb.Eval(test.X, test.Y))
    print('Done!')
