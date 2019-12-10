# Perceptron.py
# --------------
# Licensing Information:  You are free to use or extend this project for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to The Ohio State University, including a link to http://aritter.github.io/courses/5525_spring19.html
#
# Attribution Information: This assignment was developed at The Ohio State University
# by Alan Ritter (ritter.1492@osu.edu).

import sys

import numpy as np
from Eval import Eval
from scipy.sparse import csc_matrix

from imdb import IMDBdata


class Perceptron:
    def __init__(self, X, Y, N_ITERATIONS):
        self.N_ITERATIONS = N_ITERATIONS
        # initialize one weight for every feature (word) and a bias term
        self.WEIGHTS = np.zeros((1, X.shape[1]), dtype='int32')
        self.BIAS = 0

        # NOTE: self.Train will train the perceptron without averaging; comment out line 29 and uncomment
        # line 30 to train with parameter averaging
        # self.Train(X, Y)
        self.TrainWithAveraging(X, Y)

    # NOTE: I implemented parameter averaging as a second training function called TrainWithAveraging, not in a
    # separate function ComputeAverageParameters.

    def Train(self, X, Y):
        for i in range(self.N_ITERATIONS):
            for sample_idx in range(X.shape[0]):     # for each sample
                sample = X.getrow(sample_idx).toarray()
                activation = np.dot(self.WEIGHTS, np.transpose(sample)) + self.BIAS
                if Y[sample_idx] * activation <= 0:
                    self.WEIGHTS = np.add(self.WEIGHTS, sample * Y[sample_idx])
                    self.BIAS = self.BIAS + Y[sample_idx]

        return

    def TrainWithAveraging(self, X, Y):
        avg_weights = np.zeros((1, X.shape[1]))
        avg_bias = 0
        avg_count = 1
        for i in range(self.N_ITERATIONS):
            for sample_idx in range(X.shape[0]):     # for each sample
                sample = X.getrow(sample_idx).toarray()
                activation = np.dot(self.WEIGHTS, np.transpose(sample)) + self.BIAS
                if Y[sample_idx] * activation <= 0:
                    self.WEIGHTS = np.add(self.WEIGHTS, sample * Y[sample_idx])
                    avg_weights = np.add(self.WEIGHTS, avg_count * sample * Y[sample_idx])
                    self.BIAS = self.BIAS + Y[sample_idx]
                    avg_bias = avg_bias + avg_count * Y[sample_idx]
                avg_count += 1

        self.WEIGHTS = np.subtract(self.WEIGHTS, avg_weights / avg_count)
        self.BIAS = self.BIAS - avg_bias / avg_count

        return

    def printWords(self, vocab):
        # get vocab dict
        id2word = vocab.id2word
        # create an array of indices such that the value at 0 is the index in weights with the smallest value
        # and the value at len-1 is the index in weights with the largest value
        word_ids_sorted = np.argsort(self.WEIGHTS)[0]
        # collect top and bottom 20 words
        pos_words = {}
        neg_words = {}
        for i in range(20):
            neg_word = id2word[word_ids_sorted[i]]
            neg_weight = self.WEIGHTS[0, word_ids_sorted[i]]
            pos_word = id2word[word_ids_sorted[len(word_ids_sorted) - i - 1]]
            pos_weight = self.WEIGHTS[0, word_ids_sorted[len(word_ids_sorted) - i - 1]]
            pos_words.update({pos_word: pos_weight})
            neg_words.update({neg_word: neg_weight})

        print('\nPositive words:')
        for item in pos_words.items():
            print("{} {}".format(item[0], item[1]))
        print('\nNegative words:')
        for item in neg_words.items():
            print("{} {}".format(item[0], item[1]))

    def Predict(self, X):
        y_pred = []
        for i in range(X.shape[0]):
            sample = X.getrow(i).toarray()
            activation = np.dot(self.WEIGHTS, np.transpose(sample)) + self.BIAS
            # print(activation[0][0])
            y_pred.append(get_sign(activation[0][0]))
        return y_pred

    def Eval(self, X_test, Y_test):
        Y_pred = self.Predict(X_test)
        ev = Eval(Y_pred, Y_test)
        return ev.Accuracy()


def get_sign(num):
    if num >= 0:
        return 1
    else:
        return -1


if __name__ == "__main__":
    train = IMDBdata("%s/train" % sys.argv[1])
    test = IMDBdata("%s/test" % sys.argv[1], vocab=train.vocab)
    
    ptron = Perceptron(train.X, train.Y, int(sys.argv[2]))
    # ptron.ComputeAverageParameters()
    print(ptron.Eval(test.X, test.Y))

    ptron.printWords(train.vocab)

