# FFNN.py
#
# Licensing Information:  You are free to use or extend this project for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to The Ohio State University, including a link to http://aritter.github.io/courses/5525_fall19.html
#
# Attribution Information: This assignment was developed at The Ohio State University
# by Alan Ritter (ritter.1492@osu.edu).

import sys
import numpy as np
import random

from ProcessData import ProcessData
from word2vec import load_word2vec_twitter, train_word2vec_twitter

import torch
import torch.nn as nn
import torch.optim as optim


# CNN Parameters
LEARNING_RATE = 0.001
NUM_EPOCHS = 2
NONLINEARITY = nn.ReLU()
NUM_FILTERS = 300
NUM_CLASSES = 2

# Word2vec parameters
DIM_EMB = 100   # word2vec
MIN_COUNT = 2
WINDOW = 5
SKIPGRAM = 1


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # (self.VOCAB_SIZE, self.DIM_EMB, self.NUM_CLASSES) = (VOCAB_SIZE, DIM_EMB, NUM_CLASSES)
        # embedding layer (don't need because we're using Word2Vec)
        # self.emb = nn.Embedding(self.VOCAB_SIZE, self.DIM_EMB)
        # unigram
        self.conv_1 = nn.Conv1d(in_channels=DIM_EMB, out_channels=NUM_FILTERS, kernel_size=1, padding=0)
        # bigram
        self.conv_2 = nn.Conv1d(in_channels=DIM_EMB, out_channels=NUM_FILTERS, kernel_size=2, padding=0)
        # trigram
        self.conv_3 = nn.Conv1d(in_channels=DIM_EMB, out_channels=NUM_FILTERS, kernel_size=3, padding=0)
        self.maxpool = nn.AdaptiveMaxPool1d(output_size=1)
        # ReLU nonlinearity layer
        self.nonlin = NONLINEARITY
        # hidden linear layer
        self.fc = nn.Linear(NUM_FILTERS * 3, 2)
        # softmax layer
        self.sm = nn.LogSoftmax()

    def forward(self, X, train=False):
        # emb = self.emb(X)
        emb = X
        # make the tensor 3D for conv layers
        new_emb = torch.unsqueeze(emb.transpose(0, 1), 0)
        # print(new_emb.size())
        # sum = torch.sum(emb, 0) this doesn't make sense for n-gram filters
        # convolve word embeddings
        conv1 = self.conv_1(new_emb)
        mp1 = self.maxpool(conv1)
        conv2 = self.conv_2(new_emb)
        mp2 = self.maxpool(conv2)
        conv3 = self.conv_3(new_emb)
        mp3 = self.maxpool(conv3)
        pooled = torch.cat([mp1, mp2, mp3], dim=1)
        # make 2D for linear layer
        pooled = pooled.view(NUM_FILTERS * 3, 1).transpose(0, 1)
        nonlin = self.nonlin(pooled)
        fc = self.fc(nonlin)
        output = self.sm(fc)
        output = output.view(2)

        return output


def prepare_embeddings(X_str, Y, w2v_model=None):
    if w2v_model is None:
        w2v_model = load_word2vec_twitter()
    X_emb = []
    Y_emb = []
    count = 0
    for i, tweet in enumerate(X_str):
        tweet_emb = []
        for word in tweet.split():
            try:
                tweet_emb.append(w2v_model.wv[word])
            except KeyError:
                pass
        # include the tweet if it has >3 words with embeddings available
        if len(tweet_emb) > 3:
            tweet_emb = torch.tensor(tweet_emb)
            X_emb.append(tweet_emb)
            Y_emb.append(Y[i])

    X_emb, Y_emb = balance_data(X_emb, Y_emb)
    return X_emb, Y_emb


def shuffle_indices(X, Y):
    # shuffle data, maintaining x/y correspondence
    indices = [i for i in range(len(X))]
    random.shuffle(indices)
    return [X[i] for i in indices], [Y[i] for i in indices]


def balance_data(X, Y):
    neg = [i for i in range(len(Y)) if Y[i] == -1]     # indices of every negative sample
    pos = [i for i in range(len(Y)) if Y[i] == 1]      # indices of every positive sample
    random.shuffle(neg)
    random.shuffle(pos)

    if len(neg) > len(pos):
        neg = neg[:len(pos)]
    else:
        pos = pos[:len(neg)]

    total = neg + pos
    random.shuffle(total)

    return [X[i] for i in total], [Y[i] for i in total]


def Eval_FFNN(X, Y, mlp):
    num_correct = 0
    for i in range(len(X)):
        logProbs = mlp.forward(X[i], train=False)
        pred = torch.argmax(logProbs)
        if pred == Y[i]:
            num_correct += 1
    print("Accuracy: %s" % (float(num_correct) / float(len(X))))


def Train_CNN(X, Y, n_iter):
    # X is XwordList, list of tensors of length 7222. Each tensor is a different length
    print("Start Training!")
    cnn_1 = CNN()
    optimizer = optim.Adam(cnn_1.parameters(), lr=LEARNING_RATE)

    for epoch in range(n_iter):
        total_loss = 0.0
        for i in range(len(X)):
            # X_row = torch.from_numpy(X[i]).float()
            y_onehot = torch.zeros(NUM_CLASSES)
            y_onehot[int(Y[i])] = 1

            cnn_1.zero_grad()
            probs = cnn_1.forward(X[i])
            loss = torch.neg(probs).dot(y_onehot)
            total_loss += loss

            loss.backward()
            optimizer.step()

            if i % 1000 == 0:
                print('Example {}'.format(i))

        print(f"loss on epoch {epoch} = {total_loss}")
    return cnn_1


def main():
    train, test = ProcessData('data/verified_users_tweets.json', 'data/non_verified_users_tweets.json')
    X_train_str, Y_train, X_test_str, Y_test = [i[0] for i in train], [i[1] for i in train], \
                                               [i[0] for i in test], [i[1] for i in test]
    if sys.argv[1] == 'twv':
        w2v_model = train_word2vec_twitter(dim_emb=DIM_EMB,
                                           min_count=MIN_COUNT,
                                           window=WINDOW,
                                           skipgram=SKIPGRAM
                                           )
    else:
        w2v_model = None

    X_train_emb, Y_train = prepare_embeddings(X_train_str, Y_train, w2v_model=w2v_model)
    X_test_emb, Y_test = prepare_embeddings(X_test_str, Y_test, w2v_model=w2v_model)

    cnn = Train_CNN(X_train_emb, (np.array(Y_train) + 1.0) / 2.0, NUM_EPOCHS)
    Eval_FFNN(X_test_emb, (np.array(Y_test) + 1.0) / 2.0, cnn)

    # (test.Y + 1.0) / 2.0


