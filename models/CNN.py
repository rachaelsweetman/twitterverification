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
from Eval import Eval

import torch
import torch.nn as nn
import torch.optim as optim

from imdb import IMDBdata


# FFNN Parameters
LEARNING_RATE = 0.001
NUM_EPOCHS = 2
NONLINEARITY = nn.ReLU()
NUM_FILTERS = 300


class CNN(nn.Module):
    def __init__(self, X, Y, VOCAB_SIZE, DIM_EMB=10, NUM_CLASSES=2):
        super(CNN, self).__init__()
        (self.VOCAB_SIZE, self.DIM_EMB, self.NUM_CLASSES) = (VOCAB_SIZE, DIM_EMB, NUM_CLASSES)
        # embedding layer
        self.emb = nn.Embedding(self.VOCAB_SIZE, self.DIM_EMB)
        # unigram
        self.conv_1 = nn.Conv1d(in_channels=self.DIM_EMB, out_channels=NUM_FILTERS, kernel_size=1, padding=0)
        # bigram
        self.conv_2 = nn.Conv1d(in_channels=self.DIM_EMB, out_channels=NUM_FILTERS, kernel_size=2, padding=0)
        # trigram
        self.conv_3 = nn.Conv1d(in_channels=self.DIM_EMB, out_channels=NUM_FILTERS, kernel_size=3, padding=0)
        self.maxpool = nn.AdaptiveMaxPool1d(output_size=1)
        # ReLU nonlinearity layer
        self.nonlin = NONLINEARITY
        # hidden linear layer
        self.fc = nn.Linear(NUM_FILTERS * 3, 2)
        # softmax layer
        self.sm = nn.LogSoftmax()

    def forward(self, X, train=False):
        emb = self.emb(X)
        # make the tensor 3D for conv layers
        new_emb = torch.unsqueeze(emb.transpose(0, 1), 0)
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


def Eval_FFNN(X, Y, mlp):
    num_correct = 0
    for i in range(len(X)):
        logProbs = mlp.forward(X[i], train=False)
        pred = torch.argmax(logProbs)
        if pred == Y[i]:
            num_correct += 1
    print("Accuracy: %s" % (float(num_correct) / float(len(X))))


def Train_CNN(X, Y, vocab_size, n_iter):
    # X is XwordList, list of tensors of length 7222. Each tensor is a different length
    print("Start Training!")
    cnn_1 = CNN(X, Y, vocab_size)
    optimizer = optim.Adam(cnn_1.parameters(), lr=LEARNING_RATE)

    for epoch in range(n_iter):
        total_loss = 0.0
        for i in range(len(X)):
            # X_row = torch.from_numpy(X[i]).float()
            y_onehot = torch.zeros(cnn_1.NUM_CLASSES)
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


if __name__ == "__main__":
    train = IMDBdata("%s/train" % sys.argv[1])
    train.vocab.Lock()
    test  = IMDBdata("%s/test" % sys.argv[1], vocab=train.vocab)
    
    cnn = Train_CNN(train.XwordList, (train.Y + 1.0) / 2.0, train.vocab.GetVocabSize(), NUM_EPOCHS)
    Eval_FFNN(test.XwordList, (test.Y + 1.0) / 2.0, cnn)
