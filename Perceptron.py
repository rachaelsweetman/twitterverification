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
from ProcessData import ProcessData

class Perceptron:
    def __init__(self, train, N_ITERATIONS):
        self.N_ITERATIONS = N_ITERATIONS
        
        # initialize one weight for every feature (word) and a bias term
        # we will build the dictionary as we go since we don't know |V| yet
        self.WEIGHTS = {}
        self.BIAS = 0
        
        self.Train(train)

    def Train(self, train):
        for i in range(self.N_ITERATIONS):
            for tweet in train:
                words = tweet[0].split()
                activation = self.BIAS
                for w in words:
                    if(w in self.WEIGHTS.keys()):
                        activation += self.WEIGHTS[w]
                    else:
                        self.WEIGHTS[w] = 0
                if tweet[1] * activation <= 0:
                    for w in words:
                        self.WEIGHTS[w] += tweet[1]
                    self.BIAS += tweet[1]
        return

    def Predict(self, test):
        correct = 0
        for tweet in test:
            words = tweet[0].split()
            activation = self.BIAS
            for w in words:
                if(w in self.WEIGHTS.keys()):
                    activation += self.WEIGHTS[w]
            if(activation > 0 and tweet[1] == 1) or (activation <= 0 and tweet[1] == -1):
                correct += 1
                    
        return correct / len(test)

if __name__ == "__main__":
    data = ProcessData("data\\non_verified_users_tweets.json", "data\\verified_users_tweets.json")
    train = data[0]
    test = data[1]
    
    ptron = Perceptron(train, int(sys.argv[1]))
    print(ptron.Predict(test))