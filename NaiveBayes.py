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
from ProcessData import ProcessData

class NaiveBayes:
    def __init__(self, data, ALPHA):
        self.ALPHA = ALPHA
        
        # Dictionaries to hold word frequencies in both verified/unverified tweets
        self.verified = {}
        self.unverified = {}
        
        # Number of verified/unverified/total reviews
        self.V = 0
        self.U = 0
        self.TOTAL = len(data)
        
        #inserted print statements to indicate runtime
        self.Train(data)

    def Train(self, data):
        # In this loop, we get the word counts from each tweet and read it into
        # our verified/unverified dictionaries. We also count the number of verified tweets
        # and the number of unverified tweets.
        for tweet in data:
            if(tweet[1] == 0):
                self.U += 1
                for word in tweet[0].split():
                    if(word not in self.unverified.keys()):
                        self.unverified[word] = 1
                    else:
                        val = self.unverified[word]
                        self.unverified[word] = val + 1
            else:
                self.V += 1
                for word in tweet[0].split():
                    if(word not in self.verified.keys()):
                        self.verified[word] = 1
                    else:
                        val = self.verified[word]
                        self.verified[word] = val + 1
        
        #make sure both dictionaries contain the same words
        for key in self.unverified.keys():
            if(key not in self.verified.keys()):
                self.verified[key] = 0
        for key in self.verified.keys():
            if(key not in self.unverified.keys()):
                self.unverified[key] = 0

        #get probabilities, use Laplace smoothing
        for key in self.unverified.keys():
            # self.unverified[i] = P(x_i | unverified)
            self.unverified[key] = (self.unverified[key] + self.ALPHA) / (self.U + (len(self.unverified.keys()) * self.ALPHA))
            # self.unverified[i] = P(unverified) * P(x_i | unverified)
            self.unverified[key] = (self.U / self.TOTAL) * self.unverified[key]
            
        for key in self.verified.keys():
            # self.verified[i] = P(x_i | verified)
            self.verified[key] = (self.verified[key] + self.ALPHA) / (self.V + (len(self.verified.keys()) * self.ALPHA))
            # self.verified[i] = P(verified) * P(x_i | verified)
            self.verified[key] = (self.V / self.TOTAL) * self.verified[key]
        return

    def Predict(self, data):
        # In this loop, we take each test file and calculate if it's more
        # likely to be verified or unverified. We use logs to avoid floating
        # point underflow.
        correct = 0
        for tweet in data:
            verifiedChance = math.log(self.V / self.TOTAL)
            unverifiedChance = math.log(self.U / self.TOTAL)
            for word in tweet[0].split():
                if(word in self.verified.keys()):
                    verifiedChance += math.log(self.verified[word])
                    unverifiedChance += math.log(self.unverified[word])
            if(verifiedChance > unverifiedChance and tweet[1] == 1) or (verifiedChance <= unverifiedChance and tweet[1] == 0):
                correct += 1
        
        return correct / len(data)

if __name__ == "__main__":
    data = ProcessData("data\\non_verified_users_tweets.json", "data\\verified_users_tweets.json")
    train = data[0]
    test = data[1]
    nb = NaiveBayes(train, float(sys.argv[1]))
    print(nb.Predict(test))
