import json
import random
import re


def ProcessData(filename1, filename2):
    
    # read in data
    unverified_file = open(filename1, 'r')
    verified_file = open(filename2, 'r')
    
    unverified_data = json.load(unverified_file)
    verified_data = json.load(verified_file)
    
    # save our data sets into unverified_data and verified_data
    # as we add, remove all words that contain '@' or 'https' or 'RT' to remove links/mentions/'RT'
    data = []
    for user in unverified_data.values():
        for tweet in user:
            words = tweet.split()
            s = ""
            for w in words:
                if ("https" not in w) and ("@" not in w) and (w ): 
                    s += w + ' '
            data.append([preprocess_string(s), -1])
    unverified_data = data
        
    data = []
    for user in verified_data.values():
        for tweet in user:
            words = tweet.split()
            s = ""
            for w in words:
                if ("https" not in w) and ("@" not in w) and ("RT" not in w): 
                    s = s + w + ' '
            data.append([preprocess_string(s), 1])
    verified_data = data
    
    # shuffle the data
    random.shuffle(unverified_data)
    random.shuffle(verified_data)
    
    # trim so they are both the same size (we want about a 50/50 split of verified/unverified in our data)
    if(len(unverified_data) > len(verified_data)):
        unverified_data = unverified_data[:len(verified_data)]
    else:
        verified_data = verified_data[:len(unverified_data)]
    
    # split data in train/test sets
    unverifiedTrain, unverifiedTest = unverified_data[:int(len(unverified_data) * 0.8)], unverified_data[int(len(unverified_data) * 0.8):]
    verifiedTrain, verifiedTest = verified_data[:int(len(verified_data) * 0.8)], verified_data[int(len(verified_data) * 0.8):]
    
    # initialize return lists
    train = unverifiedTrain + verifiedTrain
    test = unverifiedTest + verifiedTest
    
    # shuffle the data
    random.shuffle(train)
    random.shuffle(test)
    
    # only return some of the data (this helps change our data every time we run the model)
    return [train[:len(train)], test[:len(test)]]


def preprocess_string(str_):
    return re.sub(r'[^\w\s]', '', str_.lower())
