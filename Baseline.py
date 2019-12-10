from ProcessData import ProcessData

if __name__ == "__main__":
    #Open data file
    data = ProcessData("data\\non_verified_users_tweets.json", "data\\verified_users_tweets.json")
    train = data[0]
    test = data[1]
    
    #Make a |V| x 2 data structure (frequencyChart) to keep track of word frequencies for verified/unverified tweets
    #We don't know |V|, so we will build the structure as we process the file
    frequencyChart = {}
    dictionary = []
    for tweet in train:
        words = tweet[0].split()
        for w in words:
            #If we find a new word, initialize its spot in frequencyChart and add to our list of encountered words
            if(w not in dictionary):
                frequencyChart[w] = [0] * 2
                dictionary.append(w)
            
            #Update the POS table for the word
            val = frequencyChart[w]
            if(tweet[1] == 0):
                val[0] += 1
            else:
                val[1] += 1
            frequencyChart[w] = val
        
    #Make a |V| vector populated with 0's and 1's - if mostFrequent[i] = 0, then word i 
    #was found more often in unverified tweets than verified ones - if it equals 1, then
    #word i was found more often in verified tweets than unverified ones
    mostFrequent = []
    for x in dictionary:
        if(frequencyChart[x][0] > frequencyChart[x][1]):
            mostFrequent.append(0)
        else:
            mostFrequent.append(1)
    
    #TESTING - for each test tweet, calculate the 'value' of that tweet by adding 1 for each word that is found
    #more often in verified tweets and subtracting 1 for each word that is found more often in unverified tweets.
    #If the value is greater than 0, the tweet is classified as verified. Otherwise, it is classified as unverified.
    total = len(test)
    correct = 0
    for tweet in test:
        guess = 0
        words = tweet[0].split()
        for w in words:
            if(w in dictionary):
                if(mostFrequent[dictionary.index(w)] is 0):
                    guess -= 1
                else:
                    guess += 1
        
        if(guess > 0 and tweet[1] == 1) or (guess <= 0 and tweet[1] == 0):
            correct += 1
    
    #print out accuracy of baseline model
    print(correct / total)