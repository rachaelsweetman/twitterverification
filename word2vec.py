import gensim.models
import gensim.downloader as gdl
import gensim.utils
from gensim.test.utils import get_tmpfile
from gensim.models import KeyedVectors
import json
import random
import re


def load_twitter_data():
    with open('data/verified_users_tweets.json', 'r') as f:
        verified_users = json.load(f)
    with open('data/non_verified_users_tweets.json', 'r') as f:
        nonverified_users = json.load(f)
    return verified_users, nonverified_users


def load_twitter_data_cut():
    verified_users, nonverified_users = load_twitter_data()

    new_verified = reduce_tweet_count(verified_users, 10)
    new_unverified = reduce_tweet_count(nonverified_users, 10)

    return new_verified, new_unverified


def reduce_tweet_count(users_tweets, max_tweets):
    result = {}
    for username, tweets in users_tweets.items():
        if len(tweets) > max_tweets:
            random.shuffle(tweets)
            tweets = tweets[:max_tweets]
        result.update({username: tweets})
    return result


def get_sentences(verified_users, nonverified_users):
    """Return 2 lists, where each list contains lists (tweets) whose contents are the words in the tweet."""

    verified_tweets = []
    for user, tweets in verified_users.items():
        verified_tweets.extend(tweets)

    nonverified_tweets = []
    for user, tweets in nonverified_users.items():
        nonverified_tweets.extend(tweets)

    verified_tweets = [[preprocess_string(j) for j in i.split() if 'http' not in j] for i in verified_tweets]
    nonverified_tweets = [[preprocess_string(j) for j in i.split() if 'http' not in j] for i in nonverified_tweets]

    return verified_tweets, nonverified_tweets


def preprocess_string(in_str):
    """Make lowercase and strip punctuation from a given string. Also strips apostrophes."""
    in_str = in_str.lower()
    output = re.sub(r'[^\w\s]', '', in_str)
    return output


def train_word2vec(sentences, dim_emb):
    model = gensim.models.Word2Vec(sentences=sentences,
                                   min_count=2,
                                   size=dim_emb,                # embedding dimension
                                   window=5,
                                   sg=1                     # skip gram when 1, CBOW when 0
                                   )
    return model


def train_word2vec_twitter(dim_emb=100, min_count=2, window=5, skipgram=1):
    ver, nonver = load_twitter_data()
    sentences, nonverified = get_sentences(ver, nonver)
    sentences.extend(nonverified)
    model = gensim.models.Word2Vec(sentences=sentences,
                                   min_count=min_count,
                                   size=dim_emb,                # embedding dimension
                                   window=window,
                                   sg=skipgram                    # skip gram when 1, CBOW when 0
                                   )
    return model



def load_word2vec_twitter():
    return gensim.models.Word2Vec.load('word2vec_models/word2vec_twitter_v3.pkl')


# def load_word_vectors_google_from_web():
#     return gdl.load('word2vec-google-news-300')
#
#
# def load_word_vectors_google():
#     word_vectors = KeyedVectors.load('word2vec_models/word2vec_google.pkl', mmap='r')
#     return word_vectors
#
#
# def store_google_wvs():
#     model = load_word_vectors_google_from_web()
#     word_vectors = model.wv
#     fname = get_tmpfile("word_vectors_google.kv")
#     word_vectors.save(fname)
#     return


def main():
    ver, nonver = load_twitter_data()
    sentences, nonverified = get_sentences(ver, nonver)
    sentences.extend(nonverified)
    # model = train_word2vec_twitter(all_sent)
    # model.save('word2vec_models/word2vec_twitter_v3.pkl')
    model = load_word2vec_twitter()

    print(len(model.wv.vocab))
    print(model.most_similar('twitter'))
    print(model.wv.most_similar('facebook'))
    print(model.wv.most_similar('computer'))
    print(model.wv.most_similar('book'))
    print(model.wv.most_similar('road'))



