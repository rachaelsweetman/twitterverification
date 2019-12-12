import json
import requests
import random
import time
import random
import os
import unidecode
from . import twitter_credentials as cred
from requests_oauthlib import OAuth1


"""
CREDENTIALS
"""

my_credentials = cred.my_credentials
header_oauth = OAuth1(my_credentials['CONSUMER_KEY'],
                      my_credentials['CONSUMER_KEY_SECRET'],
                      my_credentials['ACCESS_TOKEN'],
                      my_credentials['ACCESS_TOKEN_SECRET'],
                      # signature_type='auth_header'
                      )


"""
TWITTER SEARCH FUNCTIONS
"""


def from_query(twitter_handle, count=100):
    return {'q': 'from:{}'.format(twitter_handle),
            'count': count
            }


def search_tweets(username):
    """ Return a JSON object containing the tweets provided in response to the given query. """
    twitter_search_url = 'https://api.twitter.com/1.1/search/tweets.json'
    response = requests.get(twitter_search_url, auth=header_oauth, params=from_query(username))
    tweets = json.loads(response.text)
    return tweets


def filter_tweets(id_list):
    str_ids = 'follow='
    for id_ in id_list:
        str_ids += ',' + str(id_)
    filter_url = 'https://stream.twitter.com/1.1/statuses/filter.json'
    response = requests.post(filter_url, auth=header_oauth, data=str_ids)
    print(response.text)
    tweets = json.loads(response.text)
    return tweets


def get_text_of_tweets(tweets_list):
    """ Given a JSON object containing a list of Tweet objects, return the text of the tweets. """
    tweets_text = []
    for status in tweets_list['statuses']:
        tweets_text.append(status['text'])
    return tweets_text


def search_user_by_id(id_):
    """ Given a username, return a string that is the Twitter handle of that user.
    If there is no user with the given ID, return None. """
    result = None
    # query = {'q': 'screen_name:{}'.format(id_)}
    query = 'user_id={}'.format(id_)
    url = 'https://api.twitter.com/1.1/users/show.json'
    response = requests.get(url, auth=header_oauth, params=query)
    user_objects = json.loads(response.text)
    if 'errors' in user_objects:
        # there is no user with this ID
        print('Error: {}'.format(user_objects['errors']))
    else:
        result = user_objects['screen_name']
    return result


"""
MISCELLANEOUS
"""


def run_out_api_calls():
    for i in range(200):
        raw_tweets = search_tweets(from_query(''))
        print(i, end=' ')
        if 'errors' in raw_tweets:
            print('All out!')
            break


def get_followers_id_list(username):
    resource_url = 'https://api.twitter.com/1.1/followers/ids.json'
    query = 'screen_name={}'.format(username)
    response = requests.get(resource_url, auth=header_oauth, params=query)
    follower_objs = json.loads(response.text)
    if 'errors' in follower_objs:
        print('Error getting followers for id: {}'.format(follower_objs['errors']))
        exit(0)
    follower_list = follower_objs['ids']
    # print(follower_list)
    return follower_list


def get_followers_batch():
    top_10_accounts = ['BarackObama',
                       'katyperry',
                       'justinbieber',
                       'rihanna',
                       'taylorswift13',
                       'Cristiano',
                       'ladygaga',
                       'TheEllenShow',
                       'YouTube',
                       'ArianaGrande'
                       ]
    followers_list = []
    for username in top_10_accounts:
        followers = get_followers_id_list(username)
        followers_list.extend(followers)
        print(username)
    print(len(followers_list))
    return followers_list


def get_all_following(username, count=5000):
    resource_url = 'https://api.twitter.com/1.1/friends/ids.json'
    cursor = 1647344786473812929
    following_ids_list = []
    iter_count = 0
    while cursor != 0:
        query = 'screen_name={}&cursor={}&count={}'.format(username, cursor, count)
        response = requests.get(resource_url, auth=header_oauth, params=query)
        following_users = json.loads(response.text)
        if 'errors' in following_users:
            print('Ran out of api calls... waiting 15 minutes.')
            time.sleep(900)
            continue
        following_ids_list.extend(following_users['ids'])
        cursor = following_users['next_cursor']
        print('Iteration {} complete.'.format(iter_count))
        iter_count += 1
    return following_ids_list


def store_all_followers(username, file_path):
    followers = get_all_following(username)
    with open(file_path, 'w') as f:
        json.dump(followers, f, indent=4)
    return


def save_tweets(ids_list, users_count):
    saved_users = 0
    saved_usernames = []
    for user_id in ids_list:
        try:
            username = search_user_by_id(user_id)
        except ValueError as e:
            print('error fetching username')
            continue
        if username not in saved_usernames:
            try:
                raw_tweets = search_tweets(username)
            except ValueError as e:
                print('error fetching username')
                continue

            if 'errors' in raw_tweets:
                print('Ran out of API calls, waiting 15 min.')
                print('Saved {} out of {} users.'.format(saved_users, users_count))
                time.sleep(900)
                raw_tweets = search_tweets(username)
            if len(raw_tweets['statuses']) > 1:
                with open('data/less_tweets_non_v/{}.json'.format(username), 'w') as f:
                    json.dump(raw_tweets, f, indent=4, sort_keys=True)
                    print(username)
                    saved_users += 1
                    saved_usernames.append(username)
            else:
                print('Not enough tweets from {}'.format(username))

        if saved_users >= users_count:
            break

    with open('saved_usernames.json', 'w') as f:
        json.dump(saved_usernames, f, indent=4, sort_keys=True)
    return


def new_non_v_ids():
    with open('non_verified_ids_50000.json', 'r') as f:
        v_ids = json.load(f)
    random.shuffle(v_ids)
    followers_list = []
    for i in range(0, 10):
        username = search_user_by_id(v_ids[i])
        followers = get_followers_id_list(username)
        followers_list.extend(followers)
    with open('non_verified_ids_new.json', 'w') as f:
        json.dump(followers_list, f, indent=4)


def prepare_data_verified():
    users_tweets = {}
    filepath = 'data/less_tweets_non_v'
    files = os.listdir(filepath)
    for file in files:
        username = file[0:len(file) - 5]
        # print(username)
        with open(filepath + '/' + file, 'r') as f:
            tweets_obj = json.load(f)
        user_obj = tweets_obj['statuses'][0]['user']
        user_verified = user_obj['verified']
        if not user_verified:
            tweets = get_text_of_tweets(tweets_obj)
            users_tweets.update({username: tweets})
    with open('non_verified_users_tweets.json', 'w') as f:
        json.dump(users_tweets, f, indent=4, sort_keys=True)


def decode_tweets():
    with open('non_verified_users_tweets.json', 'r') as f:
        tweets = json.load(f)
    new_users_tweets = {}
    for user, tweets in tweets.items():
        new_tweets = []
        for tweet in tweets:
            new_tweets.append(unidecode.unidecode(tweet))
        new_users_tweets.update({user: new_tweets})
    with open('non_verified_users_tweets_new.json', 'w') as f:
        json.dump(new_users_tweets, f, indent=4, sort_keys=True)


def remove_newlines():
    with open('verified_users_tweets.json', 'r') as f:
        tweets = json.load(f)
    new_users_tweets = {}
    for user, tweets in tweets.items():
        new_tweets = []
        for tweet in tweets:
            new_tweets.append(tweet.replace('\n', ' '))
        new_users_tweets.update({user: new_tweets})
    with open('verified_users_tweets.json', 'w') as f:
        json.dump(new_users_tweets, f, indent=4, sort_keys=True)


def main():
    remove_newlines()










