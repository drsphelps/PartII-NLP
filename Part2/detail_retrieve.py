from bs4 import BeautifulSoup
import random
import pickle
import requests
import glob

pos_dir = "../data/POS/*.tag"
neg_dir = "../data/NEG/*.tag"

def get_all_pang_files():
    results = {}
    pos_files = glob.glob(pos_dir)
    neg_files = glob.glob(neg_dir)
    for f in pos_files:
        id = f.split('_')[1][:-4]
        with open('/mnt/d/data/pang/movie/' + id + '.html', 'r') as f:
            soup = BeautifulSoup(f.read(), 'html.parser')
            tags = soup.find_all('h1', {'class': ['title']})
            film_id = tags[0].a['href'].split('?')[1]
            genre_ids = [v['id'] for v in call_api(film_id)]
            review_profile = {'sentiment': 1, 'id': film_id, 'genres': genre_ids}
            print(review_profile)
            results[id] = review_profile
    for f in neg_files:
        id = f.split('_')[1][:-4]
        with open('/mnt/d/data/pang/movie/' + id + '.html', 'r') as f:
            soup = BeautifulSoup(f.read(), 'html.parser')
            tags = soup.find_all('h1', {'class': ['title']})
            film_id = tags[0].a['href'].split('?')[1]
            genre_ids = [v['id'] for v in call_api(film_id)]
            review_profile = {'sentiment': 0, 'id': film_id, 'genres': genre_ids}
            print(review_profile)
            results[id] = review_profile
    output = open('pang_genres.data', 'wb')
    pickle.dump(results, output)
    output.close()
    
def get_random_imdb_files():
    results = {}
    pos_files = random.sample(xrange(0, 12500), 1000)
    neg_files = random.sample(xrange(0, 12500), 1000)
    with open("../aclImdb_v1/aclImdb/train/urls_pos.txt_ign") as fp:
        for i, line in enumerate(fp):
            if i in pos_files:
                id = line.split('/')[-2]
                genres = call_api(id)
                results[id] = {'genres': genres}
    with open("../aclImdb_v1/aclImdb/train/urls_neg.txt_ign") as fp:
        for i, line in enumerate(fp):
            if i in neg_files:
                id = line.split('/')[-2]
                genres = call_api(id)
                results[id] = {'genres': genres}
    output = open('imdb_genres.data', 'wb')
    pickle.dump(results, output)
    output.close()

def call_api(id):
    request = 'https://api.themoviedb.org/3/movie/'+id+'?api_key=5ba57fd2e7a9a07195186a1ffa32e438&language=en-US'
    response = requests.get(request)
    try:   
        genres = response.json()['genres']
    except:
        genres = [] 
    return genres


get_random_imdb_files()