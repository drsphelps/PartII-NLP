from bs4 import BeautifulSoup
import pickle
import requests
import glob

pos_dir = "../data/POS/*.tag"
neg_dir = "../data/NEG/*.tag"

def get_all_files():
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
    output = open('genres.data', 'wb')
    pickle.dump(results, output)
    output.close()


def call_api(id):
    request = 'https://api.themoviedb.org/3/movie/tt'+id+'?api_key=5ba57fd2e7a9a07195186a1ffa32e438&language=en-US'
    response = requests.get(request)
    try:   
        genres = response.json()['genres']
    except:
        genres = [] 
    return genres

get_all_files()