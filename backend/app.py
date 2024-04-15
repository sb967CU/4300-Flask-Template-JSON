import json
import os
from flask import Flask, render_template, request
from flask_cors import CORS
from helpers.MySQLDatabaseHandler import MySQLDatabaseHandler
import pandas as pd

# p03 imports
from helpers.data_analysis import build_filler_words, build_inverted_index
from collections import defaultdict, Counter
from nltk.tokenize import TreebankWordTokenizer

# p04 imports
from sklearn.preprocessing import normalize
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse.linalg import svds
import numpy as np


# ROOT_PATH for linking with all your files. 
# Feel free to use a config.py or settings.py with a global export variable
os.environ['ROOT_PATH'] = os.path.abspath(os.path.join("..",os.curdir))

# Get the directory of the current script
current_directory = os.path.dirname(os.path.abspath(__file__))

# Specify the path to the JSON file relative to the current script
json_file_path = os.path.join(current_directory, 'init.json')

# Assuming your JSON data is stored in a file named 'init.json'
with open(json_file_path, 'r') as file:
    data = json.load(file) # nested dict same format as json

    gid, gname, gphone, gweb, grat, gdesc, gaddr, gnumrev, grev = zip(*((
        gym_id,
        gym['name'],
        gym['phone'],
        gym['website'],
        gym['rating'],
        gym['description'],
        gym['address'],
        gym['num_online_reviews'],
        gym['reviews']
    ) for gym_id,gym in data.items()))

    data_df =  pd.DataFrame({
        'id': gid,
        'name': gname,
        'phone': gphone,
        'website': gweb,
        'rating': grat,
        'description': gdesc,
        'address': gaddr,
        'num_online_reviews': gnumrev,
        'reviews': grev
    })

app = Flask(__name__)
CORS(app)


# p03 TMP version, sample search using json with pandas
#def json_search(query):
#    """
#    returns: JSON of gyms that contain exact query string in description (not reviews)
#    """
#    matches = []
#    merged_df = data_df
#    matches = merged_df[merged_df['description'].str.lower().str.contains(query.lower())]
#    matches_filtered = matches[['name', 'description', 'rating', 'website']]
#    matches_filtered_json = matches_filtered.to_json(orient='records') # fxn to_json() only works with a pandas dataframe
#    return matches_filtered_json


# p03 boolean search fxn
#inverted_index = build_inverted_index(data)
#filler_words = build_filler_words(data)
#
#def boolean_search(query:str, inverted_index:dict):
#    """
#    returns: JSON of gyms filtered by boolean search on their reviews
#
#    Note: data MUST be a pandas dataframe to use to_json() needed to return output (see variable data_df)
#    """
#    filtered_tokens = TreebankWordTokenizer().tokenize(query.lower())
#    # filtered_tokens = [token for token in query_tokens if token not in filler_words]
#
#    if not filtered_tokens:
#      return json.dumps([]) # convert to json
#
#    gym_ids=set()
#    gym_ids.update(inverted_index.get(filtered_tokens[0], set()))
#    for token in filtered_tokens[1:]:
#        gym_ids.intersection_update(inverted_index.get(token, set())) 
#
#    merged_df = data_df
#    matching_gyms = merged_df[merged_df['id'].isin(gym_ids)]
#    result = matching_gyms[['name', 'description', 'rating', 'website']].to_json(orient='records')
#    return result




# p04 svd using cos sim
def p04_search(query:str, k=5):
    """
    returns: JSON of k most similar gyms ranked by cosine sim & svd search

    Note: data MUST be a pandas dataframe to use to_json() needed to return output (see variable data_df)
    """
    # build TD matrix
    vectorizer = TfidfVectorizer(stop_words='english',norm='l2',max_df=0.985,min_df=1) # arbitrary max_df
    td_matrix = vectorizer.fit_transform(
        [f"{data_df['name'][i]} {data_df['description'][i]} {' '.join(data_df['reviews'][i])}" for i in data_df.index]
        )
    td_matrix_norm = normalize(TDMatrix.toarray())


    # SVD with large k=100, just for the sake of getting many sorted singular values (aka importances)
    #      U       sigma       V^T
    docs_compressed, s, words_compressed = svds(td_matrix, k=100)
    words_compressed = words_compressed.transpose()
    docs_compressed_normed = normalize(docs_compressed)
    word_to_index = vectorizer.vocabulary_
    index_to_word = {i:t for t,i in word_to_index.items()}
    words_compressed_normed = normalize(words_compressed, axis = 1) # use normalized version??


    # cosine similarity using svd (equivalent to "closest_projects" fxn from lecture code, for reference)
    tmp_query_array = vectorizer.transform([query]).toarray()
    query_vect = normalize(np.dot(tmp_query_array, words_compressed_normed)).squeeze()

    sims = docs_compressed_normed.dot(query_vect)
    asort = np.argsort(-sims)[:k+1]
    cossim_inds = [i for i in asort[1:]]


    # return results
    gym_ids = cossim_inds
    merged_df = data_df
    matching_gyms = merged_df[merged_df['id'].isin(gym_ids)]
    result = matching_gyms[['name', 'description', 'rating', 'website']].to_json(orient='records')
    return result





@app.route("/")
def home():
    return render_template('base.html',title="sample html")

# p03 TMP version:
# @app.route("/gyms")
# def gym_search():
#    text = request.args.get("query")
#    return json_search(text)


# p03 boolean search:
#@app.route("/gyms")
#def gym_search():
#   text = request.args.get("query")
#   return boolean_search(text, inverted_index)

# p04 cos sim + svd search:
@app.route("/gyms")
def gym_search():
   text = request.args.get("query")
   return p04_search(text)


if 'DB_NAME' not in os.environ:
    app.run(debug=True,host="0.0.0.0",port=5000)