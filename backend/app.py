import json
import os
from flask import Flask, render_template, request
from flask_cors import CORS
from helpers.MySQLDatabaseHandler import MySQLDatabaseHandler
import pandas as pd

###########################################################
###########################################################
# TODO: fix boolean search attempt:
from helpers.data_analysis import build_filler_words, build_inverted_index
from collections import defaultdict, Counter
from nltk.tokenize import TreebankWordTokenizer
###########################################################
###########################################################


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


# TMP version, sample search using json with pandas
def json_search(query):
    """
    returns: JSON of gyms that contain exact query string in description (not reviews)
    """
    matches = []
    merged_df = data_df
    matches = merged_df[merged_df['description'].str.lower().str.contains(query.lower())]
    matches_filtered = matches[['name', 'description', 'rating', 'website']]
    matches_filtered_json = matches_filtered.to_json(orient='records') # fxn to_json() only works with a pandas dataframe
    return matches_filtered_json


###########################################################
###########################################################
# TODO: fix boolean search attempt:

inverted_index = build_inverted_index(data)
filler_words = build_filler_words(data)

def boolean_search(query:str, inverted_index:dict):
    """
    returns: JSON of gyms filtered by boolean search on their reviews

    Note: data MUST be a pandas dataframe to use to_json() needed to return output (see variable data_df)
    """
    # TODO: edit and debug to fit spec and return same output format as json_search()
    filtered_tokens = TreebankWordTokenizer().tokenize(query.lower())
    # filtered_tokens = [token for token in query_tokens if token not in filler_words]

    if not filtered_tokens:
      return json.dumps([]) # convert to json

    gym_ids=set()
    gym_ids.update(inverted_index.get(filtered_tokens[0], set()))
    for token in filtered_tokens[1:]:
        gym_ids.intersection_update(inverted_index.get(token, set())) 

    print(len(filtered_tokens))
    merged_df = data_df
    matching_gyms = merged_df[merged_df['id'].isin(gym_ids)]
    result = matching_gyms[['name', 'description', 'rating', 'website']].to_json(orient='records')
    return result
###########################################################
###########################################################





@app.route("/")
def home():
    return render_template('base.html',title="sample html")

# # TMP version:
# @app.route("/gyms")
# def gym_search():
#    text = request.args.get("query")
#    return json_search(text)


###########################################################
###########################################################
# TODO: fix boolean search attempt:
# uncomment and fix, (comment out the TMP version):

@app.route("/gyms")
def gym_search():
   text = request.args.get("query")
   return boolean_search(text, inverted_index)

###########################################################
###########################################################


if 'DB_NAME' not in os.environ:
    app.run(debug=True,host="0.0.0.0",port=5000)