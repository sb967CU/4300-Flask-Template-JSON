import json
import os
from flask import Flask, render_template, request
from flask_cors import CORS
from helpers.MySQLDatabaseHandler import MySQLDatabaseHandler
import pandas as pd
from collections import defaultdict, Counter
from nltk.tokenize import TreebankWordTokenizer
from helpers.data_analysis import *

# ROOT_PATH for linking with all your files. 
# Feel free to use a config.py or settings.py with a global export variable
os.environ['ROOT_PATH'] = os.path.abspath(os.path.join("..",os.curdir))

# Get the directory of the current script
current_directory = os.path.dirname(os.path.abspath(__file__))

# Specify the path to the JSON file relative to the current script
json_file_path = os.path.join(current_directory, 'data.json')

# Assuming your JSON data is stored in a file named 'init.json'
with open(json_file_path, 'r') as file:
    data = json.load(file)
    episodes_df = pd.DataFrame(data['episodes'])
    reviews_df = pd.DataFrame(data['reviews'])

app = Flask(__name__)
CORS(app)

inverted_index = build_inverted_index(data)
# Sample search using json with pandas
def json_search(query):
    matches = []
    merged_df = pd.merge(episodes_df, reviews_df, left_on='id', right_on='id', how='inner')
    matches = merged_df[merged_df['title'].str.lower().str.contains(query.lower())]
    matches_filtered = matches[['title', 'descr', 'imdb_rating']]
    matches_filtered_json = matches_filtered.to_json(orient='records')
    return matches_filtered_json

def boolean_search(query:str, inverted_index:dict):
    res = set()
    if not query:
      return res
    full_query_tokens = set(TreebankWordTokenizer().tokenize(query))

    query_tokens=[]
    for token in full_query_tokens:
      if token not in filler_words:
        query_tokens.append(token)

    res=inverted_index.get(query_tokens[0])
    for token in query_tokens:
        res = res.intersection(inverted_index.get(token, set()))
    return res


@app.route("/")
def home():
    return render_template('base.html',title="sample html")

@app.route("/episodes")
def episodes_search():
    text = request.args.get("title")
    return json_search(text)

@app.route("/gyms")
def gym_search():
   text = request.args.get("query")
   return boolean_search(text, inverted_index).to_json(orient='records')


if 'DB_NAME' not in os.environ:
    app.run(debug=True,host="0.0.0.0",port=5000)