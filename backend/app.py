import json
import os
from flask import Flask, render_template, request
from flask_cors import CORS
import pandas as pd
from flask import url_for
from models.similarity import Similarity, TempSim
from models.ml import MLModel

# ROOT_PATH for linking with all your files.
# Feel free to use a config.py or settings.py with a global export variable
os.environ['ROOT_PATH'] = os.path.abspath(os.path.join("..", os.curdir))

# Get the directory of the current script
current_directory = os.path.dirname(os.path.abspath(__file__))

# Specify the path to the JSON file relative to the current script
json_file_path = os.path.join(current_directory, 'init.json')

# Assuming your JSON data is stored in a file named 'init.json'
with open(json_file_path, 'r') as file:
    data = json.load(file)
    
    # Convert the main dataset into a DataFrame
    candles_data = []
    reviews_data = []
    
    for key, candle in data.items():
        # Creating a DataFrame for candles
        candle_info = {
            'id': key,
            'name': candle['name'],
            'category': candle['category'],
            'description': candle['description'],
            'overall_rating': candle['overall_rating'],
            'overall_reviewcount': candle['overall_reviewcount'],
            'link': candle['link'],
            'img_url': candle['img_url']
        }
        candles_data.append(candle_info)
        
        # Creating a DataFrame for reviews
        for review_key, review in candle['reviews'].items():
            review_info = {
                'candle_id': key,
                'review_body': review['review_body'],
                'rating_value': review['rating_value']
            }
        reviews_data.append(review_info)

    candles_df = pd.DataFrame(candles_data)
    reviews_df = pd.DataFrame(reviews_data)

app = Flask(__name__)
CORS(app)

def json_search(query):
    matches = []
    matches = candles_df[candles_df['name'].str.lower().str.contains(query.lower()) | 
                         candles_df['description'].str.lower().str.contains(query.lower())]

    merged_df = pd.merge(matches, reviews_df, left_on='id', right_on='candle_id', how='inner')
    merged_df['img_url'] = request.url_root + 'static/candle-' + merged_df['img_url']

    matches_filtered = merged_df[['name', 'category', 'description', 'overall_rating', 
                                  'overall_reviewcount', 'img_url', 'link', 'review_body', 'rating_value']]
    matches_filtered_json = matches_filtered.to_json(orient='records')

    print(matches_filtered_json)
    return matches_filtered_json

def cosine_sim_search(query):
    sim_df = similarity.retrieve_top_k_candles(query, 15)
    print(sim_df)
    sim_df['img_url'] = request.url_root + 'static/candle-' + sim_df['img_url']

    # Merge with reviews like in json_search
    merged_df = pd.merge(sim_df, reviews_df, left_on='id', right_on='candle_id', how='inner')

    # Filter columns to match json_search output
    filtered_df = merged_df[['name', 'category', 'description', 'overall_rating', 
                           'overall_reviewcount', 'img_url', 'link', 'review_body', 'rating_value']]

    return filtered_df.to_json(orient='records')

# singletons
similarity = TempSim(candles_df, reviews_df)

@app.route("/")
def home():
    return render_template('base.html', title="Candle Search")

@app.route("/candles")
def candles_search():
    text = request.args.get("query")
    # return json_search(text)
    return cosine_sim_search(text)

if 'DB_NAME' not in os.environ:
    app.run(debug=True, host="0.0.0.0", port=5000)
