import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from scipy.sparse.linalg import svds
import sklearn
from candle import Candle

class Similarity:
    def __init__(self, candles):
        print("init")
        self.candles = candles
        self.reviews = [candle.ind_review for candle in candles]
        self.tfidf_vectorizer = TfidfVectorizer(stop_words = 'english')  # optimize parameters later
        self.tfidf_reviews = self.tfidf_vectorizer.fit_transform([r for r in self.reviews]).toarray()

        # print(self.tfidf_vectorizer.get_feature_names_out())
        # print(self.tfidf_reviews)

    # HELPER FUNCTIONS
    # Takes in string query and transforms it
    def transform_query(self, query):
        return self.tfidf_vectorizer.transform([query]).toarray()[0]
    
    def calc_cosine_sim(self, a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    # SIMILARITY FUNCTIONS
    def retrieve_top_k_candles(self, query, k):
        transformed_query = self.transform_query(query)
        if np.linalg.norm(transformed_query) == 0:
            return []
        
        sims = []
        for id in range(len(self.candles)):
            cosine_sim = self.calc_cosine_sim(transformed_query, self.tfidf_reviews[id])
            sims.append((id, cosine_sim))

        sims.sort(key=lambda x: x[1], reverse=True)
        return [self.candles[x[0]] for x in sims[:k]]

    # Get cosine similarity between two candles
    def cosine_sim_candles(self, id1, id2):
        # cosine_similarities = sklearn.metrics.pairwise.cosine_similarity(self.tfidf_reviews, self.tfidf_reviews[candle_id])
        rev1 = self.tfidf_reviews[id1]
        rev2 = self.tfidf_reviews[id2]
        cosine_sim = self.calc_cosine_sim(rev1, rev2)
        return cosine_sim

    # Get cosine similarity between a query and a candle
    def cosine_sim_query_candles(self, query, candle_id):
        # print()
        transformed_query = self.transform_query(query)
        candle_rev = self.tfidf_reviews[candle_id]
        # print(transformed_query)
        # print(candle_rev)
        norm_query = np.linalg.norm(transformed_query)
        if np.linalg.norm(transformed_query) == 0:
            return 0
        cosine_sim = self.calc_cosine_sim(transformed_query, candle_rev)
        return cosine_sim
    
    def rocchio(self, query, relevant, irrelevant, alpha = 1, beta = 0.75, gamma = 0.15):
        return

    def svd_similarity(self, candle_id):
        u, s, vt = svds(self.tfidf_reviews, k = 100)
        s_diag_matrix = np.diag(s)
        X_pred = np.dot(np.dot(u, s_diag_matrix), vt)
        cosine_similarities = self.cosine_similarities(X_pred, X_pred[candle_id])
        return cosine_similarities

# TEST
candles = [
    Candle(0, 'Candle 1', 'www.candle1.com', 'This is a candle', 4.5, 100, 'This is the first review', 4.5, 100),
    Candle(1, 'Candle 2', 'www.candle2.com', 'This is a candle', 4.5, 100, 'And this is a second review', 4.5, 100),
    Candle(2, 'Candle 3', 'www.candle3.com', 'This is a candle', 4.5, 100, 'Third review', 4.5, 100),
    Candle(3, 'Candle 4', 'www.candle4.com', 'This is a candle', 4.5, 100, 'I am the fourth review', 4.5, 100),
    Candle(4, 'Candle 5', 'www.candle5.com', 'This is a candle', 4.5, 100, 'I\'m the fifth review', 4.5, 100),
    Candle(5, 'Candle 6', 'www.candle6.com', 'This is a candle', 4.5, 100, 'This is the first review', 4.5, 100),
]
sim = Similarity(candles)

# print(sim.cosine_sim_candles(0, 5))
# print(sim.cosine_sim_query_candles('This is the fifth review', 0))
# print(sim.cosine_sim_query_candles('This is the first review', 0))
