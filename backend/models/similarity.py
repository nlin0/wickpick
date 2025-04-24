import re
import pandas as pd
import numpy as np
import sklearn
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from scipy.sparse.linalg import svds
# import nltk
# from nltk.stem.wordnet import WordNetLemmatizer
# from nltk.stem.snowball import SnowballStemmer

# from models.candle import Candle

class PandasSim:
    def __init__(self, candles_df, reviews_df):
        # print("PD Sims init")
        # nltk.download('wordnet')
        self.candles = candles_df
        # print(reviews_df)
        self.reviews = reviews_df['review_body'].tolist()
        # print(self.reviews)
        self.review_idx_to_candle_idx = {i: reviews_df.iloc[i]['candle_id'] for i in range(len(reviews_df))}
        # self.tfidf_vectorizer = TfidfVectorizer(tokenizer=self.custom_tokenizer, stop_words='english')
        self.tfidf_vectorizer = TfidfVectorizer(stop_words='english')
        # First fit on all text to establish the vocabulary
        all_text = [r if r is not None else "" for r in self.reviews] + [d if d is not None else "" for d in self.candles['description']]
        self.tfidf_vectorizer.fit(all_text)
        # Then transform each separately
        self.tfidf_reviews = self.tfidf_vectorizer.transform([r if r is not None else "" for r in self.reviews]).toarray()
        self.tfidf_description = self.tfidf_vectorizer.transform([r if r is not None else "" for r in self.candles['description']]).toarray()

    # HELPER FUNCTIONS
    # Takes in string query and transforms it
    def transform_query(self, query):
        return self.tfidf_vectorizer.transform([query]).toarray()[0]
    
    def helper_cosine_sim(self, vec1, vec2):
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

    def helper_jaccard_sim(self, vec1, vec2):
        # jaccard on unique terms
        return np.intersect1d(vec1, vec2) / np.union1d(vec1, vec2)

    # SIMILARITY FUNCTIONS
    def cosine_sim_candles(self, id1, id2):
        rev1 = self.tfidf_reviews[id1]
        rev2 = self.tfidf_reviews[id2]
        cosine_sim = self.helper_cosine_sim(rev1, rev2)
        return cosine_sim
    
    def cosine_sim_query_candles(self, query, candle_id):
        transformed_query = self.transform_query(query)
        candle_rev = self.tfidf_reviews[candle_id]
        norm_query = np.linalg.norm(transformed_query)
        if norm_query == 0:
            return 0
        cosine_sim = self.helper_cosine_sim(transformed_query, candle_rev)
        return cosine_sim
    
    def retrieve_top_k_candles(self, query, k):
        '''
        Returns the top k candles based on a custom similarity score:
        Similarity between a query and a candle is the weighted sum of 
        the cosine similarity between the query and the reviews, the
        cos sim between the query and the description, and the jaccard
        sim between the query and the candle name.
        '''
        review_sims = {}
        for i in range(len(self.reviews)):
            candle_id = self.review_idx_to_candle_idx[i]
            sim = self.cosine_sim_query_candles(query, i)
            if candle_id in review_sims:
                review_sims[candle_id] = max(review_sims[candle_id], sim)
            else:
                review_sims[candle_id] = sim
        
        desc_sims = {}
        for i in range(len(self.candles)):
            candle_id = i  
            desc_sim = self.helper_cosine_sim(self.tfidf_description[i], self.transform_query(query))
            desc_sims[candle_id] = desc_sim

        name_sims = {}
        for i in range(len(self.candles)):
            cand_name = self.candles[i]["name"]
            name_sim = self.helper_jaccard_sim(cand_name, query)
            name_sim[i] = name_sim
        
        combined_sims = {}
        for candle_id in set(review_sims.keys()) | set(desc_sims.keys()) | set(name_sims.keys()):
            rev_sim = review_sims.get(candle_id, 0)
            desc_sim = desc_sims.get(candle_id, 0)
            name_sim = name_sims.get(candle_id, 0)
            w1, w2, w3 = 0.2, 0.4, 0.4
            combined_sims[candle_id] = (w1 * name_sim) + (w2 * rev_sim) + (w3 * desc_sim)
        
        sorted_candle_ids = sorted(combined_sims.keys(), key=lambda cid: combined_sims[cid], reverse=True)
        
        # Return top k unique candles
        top_k_ids = sorted_candle_ids[:k]
        
        return self.candles.iloc[top_k_ids]

    def retrieve_bottom_k_candles(self,query,k):
        review_sims = {}
        for i in range(len(self.reviews)):
            candle_id = self.review_idx_to_candle_idx[i]
            sim = self.cosine_sim_query_candles(query, i)
            if candle_id in review_sims:
                review_sims[candle_id] = max(review_sims[candle_id], sim)
            else:
                review_sims[candle_id] = sim
        
        desc_sims = {}
        for i in range(len(self.candles)):
            candle_id = i  
            desc_sim = self.helper_cosine_sim(self.tfidf_description[i], self.transform_query(query))
            desc_sims[candle_id] = desc_sim
        
        combined_sims = {}
        for candle_id in set(review_sims.keys()) | set(desc_sims.keys()):
            rev_sim = review_sims.get(candle_id, 0)
            desc_sim = desc_sims.get(candle_id, 0)
            combined_sims[candle_id] = 0.5 * rev_sim + 0.5 * desc_sim
        
        sorted_candle_ids = sorted(combined_sims.keys(), key=lambda cid: combined_sims[cid], reverse=False)

        top_k_ids = sorted_candle_ids[:k]
        
        return self.candles.iloc[top_k_ids]

    
    def rocchio(self, query, alpha = 1, beta = 0.75, gamma = 0.15):
        # IMPLEMENT ROCCHIO HERE
        # going to use for query suggestions ("did you mean: ...?") like google
        query_vec = self.transform_query(query)
    
        relevant_candles = self.retrieve_top_k_candles(query, 10)
        relevant_ids = relevant_candles.index.tolist()
        
        irrelevant_candles = self.retrieve_bottom_k_candles(query, 10)
        irrelevant_ids = irrelevant_candles.index.tolist()
        
        #relevant centroid
        if relevant_ids:
            relevant_vecs = []
            for candle_id in relevant_ids:
                relevant_vecs.append(self.tfidf_description[candle_id])
            
            relevant_centroid = np.mean(relevant_vecs, axis=0) if relevant_vecs else np.zeros_like(query_vec)
        else:
            relevant_centroid = np.zeros_like(query_vec)
        
        # irrelevant centroid
        if irrelevant_ids:
            irrelevant_vecs = []
            for candle_id in irrelevant_ids:
                irrelevant_vecs.append(self.tfidf_description[candle_id])
            
            irrelevant_centroid = np.mean(irrelevant_vecs, axis=0) if irrelevant_vecs else np.zeros_like(query_vec)
        else:
            irrelevant_centroid = np.zeros_like(query_vec)
        
        modified_query_vec = alpha * query_vec + beta * relevant_centroid - gamma * irrelevant_centroid
        return modified_query_vec
    
    def get_query_suggestions(self, modified_query_vec, num_terms=5):
        feature_names = np.array(self.tfidf_vectorizer.get_feature_names_out())
        top_indices = modified_query_vec.argsort()[-num_terms:][::-1]
        top_terms = feature_names[top_indices]
        return " ".join(top_terms)
    
    def svd(self):
        # IMPLEMENT SVD HERE
        return