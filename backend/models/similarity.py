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

class PandasSim:
    def __init__(self, candles_df, reviews_df):
        self.candles = candles_df
        self.reviews = reviews_df['review_body'].tolist()
        self.review_idx_to_candle_idx = {i: reviews_df.iloc[i]['candle_id'] for i in range(len(reviews_df))}

        # TODO: Fix the nltk thingy so that the lemmatizer works
        # self.tfidf_vectorizer = TfidfVectorizer(tokenizer=self.custom_tokenizer, stop_words='english')
        self.tfidf_vectorizer = TfidfVectorizer(stop_words='english', min_df=1, max_df=0.8)

        # First fit on all text to establish the vocabulary
        all_text = [r if r is not None else "" for r in self.reviews] + [d if d is not None else "" for d in self.candles['description']]
        self.tfidf_vectorizer.fit(all_text)

        # Then transform each separately
        self.tfidf_reviews = self.tfidf_vectorizer.transform([r if r is not None else "" for r in self.reviews]).toarray()
        self.tfidf_description = self.tfidf_vectorizer.transform([r if r is not None else "" for r in self.candles['description']]).toarray()

        # Apply SVD to both
        self.reviews_compressed_normed, self.reviews_words_compressed = self.perform_svd(self.tfidf_reviews, k=40)
        self.descriptions_compressed_normed, self.descriptions_words_compressed = self.perform_svd(self.tfidf_description, k=40)

    # HELPER FUNCTIONS
    # def custom_tokenizer(self, corpus):
    #     stemmer = WordNetLemmatizer()
    #     words = re.sub(r"[^A-Za-z0-9\-]", " ", corpus).lower().split()
    #     return [stemmer.lemmatize(word) for word in words]
    
    def generic_tokenizer(self, corpus):
        return re.sub(r"[^A-Za-z0-9\-]", " ", corpus).lower().split()

    # SVD INITIALIZATION HELPERS
    def perform_svd(self, tfidf_mat, k=40):
        docs_compressed, s, words_compressed_T = svds(tfidf_mat, k=k)
        words_compressed = words_compressed_T.transpose()
        return normalize(docs_compressed), words_compressed
    
    def find_opt_k():
        #TODO: find the optimal k for svd
        return
    
    # GENERIC SIMILARITY FUNCTION
    def helper_cosine_sim(self, vec1, vec2):
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

    def helper_jaccard_sim(self, vec1, vec2):
        # jaccard on unique terms
        union = len(np.union1d(vec1, vec2))
        if union == 0:
            return 0
        return len(np.intersect1d(vec1, vec2)) / union

    # QUERY TRANSFORMATION
    def transform_query(self, query):
        # Takes in string query and transforms it
        return self.tfidf_vectorizer.transform([query]).toarray()[0]
    
    def transform_query_svd(self, query, words_compressed):
        # Takes in string query and transforms it (for SVD usage)
        query_tfidf = self.tfidf_vectorizer.transform([query]).toarray()
        query_vec = normalize(np.dot(query_tfidf, words_compressed)).squeeze()
        return query_vec
    
    # SIMILARITY METHODS
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
    
    def retrieve_top_k_candles_svd(self, query, k, w1=0.2, w2=0.4, w3=0.4):
        '''
        Returns the top k candles based on a custom similarity score:
        Similarity between a query and a candle is the weighted sum of 
        the cosine similarity between the query and the reviews, the
        cos sim between the query and the description, and the jaccard
        sim between the query and the candle name.

        If the query does not yield good distribution (i.e. highest and 
        lowest are not diff enough), then perform rocchio and redo?
        '''
        print("count of all candles:", len(self.candles))

        # Review similarity
        modified_query = self.transform_query_svd(query, self.reviews_words_compressed)
        review_sims_per_review = self.helper_cosine_sim(self.reviews_compressed_normed, modified_query)
        review_df = pd.DataFrame({
            'candle_id': [self.review_idx_to_candle_idx[i] for i in range(len(review_sims_per_review))],
            'similarity': review_sims_per_review
        })
        review_sims = review_df.groupby('candle_id')['similarity'].mean().to_dict()
        print("rev sims", review_sims, len(review_sims))
        print()

        # Description similarity 
        modified_query = self.transform_query_svd(query, self.descriptions_words_compressed)
        desc_sims_list = self.helper_cosine_sim(self.descriptions_compressed_normed, modified_query)
        desc_sims = {i: sim for i, sim in enumerate(desc_sims_list)}
        print("desc sims", desc_sims, len(desc_sims))

        # Name similarity
        name_sims = {}
        query_tokenized = self.generic_tokenizer(query)
        for i in range(len(self.candles)):
            cand_name_tokenized = self.generic_tokenizer(self.candles.loc[i, "name"])
            name_sims[i] = self.helper_jaccard_sim(query_tokenized, cand_name_tokenized)

        combined_sims = {}
        for candle_id in set(review_sims.keys()) | set(desc_sims.keys()) | set(name_sims.keys()):
            rev_sim = review_sims.get(candle_id, 0)
            desc_sim = desc_sims.get(candle_id, 0)
            name_sim = name_sims.get(candle_id, 0)
            combined_sims[candle_id] = (w1 * name_sim) + (w2 * rev_sim) + (w3 * desc_sim)
        
        sorted_candle_ids = sorted(combined_sims.keys(), key=lambda cid: combined_sims[cid], reverse=True)
        
        # Return top k unique candles
        top_k_ids = sorted_candle_ids[:k]
        
        return self.candles.iloc[top_k_ids]

    
    def retrieve_top_k_candles(self, query, k, w1=0.2, w2=0.4, w3=0.4):
        '''
        Returns the top k candles based on a custom similarity score:
        Similarity between a query and a candle is the weighted sum of 
        the cosine similarity between the query and the reviews, the
        cos sim between the query and the description, and the jaccard
        sim between the query and the candle name.

        If the query does not yield good distribution (i.e. highest and 
        lowest are not diff enough), then perform rocchio and redo?
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
        query_tokenized = self.generic_tokenizer(query)
        for i in range(len(self.candles)):
            # print(self.candles.loc[candle_id, 'name'], i)
            cand_name_tokenized = self.generic_tokenizer(self.candles.loc[candle_id, 'name'])
            name_sims[i] = self.helper_jaccard_sim(query_tokenized, cand_name_tokenized)
        
        combined_sims = {}
        for candle_id in set(review_sims.keys()) | set(desc_sims.keys()) | set(name_sims.keys()):
            rev_sim = review_sims.get(candle_id, 0)
            desc_sim = desc_sims.get(candle_id, 0)
            name_sim = name_sims.get(candle_id, 0)
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