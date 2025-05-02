import re
import pandas as pd
import numpy as np
import sklearn
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from scipy.sparse.linalg import svds

class PandasSim:
    def __init__(self, candles_df, reviews_df):
        self.candles = candles_df
        self.reviews = reviews_df['review_body'].tolist()
        self.review_idx_to_candle_idx = {i: int(reviews_df.iloc[i]['candle_id']) for i in range(len(reviews_df))}

        my_stop_words = [
            "scent", "notes", "note", "nice", "love", "favorite", "smells", "like", "fragrance", "time",
            "base", "top", "mid", "middle", "one", "two", "three", "oz", "ounce", "inch", "inches",
            "smell", "aroma", "candle", "candles", "burning", "burn", "jar", "wax", "wick", "flame",
            "great", "good", "loved", "liked", "beautiful", "day", "days", "time", "long", "hour", "hours",
            "yankee", "boy", "smells", "yankeecandle", "boysmells"
        ]
        stop_words_list = list(text.ENGLISH_STOP_WORDS) + my_stop_words
        self.tfidf_vectorizer = TfidfVectorizer(stop_words=stop_words_list, min_df=1, max_df=0.95)

        # Fit on all text to establish the vocabulary
        names = [n if n is not None else "" for n in self.candles['name']]
        revs = [r if r is not None else "" for r in self.reviews]
        descs = [d if d is not None else "" for d in self.candles['description']]
        all_text = names + revs + descs
        self.tfidf_vectorizer.fit(all_text)

        # Then transform each separately
        self.tfidf_reviews = self.tfidf_vectorizer.transform([r if r is not None else "" for r in self.reviews]).toarray()
        self.tfidf_description = self.tfidf_vectorizer.transform([r if r is not None else "" for r in self.candles['description']]).toarray()
        self.tfidf_all = self.getTfidfAll()

        # Apply SVD to both (tuning k by manually looking at singular values)
        self.reviews_compressed_normed, self.reviews_words_compressed = self.perform_svd(self.tfidf_reviews, k=12)
        self.descriptions_compressed_normed, self.descriptions_words_compressed = self.perform_svd(self.tfidf_description, k=12)
        self.all_compressed_normed, self.all_words_compressed = self.perform_svd(self.tfidf_all, k=12)

        # Extract vocab
        word_to_index = self.tfidf_vectorizer.vocabulary_
        self.index_to_word = {i:t for t,i in word_to_index.items()}
        
        # Get top 5 words for each candle
        self.top_words_by_id = {i: self.get_top_n_candle_dimensions(i) for i in range(len(self.candles))}

        # Get top 3 similar candles for each candle
        self.similar_candles_by_id = {i: self.find_n_similar_candles(i, n=3) for i in range(len(self.candles))}

        # prints each dim's top words + values across corpus
        # self.svd_dim_words_values(self.all_words_compressed)

    # HELPER FUNCTIONS
    def getTfidfAll(self):
        reviews_by_candle = {}
        for review_idx, candle_id in self.review_idx_to_candle_idx.items():
            candle_idx = candle_id - 1
            review_text = self.reviews[review_idx] if review_idx < len(self.reviews) and self.reviews[review_idx] is not None else ""
            
            if candle_idx not in reviews_by_candle:
                reviews_by_candle[candle_idx] = ""
            reviews_by_candle[candle_idx] += (review_text + " ")

        # Create combined corpus for each candle (name + description + all reviews)
        combined_candle_text = []
        for i in range(len(self.candles)):
            description = self.candles.loc[i, 'description'] if self.candles.loc[i, 'description'] is not None else ""
            name = self.candles.loc[i, 'name'] if self.candles.loc[i, 'name'] is not None else ""
            all_reviews_text = reviews_by_candle.get(i, "")

            combined_text = f"{name} {description} {all_reviews_text}"
            combined_candle_text.append(combined_text)

        return self.tfidf_vectorizer.transform(combined_candle_text).toarray()
    
    # edit distance algo
    def edit_distance(self, s1, s2):
        dp = np.zeros((len(s1)+1, len(s2)+1), dtype=int)
        for i in range(len(s1)+1):
            for j in range(len(s2)+1):
                if i == 0:
                    dp[i][j] = j
                elif j == 0:
                    dp[i][j] = i
                elif s1[i-1] == s2[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
        return dp[len(s1)][len(s2)]

    def generic_tokenizer(self, corpus):
        return re.sub(r"[^A-Za-z0-9\-]", " ", corpus).lower().split()

    # SVD INITIALIZATION HELPERS
    def perform_svd(self, tfidf_mat, k=12):
        docs_compressed, s, words_compressed_T = svds(tfidf_mat, k=k)
        words_compressed = words_compressed_T.transpose()
        print(f"Singular values: {s}")
        return normalize(docs_compressed), words_compressed
    
    # GENERIC SIMILARITY FUNCTION
    def helper_cosine_sim(self, vec1, vec2):
        sim = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
        if isinstance(sim, (np.ndarray, list)):
            return np.nan_to_num(sim, nan=0.0, posinf=0.0, neginf=0.0)
        return 0 if np.isnan(sim) or np.isinf(sim) else sim
    
    # def helper_cosine_sim(self, vec1, vec2):
    #     return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

    def helper_jaccard_sim(self, vec1, vec2, edit_threshold=2):
        # jaccard on unique terms
        # print(f"Comparing token sets: {vec1} vs {vec2}")
        matched1 = set()
        matched2 = set()

        for i, token1 in enumerate(vec1):
            for j, token2 in enumerate(vec2):
                dist = self.edit_distance(token1, token2)
                if dist <= edit_threshold:
                    # print(f"MATCH: {token1} ~ {token2} (dist={dist})")
                    matched1.add(i)
                    matched2.add(j)

        intersection = len(matched1)
        union = len(set(range(len(vec1))) | set(range(len(vec2))))

        if union == 0:
            return 0
        return intersection / union

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

    # SVD STUFF
    def retrieve_sorted_candles_svd(self, query):
        '''
        Returns candles sorted based on a custom similarity score:
        Similarity between a query and a candle is the weighted sum of 
        the cosine similarity between the query and the reviews, the
        cos sim between the query and the description, and the jaccard
        sim between the query and the candle name.

        If the query does not yield good distribution (i.e. highest and 
        lowest are not diff enough), then perform rocchio and redo?
        '''
        # print("count of all candles:", len(self.candles))

        # Review similarity
        modified_query = self.transform_query_svd(query, self.reviews_words_compressed)
        # review_sims_per_review = self.helper_cosine_sim(self.reviews_compressed_normed, modified_query)
        review_sims_per_review = self.reviews_compressed_normed.dot(modified_query)
        review_df = pd.DataFrame({
            'candle_id': [(self.review_idx_to_candle_idx[i] - 1) for i in range(len(review_sims_per_review))],
            'similarity': review_sims_per_review
        })
        review_sims = review_df.groupby('candle_id')['similarity'].mean().to_dict()

        # Description similarity 
        modified_query = self.transform_query_svd(query, self.descriptions_words_compressed)
        # desc_sims_list = self.helper_cosine_sim(self.descriptions_compressed_normed, modified_query)
        desc_sims_list = self.descriptions_compressed_normed.dot(modified_query)
        desc_sims = {i: sim for i, sim in enumerate(desc_sims_list)}

        # Name similarity
        name_sims = {}
        query_tokenized = self.generic_tokenizer(query)
        for i in range(len(self.candles)):
            cand_name_tokenized = self.generic_tokenizer(self.candles.loc[i, "name"])
            name_sims[i] = self.helper_jaccard_sim(query_tokenized, cand_name_tokenized)

        # Weighted sum sim
        combined_sims = {}
        for candle_id in set(review_sims.keys()) | set(desc_sims.keys()) | set(name_sims.keys()):
            rev_sim = review_sims.get(candle_id, 0)
            desc_sim = desc_sims.get(candle_id, 0)
            name_sim = name_sims.get(candle_id, 0)

            w1, w2, w3 = self.get_optimal_weights(name_sim, rev_sim, desc_sim)
            # w1, w2, w3 = 0.9, 0.6, 0.6
            combined_sims[candle_id] = (w1 * name_sim) + (w2 * rev_sim) + (w3 * desc_sim)

        # Pretty print similarities
        # print("\nSimilarity Scores by Candle:")
        # print("-" * 50)
        # for candle_id in combined_sims.keys():
        #     print(f"\nCandle {candle_id} ({self.candles.loc[candle_id, 'name']}):")
        #     print(f"Review similarity: {review_sims.get(candle_id, 0):.3f}")
        #     print(f"Description similarity: {desc_sims.get(candle_id, 0):.3f}")
        #     print(f"Name similarity: {name_sims.get(candle_id, 0):.3f}")
        #     print(f"Combined similarity: {combined_sims[candle_id]:.3f}")

        sorted_candle_ids = sorted(combined_sims.keys(), key=lambda cid: combined_sims[cid], reverse=True)
        result_df = self.candles.iloc[sorted_candle_ids].copy()
        result_df['sim_score'] = [combined_sims[idx] for idx in sorted_candle_ids]

        return result_df
    
    def get_optimal_weights(self, name_sim, rev_sim, desc_sim):
        """
        Calculate optimal weights for combining name, review, and description 
        similarities, handling potentially negative similarity values.
        
        Parameters:
        name_sim (float): Name similarity score
        rev_sim (float): Review similarity score
        desc_sim (float): Description similarity score
        
        Returns:
        (w1, w2, w3): Tuple of optimal weights for name, review, description
        """
        
        abs_name = abs(name_sim)
        abs_rev = abs(rev_sim)
        abs_desc = abs(desc_sim)
        
        if abs_name < 1e-6 and abs_rev < 1e-6 and abs_desc < 1e-6:
            return (0.33, 0.33, 0.34)
        
        if name_sim == 1.0:
            return (1.0, 0, 0)
        
        total_abs = abs_name + abs_rev + abs_desc
        
        w1 = abs_name / total_abs
        w2 = abs_rev / total_abs
        w3 = abs_desc / total_abs
        
        alpha = 0.1  # smoothing factor
        w1 = (1 - alpha) * w1 + alpha/3
        w2 = (1 - alpha) * w2 + alpha/3
        w3 = (1 - alpha) * w3 + alpha/3
        
        total = w1 + w2 + w3
        w1, w2, w3 = w1/total, w2/total, w3/total

        return (w1, w2, w3)
    
    def retrieve_top_k_candles_svd(self, query, k, w1=0.4, w2=0.3, w3=0.3):
        '''
        Returns the top k candles based on a custom similarity score:
        Similarity between a query and a candle is the weighted sum of 
        the cosine similarity between the query and the reviews, the
        cos sim between the query and the description, and the jaccard
        sim between the query and the candle name.

        If the query does not yield good distribution (i.e. highest and 
        lowest are not diff enough), then perform rocchio and redo?
        '''
        # print("count of all candles:", len(self.candles))

        # Review similarity
        modified_query = self.transform_query_svd(query, self.reviews_words_compressed)
        # review_sims_per_review = self.helper_cosine_sim(self.reviews_compressed_normed, modified_query)
        review_sims_per_review = self.reviews_compressed_normed.dot(modified_query)
        review_df = pd.DataFrame({
            'candle_id': [(self.review_idx_to_candle_idx[i] - 1) for i in range(len(review_sims_per_review))],
            'similarity': review_sims_per_review
        })
        review_sims = review_df.groupby('candle_id')['similarity'].mean().to_dict()
        # print("rev sims", review_sims, len(review_sims))
        # print()

        # Description similarity 
        modified_query = self.transform_query_svd(query, self.descriptions_words_compressed)
        # desc_sims_list = self.helper_cosine_sim(self.descriptions_compressed_normed, modified_query)
        desc_sims_list = self.descriptions_compressed_normed.dot(modified_query)
        desc_sims = {i: sim for i, sim in enumerate(desc_sims_list)}
        # print("desc sims", desc_sims, len(desc_sims))

        # Name similarity
        name_sims = {}
        query_tokenized = self.generic_tokenizer(query)
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
    
    # def svd_dim_labels(self, candle_id, k=12, top_n_words=1):
    #     '''
    #     returns words and each word's associated value of top k svd dimensions for given candle
    #     '''
    #     vocab = np.array(self.tfidf_vectorizer.get_feature_names_out())
    #     svd_matrix = self.all_words_compressed
    #     candle_svd_vec = self.all_compressed_normed[candle_id]

    #     top_indices = np.argsort(np.abs(candle_svd_vec))[-k:][::-1]
    #     results = []

    #     for dim in top_indices:
    #         top_pos_idx = np.argsort(svd_matrix[:, dim])[-top_n_words:][::-1]
    #         top_neg_idx = np.argsort(svd_matrix[:, dim])[:top_n_words]

    #         top_pos = [{"word": vocab[i], "value": round(svd_matrix[i, dim], 3)} for i in top_pos_idx]
    #         top_neg = [{"word": vocab[i], "value": round(svd_matrix[i, dim], 3)} for i in top_neg_idx]

    #         results.append({
    #             "dimension": f"dim{dim}",
    #             "value": round(candle_svd_vec[dim], 3),
    #             "top_positive": top_pos,
    #             "top_negative": top_neg
    #         })

    #     return results

    def svd_dim_labels_values(self, candle_id, k=12):
        '''
        returns labels and values of top k svd dimensions for given candle
        '''

        svd_labels = {
            0: "Fruity & Refreshing",
            1: "Floral & Fresh",
            2: "Tropical",
            3: "Sweet & Comforting",
            4: "Tropical Fruits",
            5: "Spicy & Sweet",
            6: "Cozy & Relaxing",
            7: "Citrus & Fresh",
            8: "Festive & Warm",
            9: "Winter Holiday Spice",
            10: "Baking Spices & Sweets",
            11: "Clean & Fresh"
        }

        candle_svd_vec = self.all_compressed_normed[candle_id]
        top_indices = np.argsort(np.abs(candle_svd_vec))[-k:][::-1]

        results = []
        for dim in top_indices:
            results.append({
                "label": svd_labels.get(dim),
                "value": round(candle_svd_vec[dim], 3)
            })

        return results

    def svd_dim_words_values(self, words_compressed, k=12, top_n_words=18):
    # prints top_n_words + values for each of the k dimensions. this is across the entire corpus, not per candle
        for dim in range(k):
            dim_vector = words_compressed[:, dim]
            top_word_indices = np.argsort(np.abs(dim_vector))[-top_n_words:][::-1]
            
            print(f"dim {dim}:")
            for i in top_word_indices:
                if i in self.index_to_word:
                    word = self.index_to_word[i]
                    value = dim_vector[i]
                    print(f"  {word}: {value}")
    
    def get_top_n_candle_dimensions(self, candle_id, n=5):
        """
        Returns the top N words associated with a specific candle based on SVD dimensions.
        
        Args:
            candle_id: The ID of the candle to analyze
            top_n: Number of top words to return (default: 5)
        
        Returns:
            List of top words associated with this candle
        """
        candle_svd_vec = self.all_compressed_normed[candle_id]
        sims = self.all_words_compressed.dot(candle_svd_vec)
        asort = np.argsort(-sims)[:n]
        return [(self.index_to_word[i], sims[i]) for i in asort]

    # NO SVD BAD BAD
    def retrieve_top_k_candles(self, query, k, w1=0.4, w2=0.2, w3=0.2):
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
            cand_name_tokenized = self.generic_tokenizer(self.candles.loc[i, 'name'])
            name_sims[i] = self.helper_jaccard_sim(query_tokenized, cand_name_tokenized)
        
        combined_sims = {}
        for candle_id in set(review_sims.keys()) | set(desc_sims.keys()) | set(name_sims.keys()):
            rev_sim = review_sims.get(candle_id, 0)
            desc_sim = desc_sims.get(candle_id, 0)
            name_sim = name_sims.get(candle_id, 0)
            combined_sims[candle_id] = (w1 * name_sim) + (w2 * rev_sim) + (w3 * desc_sim)
        
        sorted_candle_ids = sorted(combined_sims.keys(), key=lambda cid: combined_sims[cid], reverse=True)
        sorted_combined_sims = sorted(combined_sims.items(), key=lambda item: item[1], reverse=True)

        
        # Return top k unique candles
        top_k_ids = sorted_candle_ids[:k]
        top_k_sims = sorted_combined_sims[:k]
        # print(top_k_sims)
        
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
        query_vec = self.transform_query(query) if isinstance(query, str) else query
    
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
    
    def find_n_similar_candles(self, candle_id, n=3):
        candle_name = self.candles.loc[candle_id, 'name']
        candle_svd_vec = self.all_compressed_normed[candle_id]
        ranked_candles = self.all_compressed_normed.dot(candle_svd_vec)
        asort = np.argsort(-ranked_candles)
        return [{'name': self.candles.loc[i, 'name'], 'score': ranked_candles[i]} for i in asort if self.candles.loc[i, 'name'] != candle_name][:n]