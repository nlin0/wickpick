# 🕯️ WickPick

**Team Members:**  
Nicole Lin • Jolly Zheng • Elaine Wu • Asen Ou • Joanna Lin

[**Live Demo**](http://4300showcase.infosci.cornell.edu:5253/)

## 🌟 Overview

WickPick is an NLP-powered web application that recommends candles based on a user’s descriptive mood or scent query.   Users can type something like “**cozy autumn night**” or “**fresh and clean after a storm**,” and WickPick will return candles that best match the vibe. This works even if you misspell words like “chrismas” or “lamon.”

Our goal was to make search by feeling possible. We wanted to translate human emotions and vague scent descriptions into accurate, data-driven product matches. WickPick combines information retrieval, text mining, and natural language processing techniques.

| Example Input | Top Recommendations | Why It Works |
|----------------|---------------------|---------------|
| **“calm and serene after a long day”** | Lavender Vanilla, Catching Rays, Breeze | Uses mood-based keywords + cosine similarity |
| **“kitchen dessert”** | Kitchen Spice, Jelly Beans, Cinnamon Stick | Matches food-related terms and co-occurrence patterns |
| **“tropical beach citrus”** | Bahama Breeze, Coconut Beach, Sicilian Lemon | Captures summer/fruity scent clusters |
| **“tangerine & vanilla”** | Tangerine & Vanilla, Vanilla Cupcake, Lemon Lavender | Identifies exact match + semantic relatives |

## 🔍 What Makes WickPick Unique

**Intelligent Recommendations:**  
WickPick analyzes candle names, descriptions, and real user reviews using machine learning and natural language processing.

**Smart Search Algorithms:**  
We combined TF-IDF, Cosine Similarity, Jaccard Similarity, and Edit Distance to handle fuzzy matches and typos.  
To find deeper, hidden scent relationships, we added SVD (Singular Value Decomposition) for semantic dimensionality reduction.

**Adaptive Query Refinement (Rocchio Algorithm):**  
Our app *learns* from user feedback by adjusting queries based on relevant results and improving recommendations over time.

**Interactive Interface:**  
The front end displays results as clickable cards that open detailed pop-ups with similarity scores, SVD dimensions, and real candle reviews. Users can also filter by fragrance family like “Fresh & Clean” or “Floral.”

## 🧰 Tech Stack

| Layer | Tools & Technologies |
|-------|----------------------|
| **Frontend** | HTML, CSS, JavaScript, D3.js |
| **Backend** | Flask (Python) |
| **Data Processing & ML** | pandas, NumPy, scikit-learn, NLTK |
| **Algorithms Implemented** | TF-IDF, Cosine Similarity, Jaccard + Edit Distance, SVD, Rocchio |
| **Data Source** | Custom web-scraped dataset of 100+ Yankee Candles (names, descriptions, reviews, fragrance families) |
