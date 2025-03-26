import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from scipy.sparse.linalg import svds
import sklearn
from models.candle import Candle

class MLModel:
    def __init__(self):
        self.model = None
    
    def train(self):
        return