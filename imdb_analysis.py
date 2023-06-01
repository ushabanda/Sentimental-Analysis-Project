# Load the libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelBinarizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from wordcloud import WordCloud, STOPWORDS
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize, sent_tokenize
from bs4 import BeautifulSoup
import spacy
import re
import string
import unicodedata
from nltk.tokenize.toktok import ToktokTokenizer
from nltk.stem import LancasterStemmer, WordNetLemmatizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from textblob import TextBlob
from textblob import Word
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

import os
# print("\n\n\n",os.listdir("../input"))
import warnings
warnings.filterwarnings('ignore')

path = r'C:\Users\Administrator\Downloads\archive\IMDB_Dataset.csv'
imdb_data = pd.read_csv(path)

# print("\n\n\n",imdb_data.shape)
# print("\n\n\n\n =========  review ============",imdb_data.review[9], "\n\n\n")
# print("\n\n\n\n ============ sentiment  =========",imdb_data.sentiment[9], "\n\n\n")
print("\n\n\n\n ============ sentiment  =========",imdb_data['sentiment'].value_counts())
print("\n\n\n\n ============ positive sentiment count =========",imdb_data['sentiment'].value_counts()['positive'])
print("\n\n\n\n ============ negative sentiment count  =========",imdb_data['sentiment'].value_counts()['negative'])

if imdb_data['sentiment'].value_counts()['negative'] > imdb_data['sentiment'].value_counts()['positive']:
    print("\n\n\n\n ============ overall NEGATIVE sentiment  =========")
elif imdb_data['sentiment'].value_counts()['positive'] > imdb_data['sentiment'].value_counts()['negative']:
    print("\n\n\n\n ============ overall POSITIVE sentiment  =========")
else:
    print("\n\n\n\n ============ overall SAME sentiment  =========")

print("\n\n\n\n =====================", imdb_data.head(10), "\n\n\n")
