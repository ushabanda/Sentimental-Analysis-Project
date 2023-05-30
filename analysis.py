from os import path
import pandas as pd

path = r'C:\Users\Administrator\Downloads\archive\IMDB_Dataset.csv'
df=pd.read_csv(path)
df.head()

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

vect = CountVectorizer()
docs = np.array(['i am usha working on sentimental analysis project'])
bag=vect.fit_transform(docs)
print(vect.vocabulary_)

print(bag.toarray())

from sklearn.feature_extraction.text import TfidfTransformer
np.set_printoptions(precision=2)
tfidf = TfidfTransformer(use_idf=True, norm='l2',smooth_idf=True)
print(tfidf.fit_transform(bag).toarray())

import nltk
nltk.download('stopwords')

from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(use_idf=True, norm='l2',smooth_idf=True)

y=df.sentiment