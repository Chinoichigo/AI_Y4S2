import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

# Load your dataset containing 'overview' and 'genre' columns
movies_data = pd.read_csv('dataset.csv')

# Combine 'overview' and 'genre' columns to create 'tags' column
movies_data['tags'] = movies_data['overview'].fillna('') + ' ' + movies_data['genre'].fillna('')

# Training TF-IDF Vectorizer
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(movies_data['tags'])

# Save TF-IDF vectorizer to file
with open('tfidf_vectorizer.pkl', 'wb') as file:
    pickle.dump(tfidf_vectorizer, file)

# Training KNN Model
knn_model = NearestNeighbors(n_neighbors=5, algorithm='brute', metric='cosine')
knn_model.fit(tfidf_matrix)

# Save KNN model to file
with open('knn_model.pkl', 'wb') as file:
    pickle.dump(knn_model, file)
