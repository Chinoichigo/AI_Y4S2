import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
import pickle

# Read the dataset
movies = pd.read_csv('dataset.csv')

# Check the columns in the dataset
print("Columns in the dataset:")
print(movies.columns)

# Ensure the 'votecount' column is included in the DataFrame
if 'votecount' not in movies.columns:
    raise ValueError("The 'votecount' column is missing from the dataset.")

# Feature selection (include 'votecount' in feature selection)
movies = movies[['id', 'title', 'overview', 'genre', 'votecount']]

# Combine 'overview' and 'genre' columns to create 'tags' column
movies['tags'] = movies['overview'].fillna('') + ' ' + movies['genre'].fillna('') + ' ' + movies['votecount'].fillna('').astype(str)

# Training TF-IDF Vectorizer
print("Training TF-IDF Vectorizer...")
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(movies['tags'])

# Save TF-IDF vectorizer to file
print("Saving TF-IDF Vectorizer to file...")
with open('tfidf_vectorizer.pkl', 'wb') as file:
    pickle.dump(tfidf_vectorizer, file)

# Training KNN Model
print("Training KNN Model...")
knn_model = NearestNeighbors(n_neighbors=5, algorithm='brute', metric='cosine')
knn_model.fit(tfidf_matrix)

# Save KNN model to file
print("Saving KNN Model to file...")
with open('knn_model.pkl', 'wb') as file:
    pickle.dump(knn_model, file)

print("Training and saving completed successfully!")
