import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
from scipy.sparse import csr_matrix

# Read the dataset
movies = pd.read_csv('dataset.csv')

# Feature selection (include 'votecount' in feature selection)
movies['tags'] = movies['overview'].fillna('') + ' ' + movies['genre'].fillna('') + ' ' + movies['votecount'].fillna('').astype(str)

# Vectorize 'tags' using CountVectorizer
cv = CountVectorizer(max_features=10000, stop_words='english')
vector = cv.fit_transform(movies['tags'].values.astype('U'))

# Convert vector to CSR matrix
vector_csr = csr_matrix(vector)

# Calculate cosine similarity
similarity = cosine_similarity(vector_csr)

# Save data and similarity matrix to files
pickle.dump(movies, open('movies_list.pkl', 'wb'))
pickle.dump(similarity, open('similarity.pkl', 'wb'))

# Check if 'votecount' column is included in the processed data
print("\nColumns after feature selection:")
print(movies.columns)

# Check the first few rows of the processed data to verify the 'tags' column
print("\nFirst few rows of processed data:")
print(movies.head())

# Load saved data and similarity matrix
loaded_data = pickle.load(open('movies_list.pkl', 'rb'))
loaded_similarity = pickle.load(open('similarity.pkl', 'rb'))

# Check the loaded data
print("\nLoaded data:")
print(loaded_data.head())
