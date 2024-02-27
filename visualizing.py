import streamlit as st
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors

# Load movie data and similarity matrix
movies = pickle.load(open("movies_list.pkl", 'rb'))
similarity = pickle.load(open("similarity.pkl", 'rb'))

# Load TF-IDF vectorizer and KNN model
tfidf_vectorizer = pickle.load(open('tfidf_vectorizer.pkl', 'rb'))
knn_model = pickle.load(open('knn_model.pkl', 'rb'))

# Streamlit app header
st.header("KNN Graph Visualization")

# Select a movie
selected_movie = st.selectbox("Select a movie", movies['title'].values)

# Get the movie index
movie_index = movies[movies['title'] == selected_movie].index[0]

# Get the feature vector for the selected movie
feature_vector = tfidf_vectorizer.transform([movies.iloc[movie_index]['tags']])

# Find k nearest neighbors
k = 5  # You can adjust k as needed
distances, indices = knn_model.kneighbors(feature_vector, n_neighbors=k)

# Extract movie titles and distances
neighbor_movie_titles = movies.iloc[indices[0]]['title'].tolist()
neighbor_distances = distances[0]
neighbor_similarities = 1 - neighbor_distances  # Similarity is the inverse of distance

# Display the selected movie
st.write("Selected Movie:", selected_movie)

# Display the KNN graph
st.write("K Nearest Neighbors:")
fig, ax = plt.subplots()
bar_width = 0.35
index = np.arange(k)
opacity = 0.8

# Plot distances
ax.barh(index, neighbor_distances[::-1], bar_width, alpha=opacity, color='b', label='Distances')

# Plot similarities
ax.barh(index + bar_width, neighbor_similarities[::-1], bar_width, alpha=opacity, color='r', label='Similarities')

ax.set_xlabel('Value')
ax.set_ylabel('Movie')
ax.set_title(f'Distances and Similarities for K Nearest Neighbors of {selected_movie}')
ax.set_yticks(index + bar_width / 2)
ax.set_yticklabels(neighbor_movie_titles[::-1])
ax.legend()

st.pyplot(fig)
