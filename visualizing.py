import streamlit as st
import pickle
import requests
import numpy as np
import matplotlib.pyplot as plt
from fuzzywuzzy import process
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
import streamlit.components.v1 as components






# Function to fetch movie poster using TMDB API
def fetch_poster(movie_id):
    url = "https://api.themoviedb.org/3/movie/{}?api_key=c7ec19ffdd3279641fb606d19ceb9bb1&language=en-US".format(movie_id)
    data = requests.get(url).json()
    poster_path = data['poster_path']
    full_path = "https://image.tmdb.org/t/p/w500/" + poster_path
    return full_path






# Load movie data and similarity matrix
movies = pickle.load(open("movies_list.pkl", 'rb'))
similarity = pickle.load(open("similarity.pkl", 'rb'))
movies_list = movies['title'].values

# Load TF-IDF vectorizer and KNN model
tfidf_vectorizer = pickle.load(open('tfidf_vectorizer.pkl', 'rb'))
knn_model = pickle.load(open('knn_model.pkl', 'rb'))

imageCarouselComponent = components.declare_component("image-carousel-component", path="frontend/public")

imageUrls = [
    fetch_poster(507569),
    fetch_poster(635302),
    fetch_poster(225745),
    fetch_poster(42994),
    fetch_poster(429422),
    fetch_poster(9722),
    fetch_poster(13972),
    fetch_poster(240),
    fetch_poster(155),
    fetch_poster(598),
    fetch_poster(914),
    fetch_poster(255709),
    fetch_poster(572154)
]

imageCarouselComponent(imageUrls=imageUrls, height=200) 



# Function to recommend similar movies using KNN
def recommend(movie):
    if movie and movie.lower() in map(str.lower, movies['title'].values):
        index = movies[movies['title'].str.lower() == movie.lower()].index[0]
        distances, indices = knn_model.kneighbors(tfidf_vectorizer.transform([movies.iloc[index]['tags']]))

        # Get similar movies
        similar_movies = movies.iloc[indices[0]].copy()
        similar_movies['distance'] = distances.flatten()

        # Fetch 'votecount' from original dataset
        similar_movies['votecount'] = movies.iloc[indices[0]]['votecount']

        # Calculate similarity
        similar_movies['similarity'] = 1 - similar_movies['distance']

        # Sort by vote count and distance
        similar_movies = similar_movies.sort_values(by=['votecount', 'distance'], ascending=[False, True]) 

        # Filter out movies with low similarity
        similar_movies = similar_movies[similar_movies['distance'] <= 10]  # Adjust similarity threshold as needed

        # Select top recommendations
        recommend_movie = similar_movies['title'].tolist()
        recommend_poster = [fetch_poster(movie_id) for movie_id in similar_movies.id.tolist()]
        return recommend_movie, recommend_poster, similar_movies
    else:
        return [], [], []


# Function for fuzzy search
def fuzzy_search(movie):
    choices = movies['title'].values
    match, score = process.extractOne(movie, choices)
    if score >= 80:  # You can adjust the threshold as needed
        return match
    else:
        return None

# Function to visualize recommendations
def visualize_recommendations(movie_name, movie_poster, similar_movies):
    st.header("Visualization of Recommendations")
    if similar_movies.empty:
        st.write("No recommendations to visualize.")
        return
    
    # Display the selected movie and its tags
    st.subheader("Selected Movie:")
    st.text(movie_name)
    st.image(movie_poster, width=150, caption=movie_name)
    st.subheader("Tags:")
    st.text(similar_movies.iloc[0]['tags'])

    # Display the KNN graph
    st.subheader("K Nearest Neighbors:")
    fig, ax1 = plt.subplots()

    # Create twin Axes sharing the xaxis
    ax2 = ax1.twiny()
    ax3 = ax1.twiny()

    index = np.arange(len(similar_movies))
    bar_width = 0.3
    opacity = 0.8

    # Plot distances
    ax1.barh(index, similar_movies['distance'], bar_width, alpha=opacity, color='b', label='Distances')

    # Plot vote counts
    ax2.barh(index + bar_width, similar_movies['votecount'], bar_width, alpha=opacity, color='g', label='Vote Counts')

    # Plot similarity
    ax3.barh(index + 2*bar_width, similar_movies['similarity'], bar_width, alpha=opacity, color='r', label='Similarity')

    ax1.set_xlabel('Distance')
    ax2.set_xlabel('Vote Count')
    ax3.set_xlabel('Similarity')
    ax1.set_ylabel('Movie')
    ax1.set_title(f'Distances, Vote Counts, and Similarities for Similar Movies of {movie_name}')
    ax1.set_yticks(index + bar_width)
    ax1.set_yticklabels(similar_movies['title'])

    ax1.legend(loc='upper right')
    ax2.legend(loc='upper right')
    ax3.legend(loc='upper right')

    st.pyplot(fig)

# Streamlit app header
st.header("Movie Recommender System")

# Radio button for search option
search_option = st.radio("Choose search option:", ("Manual Search", "Dropdown"))

# Manual search text input
if search_option == "Manual Search":
    manual_search = st.text_input("Enter movie title for manual search:")
    selectvalue = None  # Set selectvalue to None for later check
else:
    manual_search = None  # Set manual_search to None for later check
    selectvalue = st.selectbox("Select movie from dropdown", movies_list)

# Button to show recommendations
if st.button("Show Recommendations"):
    if selectvalue:
        movie_name, movie_poster, similar_movies = recommend(selectvalue)
    elif manual_search:
        movie_name, movie_poster, similar_movies = recommend(manual_search.lower())

    if movie_name:
        # Display recommended movies and posters
        for name, poster in zip(movie_name, movie_poster):
            st.text(name)
            st.image(poster, width=150, caption=name)
        
        # Visualize recommendations
        visualize_recommendations(movie_name[0], movie_poster[0], similar_movies)

# Button to show recommendations for manual search
if st.button("Show Recommendations (Manual Search)"):
    if manual_search:
        movie_name_manual, movie_poster_manual, similar_movies_manual = recommend(manual_search.lower())
    elif selectvalue:
        movie_name_manual, movie_poster_manual, similar_movies_manual = recommend(selectvalue)

    if not movie_name_manual:
        # If not found, try fuzzy search
        fuzzy_match = fuzzy_search(manual_search)
        if fuzzy_match:
            st.write(f"Did you mean: {fuzzy_match}?")
            movie_name_manual, movie_poster_manual, similar_movies_manual = recommend(fuzzy_match.lower())
    
    if movie_name_manual:
        # Display recommended movies and posters for manual search
        for name, poster in zip(movie_name_manual, movie_poster_manual):
            st.text(name)
            st.image(poster, width=150, caption=name)
        
        # Visualize recommendations
        visualize_recommendations(movie_name_manual[0], movie_poster_manual[0], similar_movies_manual)
