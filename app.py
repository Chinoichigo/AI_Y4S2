import streamlit as st
import pickle
import requests
from fuzzywuzzy import process
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

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

        # Sort by distance and vote count
        similar_movies = similar_movies.sort_values(by=['distance', 'votecount'], ascending=[True, False])

        # Filter out movies with low similarity
        similar_movies = similar_movies[similar_movies['distance'] <= 3]  # Adjust similarity threshold as needed

        # Select top recommendations
        recommend_movie = similar_movies['title'].tolist()
        recommend_poster = [fetch_poster(movie_id) for movie_id in similar_movies.id.tolist()]
        return recommend_movie, recommend_poster
    else:
        return [], []



# Function for fuzzy search
def fuzzy_search(movie):
    choices = movies['title'].values
    match, score = process.extractOne(movie, choices)
    if score >= 80:  # You can adjust the threshold as needed
        return match
    else:
        return None

# Button to show recommendations
if st.button("Show Recommendations"):
    if selectvalue:
        movie_name, movie_poster = recommend(selectvalue)
    elif manual_search:
        movie_name, movie_poster = recommend(manual_search.lower())

    if movie_name:
        # Display recommended movies and posters
        for name, poster in zip(movie_name, movie_poster):
            st.text(name)
            st.image(poster, width=150, caption=name)

# Button to show recommendations for manual search
if st.button("Show Recommendations (Manual Search)"):
    if manual_search:
        movie_name_manual, movie_poster_manual = recommend(manual_search.lower())
    elif selectvalue:
        movie_name_manual, movie_poster_manual = recommend(selectvalue)

    if not movie_name_manual:
        # If not found, try fuzzy search
        fuzzy_match = fuzzy_search(manual_search)
        if fuzzy_match:
            st.write(f"Did you mean: {fuzzy_match}?")
            movie_name_manual, movie_poster_manual = recommend(fuzzy_match.lower())
    
    if movie_name_manual:
        # Display recommended movies and posters for manual search
        for name, poster in zip(movie_name_manual, movie_poster_manual):
            st.text(name)
            st.image(poster, width=150, caption=name)
