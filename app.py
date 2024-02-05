import streamlit as st
import pickle
import requests
from fuzzywuzzy import process

def fetch_poster(movie_id):
    url = "https://api.themoviedb.org/3/movie/{}?api_key=c7ec19ffdd3279641fb606d19ceb9bb1&language=en-US".format(movie_id)
    data = requests.get(url).json()
    poster_path = data['poster_path']
    full_path = "https://image.tmdb.org/t/p/w500/" + poster_path
    return full_path

movies = pickle.load(open("movies_list.pkl", 'rb'))
similarity = pickle.load(open("similarity.pkl", 'rb'))
movies_list = movies['title'].values

st.header("Movie Recommender System")

import streamlit.components.v1 as components

imageCarouselComponent = components.declare_component("image-carousel-component", path="frontend/public")

imageUrls = [
    fetch_poster(1632),
    fetch_poster(299536),
    fetch_poster(17455),
    fetch_poster(2830),
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

search_option = st.radio("Choose search option:", ("Manual Search", "Dropdown"))

if search_option == "Manual Search":
    manual_search = st.text_input("Enter movie title for manual search:")
    selectvalue = None  # Set selectvalue to None for later check
else:
    manual_search = None  # Set manual_search to None for later check
    selectvalue = st.selectbox("Select movie from dropdown", movies_list)

def recommend(movie):
    if movie and movie.lower() in map(str.lower, movies['title'].values):
        index = movies[movies['title'].str.lower() == movie.lower()].index[0]
        distance = sorted(list(enumerate(similarity[index])), reverse=True, key=lambda vector: vector[1])
        recommend_movie = []
        recommend_poster = []
        for i in distance[1:11]:
            movies_id = movies.iloc[i[0]].id
            recommend_movie.append(movies.iloc[i[0]].title)
            recommend_poster.append(fetch_poster(movies_id))
        return recommend_movie, recommend_poster
    else:
        return [], []

def fuzzy_search(movie):
    choices = movies['title'].values
    match, score = process.extractOne(movie, choices)
    if score >= 80:  # You can adjust the threshold as needed
        return match
    else:
        return None

if st.button("Show Recommend"):
    if selectvalue:
        movie_name, movie_poster = recommend(selectvalue)
    elif manual_search:
        movie_name, movie_poster = recommend(manual_search.lower())

    if movie_name:
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.text(movie_name[0])
            st.image(movie_poster[0], width=150, caption=movie_name[0])
        with col2:
            st.text(movie_name[1])
            st.image(movie_poster[1], width=150, caption=movie_name[1])
        with col3:
            st.text(movie_name[2])
            st.image(movie_poster[2], width=150, caption=movie_name[2])
        with col4:
            st.text(movie_name[3])
            st.image(movie_poster[3], width=150, caption=movie_name[3])
        with col5:
            st.text(movie_name[4])
            st.image(movie_poster[4], width=150, caption=movie_name[4])
      
    

if st.button("Show Recommend (Manual Search)"):
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
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.text(movie_name_manual[0])
            st.image(movie_poster_manual[0], width=150, caption=movie_name_manual[0])
        with col2:
            st.text(movie_name_manual[1])
            st.image(movie_poster_manual[1], width=150, caption=movie_name_manual[1])
        with col3:
            st.text(movie_name_manual[2])
            st.image(movie_poster_manual[2], width=150, caption=movie_name_manual[2])
        with col4:
            st.text(movie_name_manual[3])
            st.image(movie_poster_manual[3], width=150, caption=movie_name_manual[3])
        with col5:
            st.text(movie_name_manual[4])
            st.image(movie_poster_manual[4], width=150, caption=movie_name_manual[4])
       