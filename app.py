%%writefile app.py
import pandas as pd
import logging
from tqdm import tqdm
from rapidfuzz import process
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

logging.basicConfig(level=logging.INFO)

@st.cache_data
def load_data(filepath):
    df = pd.read_excel(filepath)
    selected_features = ['genres', 'keywords', 'tagline', 'cast', 'director']
    df[selected_features] = df[selected_features].fillna('')
    tqdm.pandas(desc="Combining features")
    df['combined_features'] = df[selected_features].progress_apply(lambda row: ' '.join(row), axis=1)
    return df

@st.cache_data
def compute_similarity(df):
    vectorizer = TfidfVectorizer()
    feature_vectors = vectorizer.fit_transform(df['combined_features'])
    return cosine_similarity(feature_vectors)

def get_best_match(movie_name, all_titles):
    match, score, idx = process.extractOne(movie_name, all_titles, score_cutoff=60)
    return match if score else None

def get_recommendations(df, similarity_matrix, movie_name, num_recommendations=10):
    all_titles = df['title'].astype(str).tolist()
    best_match = get_best_match(movie_name, all_titles)
    if not best_match:
        return f"No close match found for '{movie_name}'.", []
    movie_index = df[df['title'] == best_match].index[0]
    similarity_scores = list(enumerate(similarity_matrix[movie_index]))
    sorted_movies = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    recommended_titles = []
    for idx, score in sorted_movies:
        if idx != movie_index and len(recommended_titles) < num_recommendations:
            recommended_titles.append(df.iloc[idx]['title'])
    return f"Top {num_recommendations} movies similar to '{best_match}':", recommended_titles

def main():
    st.title("🎬 Movie Recommender in Colab")

    filepath = '/content/movies.xlsx'
    try:
        df = load_data(filepath)
        similarity_matrix = compute_similarity(df)

        movie_name = st.text_input("Enter your favorite movie name:")
        num_recommendations = st.slider("Number of Recommendations", 5, 50, 10)

        if st.button("Get Recommendations"):
            st.write("⏳ Finding matches...")
            message, recommendations = get_recommendations(df, similarity_matrix, movie_name, num_recommendations)
            st.success(message)
            for i, title in enumerate(recommendations, start=1):
                st.write(f"{i}. {title}")
    except FileNotFoundError:
        st.error("❌ movies.xlsx not found in /content. Please upload it first.")

if __name__ == "__main__":
    main()
