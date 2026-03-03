import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Title
st.title("🛍 Ecommerce Product Recommendation System")

# Load dataset
df = pd.read_csv("content_based_recommendation_dataset.csv")

# Combine features
df["combined_features"] = (
    df["Brand of the product"].astype(str) + " " +
    df["Season"].astype(str) + " " +
    df["Geographical locations"].astype(str) + " " +
    df["Gender"].astype(str)
)

# TF-IDF
vectorizer = TfidfVectorizer(stop_words="english")
tfidf_matrix = vectorizer.fit_transform(df["combined_features"])

# Similarity
similarity = cosine_similarity(tfidf_matrix)

# Dropdown for selecting product
product_index = st.number_input("Enter Product Index", min_value=0, max_value=len(df)-1, step=1)

if st.button("Recommend"):
    scores = list(enumerate(similarity[product_index]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)

    st.subheader("Recommended Products:")
    
    for i in scores[1:6]:
        product = df.iloc[i[0]]
        st.write("Brand:", product["Brand of the product"])
        st.write("Season:", product["Season"])
        st.write("Location:", product["Geographical locations"])
        st.write("Gender:", product["Gender"])
        st.write("---")
