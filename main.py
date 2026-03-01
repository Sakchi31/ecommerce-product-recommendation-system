import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset
df = pd.read_csv("content_based_recommendation_dataset.csv")

# Combine selected features
df["combined_features"] = (
    df["Brand"].astype(str) + " " +
    df["Season"].astype(str) + " " +
    df["Geography"].astype(str) + " " +
    df["Gender"].astype(str)
)

# Convert text data to numerical data
vectorizer = TfidfVectorizer(stop_words="english")
tfidf_matrix = vectorizer.fit_transform(df["combined_features"])

# Compute similarity matrix
similarity = cosine_similarity(tfidf_matrix)

def recommend(product_index):
    scores = list(enumerate(similarity[product_index]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)

    print("\nRecommended Products:\n")
    for i in scores[1:6]:
        print(df.iloc[i[0]]["Brand"])

# Example
recommend(0)
