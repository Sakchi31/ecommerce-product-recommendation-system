# Ecommerce Product Recommendation System

This project is a Content-Based Recommendation System built using Python and Machine Learning techniques.

## Project Overview
The system recommends similar products based on product features like:
- Brand of the product
- Season
- Geographical locations
- Gender

It uses TF-IDF Vectorization and Cosine Similarity to calculate similarity between products.

## Technologies Used
- Python
- Pandas
- Scikit-learn

## Project Structure
ecommerce-product-recommendation-system
│
├── content_based_recommendation_dataset.csv
├── main.py
├── app.py       
├── requirements.txt
└── README.md

## How to Run
1. Install dependencies:
   pip install -r requirements.txt

2. Run Terminal Version:
   python main.py

3. Run Web App Version (Streamlit):
   streamlit run app.py

## Future Improvements
- Add Streamlit web interface
- Improve recommendation accuracy
- Deploy the project online
