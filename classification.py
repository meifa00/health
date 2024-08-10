import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

openai.api_key = st.secrets["mykey"]

# Load the dataset
def load_data():
    data = pd.read_csv('qa_dataset_with_embeddings.csv')
    return data

# Load the pre-calculated embeddings
def load_embeddings(data):
    embeddings = np.array(data['Question_Embedding'].apply(lambda x: np.fromstring(x[1:-1], sep=' ')).tolist())
    return embeddings

# Initialize the embedding model
def initialize_model():
    model = SentenceTransformer('all-MiniLM-L6-v2')
    return model

# Streamlit app
def main():
    st.title("Health Q&A System")

    # Load data and embeddings
    data = load_data()
    embeddings = load_embeddings(data)

    # Initialize the embedding model
    model = initialize_model()

    # User input for question
    user_question = st.text_input("Ask a question about heart, lung, or blood-related health:")

    # Button to search for answers
    if st.button("Get Answer"):
        if user_question:
            # Generate embedding for user question
            user_embedding = model.encode([user_question])

            # Calculate cosine similarity
            similarities = cosine_similarity(user_embedding, embeddings)

            # Find the most similar question
            most_similar_idx = np.argmax(similarities)
            most_similar_score = similarities[0][most_similar_idx]

            # Define a threshold for similarity score
            threshold = 0.7

            if most_similar_score > threshold:
                answer = data.iloc[most_similar_idx]['Answer']
                st.subheader("Answer:")
                st.write(answer)
                st.write(f"Similarity Score: {most_similar_score:.2f}")
            else:
                st.write("I apologize, but I don't have information on that topic yet. Could you please ask other questions?")
        else:
            st.write("Please enter a question.")

if __name__ == "__main__":
    main()
